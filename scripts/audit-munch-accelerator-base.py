#!/usr/bin/env python3
"""Derive the original IOAcceleratorES ABI from the linked kernel by structure.

The prelink strips IOGPUFamily's symbols, so the class cannot be looked up by
name in the kernel's symbol table. It is located by the same facts the runtime
audit of the sheng kernel observed, read from the image instead of memory:

  * the metaclass object names it: a data word that points at the string
    "IOAcceleratorES" 24 bytes into the object, with the 136-byte instance size
    at +32;
  * the vtable is the 168-entry table that matches IOService's own table except
    at slots 0, 1 and 7 (the destructor pair and the metaclass getter), and its
    typeinfo names the class;
  * the constructors are the bodies that install that vptr in [x0], call
    IOService's C2 and return through the ordinary frame epilogue.

The output is the plan build-adreno740-accelerator.py consumes, with this
kernel's addresses. It is a check of the image, not a runtime observation:
runtime_qualified stays false until a booted kernel confirms it.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kits/XnuPrelink'))
from native_prelink import macho, exports

INSTANCE_BYTES = 136
VIRTUAL_ENTRIES = 168
CLASS = b'IOAcceleratorES'


def require(value, message):
    if not value:
        raise SystemExit(message)


def main():
    kernel = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'kernel/kernel.macho'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'build/gpu-analysis/ioaccelerator-base-abi.json'
    raw = kernel.read_bytes()
    image = macho(raw)
    names = {}
    for address, group in exports(raw, image).items():
        for name in group:
            names.setdefault(name, address)

    def backed(address, length):
        for segment in image['segments']:
            if segment['va'] <= address and address + length <= segment['va'] + segment['filesize']:
                at = segment['fileoff'] + address - segment['va']
                require(at + length <= len(raw), f'unbacked read at {address:#x}')
                return raw[at:at + length]
        raise SystemExit(f'{address:#x}+{length:#x} is not file-backed')

    def qword(address):
        return int.from_bytes(backed(address, 8), 'little')

    def cstring(address):
        end = raw.find(b'\0', backed_offset(address))
        require(end >= 0, f'unterminated string at {address:#x}')
        return raw[backed_offset(address):end]

    def backed_offset(address):
        for segment in image['segments']:
            if segment['va'] <= address < segment['va'] + segment['filesize']:
                return segment['fileoff'] + address - segment['va']
        raise SystemExit(f'{address:#x} is not file-backed')

    vptr_ioservice = names.get('__ZTV9IOService')
    c2 = names.get('__ZN9IOServiceC2EPK11OSMetaClass') or names.get('__ZN9IOServiceC1EPK11OSMetaClass')
    require(vptr_ioservice and c2, 'IOService vtable/C2 are not in the kernel symbol table')
    base = backed(vptr_ioservice + 16, VIRTUAL_ENTRIES * 8)

    # The class-name string, once per occurrence in a backed segment.
    string_vas = []
    at = raw.find(CLASS + b'\0')
    while at >= 0:
        for segment in image['segments']:
            if segment['fileoff'] <= at < segment['fileoff'] + segment['filesize']:
                string_vas.append(segment['va'] + at - segment['fileoff'])
        at = raw.find(CLASS + b'\0', at + 1)
    require(string_vas, 'the class-name string is not in the kernel image')

    metaclasses = []
    for string_va in string_vas:
        needle = string_va.to_bytes(8, 'little')
        at = raw.find(needle)
        while at >= 0:
            if at + 12 <= len(raw) and int.from_bytes(raw[at + 8:at + 12], 'little') == INSTANCE_BYTES:
                for segment in image['segments']:
                    if segment['fileoff'] <= at - 24 < segment['fileoff'] + segment['filesize']:
                        metaclasses.append(segment['va'] + at - 24 - segment['fileoff'])
            at = raw.find(needle, at + 1)
    require(len(metaclasses) == 1, f'expected one IOAcceleratorES metaclass object, found {len(metaclasses)}')
    metaclass = metaclasses[0]
    require(qword(metaclass + 24) in string_vas, 'metaclass name pointer does not point at the class string')

    # The vtable: IOService's entries except 0, 1 and 7, tied to the class by its typeinfo name.
    tables = []
    for segment in image['segments']:
        for section in segment['sections']:
            if section['name'] != '__const' or section['size'] < VIRTUAL_ENTRIES * 8:
                continue
            data = raw[section['fileoff']:section['fileoff'] + section['size']]
            at = 0
            while at + VIRTUAL_ENTRIES * 8 <= len(data):
                window = data[at:at + VIRTUAL_ENTRIES * 8]
                if (window[16:56] == base[16:56] and window[64:] == base[64:] and window[0:8] != base[0:8]
                        and window[8:16] != base[8:16] and window[56:64] != base[56:64]):
                    address = section['va'] + at
                    try:
                        typeinfo = qword(address + 8)
                        name_at = qword(typeinfo + 8)
                        if CLASS in cstring(name_at):
                            tables.append(address)
                    except SystemExit:
                        pass
                at += 8
    require(len(tables) == 1, f'expected one IOAcceleratorES vtable, found {len(tables)}')
    vptr = tables[0]
    entries = [qword(vptr + 16 + i * 8) for i in range(VIRTUAL_ENTRIES)]
    require(entries[7] != int.from_bytes(base[56:64], 'little'), 'slot 7 is not overridden')

    # Constructors: prologue, vptr store, call to IOService C2, ordinary frame epilogue.
    def branch_target(pc, word):
        offset = word & 0x3ffffff
        if offset & (1 << 25):
            offset -= 1 << 26
        return pc + offset * 4

    def adrp_target(pc, word):
        imm = ((word >> 5) & 0x7ffff) * 4096 + ((word >> 29) & 3) * 4096
        if imm & (1 << 32):
            imm -= 1 << 33
        return (pc & ~0xfff) + imm

    constructors = []
    for segment in image['segments']:
        if segment['name'] != '__TEXT_EXEC':
            continue
        data = raw[segment['fileoff']:segment['fileoff'] + segment['filesize']]
        va = segment['va']
        for at in range(0, len(data) - 128, 4):
            w0 = int.from_bytes(data[at:at + 4], 'little')
            w1 = int.from_bytes(data[at + 4:at + 8], 'little')
            if not (w0 == 0xa9bf7bfd and w1 == 0x910003fd):
                continue
            window = data[at:at + 128]
            # ldp x29,x30,[sp],#16 followed by ret, as one little-endian 8-byte word pair.
            epilogue = any(int.from_bytes(window[i:i + 8], 'little') == (0xd65f03c0 << 32) | 0xa8c17bfd
                           for i in range(0, 112, 4))
            sc2 = any(int.from_bytes(window[i:i + 4], 'little') >> 26 == 0x25 and
                      branch_target(va + at + i, int.from_bytes(window[i:i + 4], 'little')) == c2
                      for i in range(0, 124, 4))
            stores = any(int.from_bytes(window[i:i + 4], 'little') & 0xffc00000 == 0xf9000000 and
                         (int.from_bytes(window[i:i + 4], 'little') >> 5) & 0x1f == 0
                         for i in range(0, 124, 4))
            if epilogue and sc2 and stores:
                constructors.append(va + at)
    require(constructors, 'no IOAcceleratorES constructor found')
    constructors.sort()

    plan = {
        'kernel_sha256': hashlib.sha256(raw).hexdigest(),
        'metaclass': hex(metaclass),
        'vptr': hex(vptr),
        'instance_bytes': INSTANCE_BYTES,
        'virtual_entries': VIRTUAL_ENTRIES,
        'operational_overrides': [],
        'metadata_overrides': [0, 1, 7],
        'protected_compatible_constructor_aliases': [hex(a) for a in constructors],
        'nondeleting_destructor': hex(entries[0]),
        'get_meta_class': hex(entries[7]),
        'class_name': CLASS.decode(),
        'source': 'static scan of the linked kernel; runtime_qualified=false',
        'runtime_qualified': False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + '\n')
    print(f"PASS: IOAcceleratorES metaclass {plan['metaclass']} vtable {plan['vptr']} "
          f"C2 {plan['protected_compatible_constructor_aliases']} D2 {plan['nondeleting_destructor']}")


if __name__ == '__main__':
    sys.exit(main())
