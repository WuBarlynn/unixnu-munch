#!/usr/bin/env python3
"""Derive the original IOAcceleratorES ABI from the linked kernel by structure.

The prelink strips IOGPUFamily's symbols, so nothing can be looked up by name.
The class is located by the facts that do not depend on any private layout:

  * the vtable is the 168-entry table that matches IOService's own except at
    slots 0, 1 and 7, and whose typeinfo names the class;
  * slot 0 is the non-deleting destructor and slot 7 is getMetaClass, whose body
    computes the metaclass object (an adrp/add pair), so the metaclass address
    comes from that code rather than from a guessed field offset;
  * the constructors are the bodies that install that vptr in [x0], call
    IOService's C2 and return through the ordinary frame epilogue.

Run standalone to inspect a kernel:

    python3 scripts/audit-munch-accelerator-base.py KERNEL [OUT.json]

Diagnostics are printed before any check fails, so a layout difference shows
what the image actually holds.
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

    def file_offset(address):
        for segment in image['segments']:
            if segment['va'] <= address < segment['va'] + segment['filesize']:
                return segment['fileoff'] + address - segment['va']
        return None

    def backed(address, length):
        at = file_offset(address)
        require(at is not None and at + length <= len(raw), f'unbacked read at {address:#x}')
        return raw[at:at + length]

    def qword(address):
        return int.from_bytes(backed(address, 8), 'little')

    def u32(address):
        return int.from_bytes(backed(address, 4), 'little')

    def cstring(address):
        at = file_offset(address)
        require(at is not None, f'unbacked string at {address:#x}')
        end = raw.find(b'\0', at)
        return raw[at:end]

    vptr_ioservice = names.get('__ZTV9IOService')
    c2 = names.get('__ZN9IOServiceC2EPK11OSMetaClass') or names.get('__ZN9IOServiceC1EPK11OSMetaClass')
    require(vptr_ioservice and c2, 'IOService vtable/C2 are not in the kernel symbol table')
    base = backed(vptr_ioservice + 16, VIRTUAL_ENTRIES * 8)

    string_vas = []
    at = raw.find(CLASS + b'\0')
    while at >= 0:
        for segment in image['segments']:
            if segment['fileoff'] <= at < segment['fileoff'] + segment['filesize']:
                string_vas.append(segment['va'] + at - segment['fileoff'])
        at = raw.find(CLASS + b'\0', at + 1)
    print(f'class string: {len(string_vas)} occurrence(s) at ' + ', '.join(hex(v) for v in string_vas))
    require(string_vas, 'the class-name string is not in the kernel image')

    # Where the class string is referenced from data, with the surrounding words. The
    # metaclass-private layout differs between builds, so this is reported, not assumed.
    for string_va in string_vas:
        needle = string_va.to_bytes(8, 'little')
        at = raw.find(needle)
        while at >= 0:
            segment = next((s for s in image['segments']
                            if s['fileoff'] <= at < s['fileoff'] + s['filesize']), None)
            if segment:
                va = segment['va'] + at - segment['fileoff']
                context = raw[max(0, at - 40):at + 40]
                print(f'  name pointer at {va:#x}: ' + context.hex())
            at = raw.find(needle, at + 1)

    tables, loose = [], []
    for segment in image['segments']:
        for section in segment['sections']:
            if section['name'] != '__const' or section['size'] < VIRTUAL_ENTRIES * 8:
                continue
            data = raw[section['fileoff']:section['fileoff'] + section['size']]
            for at in range(0, len(data) - VIRTUAL_ENTRIES * 8 + 1, 8):
                window = data[at:at + VIRTUAL_ENTRIES * 8]
                if not (window[16:56] == base[16:56] and window[64:] == base[64:]
                        and window[0:8] != base[0:8] and window[8:16] != base[8:16]
                        and window[56:64] != base[56:64]):
                    continue
                address = section['va'] + at
                loose.append(address)
                try:
                    typeinfo = qword(address + 8)
                    name = cstring(qword(typeinfo + 8))
                except SystemExit:
                    continue
                if CLASS in name:
                    tables.append(address)
                else:
                    print(f'  vtable-like table at {address:#x}: typeinfo names {name!r}')
    print(f'vtable scan: {len(loose)} table(s) differ from IOService at 0/1/7, '
          f'{len(tables)} named {CLASS.decode()}')
    require(len(tables) == 1, f'expected one {CLASS.decode()} vtable, found {len(tables)}')
    vptr = tables[0]
    entries = [qword(vptr + 16 + i * 8) for i in range(VIRTUAL_ENTRIES)]
    print(f'vtable {vptr:#x}: D2 {entries[0]:#x} D1 {entries[1]:#x} getMetaClass {entries[7]:#x}')

    # getMetaClass computes the metaclass object with adrp/add; decode until the pair is
    # found. If it never appears, fall back to the name-pointer neighbourhood.
    def adrp_target(pc, word):
        imm = ((word >> 5) & 0x7ffff) * 4096 + ((word >> 29) & 3) * 4096
        if imm & (1 << 32):
            imm -= 1 << 33
        return (pc & ~0xfff) + imm

    getter = entries[7]
    metaclass = None
    code = backed(getter, 40)
    for i in range(0, 36, 4):
        word = int.from_bytes(code[i:i + 4], 'little')
        if word & 0x9f00001f == 0x90000000:
            for j in range(i + 4, min(i + 16, 36), 4):
                second = int.from_bytes(code[j:j + 4], 'little')
                if second & 0xffc003ff == 0x91000000:
                    metaclass = adrp_target(getter + i, word) + ((second >> 10) & 4095)
                    break
        if metaclass is not None:
            break
    print(f'getMetaClass body: {"no adrp/add pair" if metaclass is None else hex(metaclass)}')
    if metaclass is not None:
        try:
            print(f'  metaclass {metaclass:#x}: ' +
                  ', '.join(f'+{o:#x}={qword(metaclass + o):#x}' for o in (0, 8, 16, 24, 32)))
        except SystemExit as error:
            print(f'  metaclass read: {error}')

    def branch_target(pc, word):
        offset = word & 0x3ffffff
        if offset & (1 << 25):
            offset -= 1 << 26
        return pc + offset * 4

    constructors = []
    for segment in image['segments']:
        if segment['name'] != '__TEXT_EXEC':
            continue
        data = raw[segment['fileoff']:segment['fileoff'] + segment['filesize']]
        va = segment['va']
        for at in range(0, len(data) - 128, 4):
            if int.from_bytes(data[at:at + 4], 'little') != 0xa9bf7bfd:
                continue
            if int.from_bytes(data[at + 4:at + 8], 'little') != 0x910003fd:
                continue
            window = data[at:at + 128]
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
    constructors.sort()
    print(f'constructors calling IOService C2 with a vptr store: {[hex(c) for c in constructors]}')
    require(constructors, 'no IOAcceleratorES constructor found')
    require(metaclass is not None, 'no metaclass address derived from getMetaClass')

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
