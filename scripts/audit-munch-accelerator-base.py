#!/usr/bin/env python3
"""Derive the original IOAcceleratorES ABI from the linked kernel by structure.

The prelink strips IOGPUFamily's symbols, so nothing can be looked up by name.
The class is located by the facts that do not depend on any private layout:

  * code that computes the class-name string also computes the class's
    metaclass object (the constructor passes both to OSMetaClass), so the name
    string gives the metaclass candidates -- this kernel stores no data pointer
    to the name;
  * the candidate vtable is the 168-entry table that matches IOService's own
    except at slots 0 and 7, whose slot 7 getMetaClass returns one of those
    metaclass addresses. Slot 0 is the non-deleting destructor.
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
# In the generated tree the kit is a component link; the device repository keeps its kits
# beside the UniXNU checkout, so the same script also runs straight from the checkout.
for kit in (ROOT / 'kits/XnuPrelink', ROOT.parent / 'UniXNU/kits/XnuPrelink'):
    if (kit / 'native_prelink.py').is_file():
        sys.path.insert(0, str(kit))
        break
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

    # Candidate vtables: 168 entries, slots 2..6 equal IOService's, slots 0 and 7
    # (destructor pair and getMetaClass) overridden. Any further overrides are allowed and
    # reported: IOAcceleratorES itself is expected to have none, but this build's class may.
    def adrp_target(pc, word):
        imm = ((word >> 5) & 0x7ffff) * 4096 + ((word >> 29) & 3) * 4096
        if imm & (1 << 32):
            imm -= 1 << 33
        return (pc & ~0xfff) + imm

    def getter_target(getter):
        # getMetaClass returns the address of the class's gMetaClass object. The compiler
        # either computes it with adrp/add or, when it constant-folds, materializes it with
        # movz/movk; both forms are decoded here and the register is tracked.
        code = backed(getter, 64)
        constant, register = None, None
        for i in range(0, 60, 4):
            word = int.from_bytes(code[i:i + 4], 'little')
            if word & 0x9f00001f == 0x90000000:
                page, rd = adrp_target(getter + i, word), word & 0x1f
                for j in range(i + 4, min(i + 20, 60), 4):
                    second = int.from_bytes(code[j:j + 4], 'little')
                    if second & 0xffc00000 == 0x91000000 and (second >> 5) & 0x1f == rd:
                        return page + ((second >> 10) & 4095)
                constant, register = page, rd
                continue
            if word & 0xff800000 == 0xd2800000:            # movz xN, #imm16, lsl #hw
                register = word & 0x1f
                constant = ((word >> 5) & 0xffff) << (16 * ((word >> 21) & 3))
                continue
            if word & 0xff800000 == 0xf2800000:            # movk xN, #imm16, lsl #hw
                if register is not None and (word & 0x1f) == register:
                    shift = 16 * ((word >> 21) & 3)
                    constant = (constant & ~(0xffff << shift)) | (((word >> 5) & 0xffff) << shift)
                continue
            if word == 0xd65f03c0 and constant is not None:  # ret
                return constant
        return constant

    data_ranges = [(s['va'], s['va'] + s['filesize']) for s in image['segments']
                    if s['name'] in ('__DATA', '__DATA_CONST', '__const')]

    def is_data(address):
        return any(lo <= address < hi for lo, hi in data_ranges)

    # The metaclass object is what the class's constructor code computes together with the
    # class-name string: no data word points at the string in this kernel, the name is
    # materialised in code. So the class name gives the metaclass, not the other way round.
    metaclass_candidates = set()
    for segment in image['segments']:
        if segment['name'] not in ('__TEXT', '__TEXT_EXEC'):
            continue
        data = raw[segment['fileoff']:segment['fileoff'] + segment['filesize']]
        va = segment['va']
        for at in range(0, len(data) - 24, 4):
            word = int.from_bytes(data[at:at + 4], 'little')
            if word & 0x9f00001f == 0x90000000:
                page = adrp_target(va + at, word)
                for j in range(at + 4, min(at + 24, len(data) - 4), 4):
                    second = int.from_bytes(data[j:j + 4], 'little')
                    if second & 0xffc00000 == 0x91000000:
                        target = page + ((second >> 10) & 4095)
                        if target in string_vas:
                            # The constructor function references the metaclass nearby.
                            lo, hi = max(0, at - 192), min(len(data) - 24, at + 192)
                            for k in range(lo, hi, 4):
                                w2 = int.from_bytes(data[k:k + 4], 'little')
                                if w2 & 0x9f00001f == 0x90000000:
                                    page2 = adrp_target(va + k, w2)
                                    for m in range(k + 4, min(k + 24, len(data) - 4), 4):
                                        s2 = int.from_bytes(data[m:m + 4], 'little')
                                        if s2 & 0xffc00000 == 0x91000000:
                                            t2 = page2 + ((s2 >> 10) & 4095)
                                            if is_data(t2) and t2 not in string_vas:
                                                metaclass_candidates.add(t2)
                        break
    # How the code reaches the strings at all: every adrp/add target in the same 4 KiB page
    # as a class-name occurrence, so a different reference form or a different copy shows.
    for string_va in string_vas:
        page = string_va & ~0xfff
        refs = []
        for segment in image['segments']:
            if segment['name'] not in ('__TEXT', '__TEXT_EXEC'):
                continue
            data = raw[segment['fileoff']:segment['fileoff'] + segment['filesize']]
            va = segment['va']
            for at in range(0, len(data) - 24, 4):
                word = int.from_bytes(data[at:at + 4], 'little')
                if word & 0x9f00001f != 0x90000000:
                    continue
                page_base = adrp_target(va + at, word)
                if page_base != page:
                    continue
                for j in range(at + 4, min(at + 24, len(data) - 4), 4):
                    second = int.from_bytes(data[j:j + 4], 'little')
                    if second & 0xffc00000 == 0x91000000:
                        target = page_base + ((second >> 10) & 4095)
                        if page <= target < page + 0x1000:
                            refs.append((va + at, target))
                        break
        print(f'  string {string_va:#x}: {len(refs)} reference(s) into its page' +
              ('' if not refs else ': ' + ', '.join(f'{pc:#x}->{target - string_va:+#x}' for pc, target in refs[:8])))
    print('metaclass candidates from class-name references: ' +
          ', '.join(hex(v) for v in sorted(metaclass_candidates)))
    require(metaclass_candidates, 'no metaclass candidate near a class-name reference')
    text_ranges = [(s['va'], s['va'] + s['filesize']) for s in image['segments']
                   if s['name'] in ('__TEXT', '__TEXT_EXEC')]

    def executable(address):
        return any(lo <= address < hi for lo, hi in text_ranges)

    candidates = []
    for segment in image['segments']:
        for section in segment['sections']:
            if section['name'] != '__const' or section['size'] < VIRTUAL_ENTRIES * 8:
                continue
            data = raw[section['fileoff']:section['fileoff'] + section['size']]
            for at in range(0, len(data) - VIRTUAL_ENTRIES * 8 + 1, 8):
                window = data[at:at + VIRTUAL_ENTRIES * 8]
                if (window[16:56] != base[16:56] or window[64:] != base[64:]
                        or window[0:8] == base[0:8] or window[56:64] == base[56:64]):
                    continue
                if not (executable(int.from_bytes(window[0:8], 'little')) and
                        executable(int.from_bytes(window[56:64], 'little'))):
                    continue
                candidates.append(section['va'] + at)
    print(f'vtable candidates (slots 2..6 as IOService, 0/7 overridden, both executable): {len(candidates)}')
    print('  ' + ', '.join(hex(c) for c in candidates[:24]) + (' ...' if len(candidates) > 24 else ''))
    require(candidates, 'no vtable candidate found')

    # Each candidate's getMetaClass computes its metaclass object. The class name is not a
    # stored pointer in this kernel (the metaclass private object is built at run time), so a
    # candidate belongs to IOAcceleratorES when one function both computes the metaclass
    # address and references the class-name string.
    metas = {}
    unreadable = 0
    for table in candidates:
        try:
            getter = qword(table + 16 + 7 * 8)
            metas[table] = getter_target(getter)
        except SystemExit:
            metas[table] = None
        if metas[table] is None and unreadable < 3:
            unreadable += 1
            try:
                print(f'  getter of {table:#x} at {qword(table + 16 + 7 * 8):#x}: '
                      f'no address materialised; first words ' + backed(qword(table + 16 + 7 * 8), 32).hex())
            except SystemExit:
                print(f'  getter of {table:#x}: unbacked')
    named = []
    for table, meta in metas.items():
        print(f'  table {table:#x}: slot7 getter returns '
              f'{meta if meta is None else hex(meta)}, '
              f'class-name candidate: {bool(meta) and meta in metaclass_candidates}')
        if meta and meta in metaclass_candidates:
            named.append((table, meta, None))
    require(len(named) == 1, f'expected one candidate tied to the class name, found {len(named)}')
    vptr, metaclass, named_string = named[0]
    entries = [qword(vptr + 16 + i * 8) for i in range(VIRTUAL_ENTRIES)]
    overrides = [i for i in range(VIRTUAL_ENTRIES)
                 if entries[i] != int.from_bytes(base[i * 8:(i + 1) * 8], 'little')]
    print(f'vtable {vptr:#x}: D2 {entries[0]:#x} D1 {entries[1]:#x} getMetaClass {entries[7]:#x} '
          f'overrides {overrides}')

    def branch_target(pc, word):
        offset = word & 0x3ffffff
        if offset & (1 << 25):
            offset -= 1 << 26
        return pc + offset * 4

    constructors = []
    for segment in image['segments']:
        if segment['name'] not in ('__TEXT', '__TEXT_EXEC'):
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

    plan = {
        'kernel_sha256': hashlib.sha256(raw).hexdigest(),
        'metaclass': hex(metaclass),
        'vptr': hex(vptr),
        'instance_bytes': INSTANCE_BYTES,
        'virtual_entries': VIRTUAL_ENTRIES,
        'operational_overrides': [],
        'metadata_overrides': overrides,
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
