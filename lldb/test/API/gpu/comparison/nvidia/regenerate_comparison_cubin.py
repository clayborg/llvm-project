#!/usr/bin/env python3
"""Build and relocate comparison_artificial.cubin for the artificial core test."""

import argparse
import pathlib
import re
import struct
import subprocess
import tempfile


SECTION_ADDRESSES = {
    ".nv.constant0.compare_kernel": 0x7FFFCF200000,
    ".text.leaf": 0x7FFFCF201000,
    ".text.middle": 0x7FFFCF202000,
    ".text.compare_kernel": 0x7FFFCF203000,
}
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
SYMBOL = struct.Struct("<IBBHQQ")


def _c_string(table, offset):
    return table[offset:].split(b"\0", 1)[0].decode()


def relocate_cubin(raw):
    data = bytearray(raw)
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 cubin")

    section_table = struct.unpack_from("<Q", data, 40)[0]
    section_size, section_count, name_table_index = struct.unpack_from(
        "<HHH", data, 58
    )
    if section_size != SECTION_HEADER.size:
        raise ValueError("unexpected ELF section-header size")

    sections = [
        SECTION_HEADER.unpack_from(data, section_table + i * section_size)
        for i in range(section_count)
    ]
    name_section = sections[name_table_index]
    names = data[name_section[4] : name_section[4] + name_section[5]]
    section_names = [_c_string(names, section[0]) for section in sections]

    relocated_sections = {}
    for name, address in SECTION_ADDRESSES.items():
        try:
            index = section_names.index(name)
        except ValueError as error:
            raise ValueError(f"missing cubin section: {name}") from error
        relocated_sections[index] = address
        struct.pack_into(
            "<Q", data, section_table + index * section_size + 16, address
        )

    symbols = next(section for section in sections if section[1] == 2)  # SHT_SYMTAB
    strings_section = sections[symbols[6]]
    strings = data[
        strings_section[4] : strings_section[4] + strings_section[5]
    ]
    kernel_address = None
    for offset in range(symbols[4], symbols[4] + symbols[5], SYMBOL.size):
        name, _, _, section, value, _ = SYMBOL.unpack_from(data, offset)
        if section not in relocated_sections:
            continue
        value += relocated_sections[section]
        struct.pack_into("<Q", data, offset + 8, value)
        if _c_string(strings, name) == "compare_kernel":
            kernel_address = value

    expected = SECTION_ADDRESSES[".text.compare_kernel"]
    if kernel_address != expected:
        raise ValueError(
            f"compare_kernel is at {kernel_address!r}, expected {expected:#x}"
        )

    # nvcc embeds a process-specific temporary filename in command metadata.
    return re.sub(rb"tmpxft_[0-9a-fA-F]{8}_", b"tmpxft_00000000_", data)


def main():
    directory = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc", help="nvcc executable")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=directory / "comparison_artificial.cubin",
        help="output cubin path",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary_directory:
        raw_path = pathlib.Path(temporary_directory) / "comparison_artificial.cubin"
        subprocess.run(
            [
                args.nvcc,
                "-cubin",
                "-G",
                "-arch=sm_80",
                "comparison_artificial.cu",
                "-o",
                str(raw_path),
            ],
            cwd=directory,
            check=True,
        )
        relocated = relocate_cubin(raw_path.read_bytes())

    args.output.write_bytes(relocated)
    print(f"wrote {args.output} ({len(relocated)} bytes)")


if __name__ == "__main__":
    main()
