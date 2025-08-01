#!/usr/bin/env python3
# This is a helper script to generate data for the register tests.
# It generates
#
#   1. Python lists of expected values.
#   2. The kernels with inline assembly to write those values to registers.
#
# The python lists will be included in the python test file and the kernels
# will be included in the hip file.
import random
from collections import namedtuple
from typing import List, Tuple, TypeAlias
from pprint import pp
from io import StringIO

RegClass = namedtuple("RegClass", ["gpr", "mov"])
GprValuesList: TypeAlias = List[int]
RegClassData: TypeAlias = Tuple[RegClass, GprValuesList]
RegClassesList: TypeAlias = List[RegClassData]


def replicate_byte(value):
    """Replicate the byte across the 4 bytes of a 32-bit value."""
    value = value & 0xFF
    return (value << 24) | (value << 16) | (value << 8) | value


def generate_gpr_values(count) -> GprValuesList:
    """Generate a list of random 32-bit integer values."""
    actual_values = []
    for i in range(count):
        actual_values.append(replicate_byte(i))
    return actual_values


def generate_gpr_asm(reg_classes: RegClassesList) -> str:
    """Generate the inline assembly string to write the values to the GPRs.
    Example Output:
        asm volatile(
            "v_mov_b32_e32   v0, 0xa3b1799d\n"
            "v_mov_b32_e32   v1, 0x46685257\n"
            : : : "v0", "v1");
    """
    asm_lines = []
    regs = dict()  # Use dict to get an ordered set of registers.
    for rc, values in reg_classes:
        for i in range(len(values)):
            reg = f"{rc.gpr}{i}"
            if rc.gpr == "a":
                # The agpr registers cannot be initialized with an immediate value
                # so move it to a vgpr first.
                regs.setdefault("v0")
                asm_lines.append(f'  "v_mov_b32             v0, 0x{values[i]:08X}\\n"')
                asm_lines.append(f'  "{rc.mov} {reg:>4}, v0\\n"')
            else:
                asm_lines.append(f'  "{rc.mov} {reg:>4}, 0x{values[i]:08X}\\n"')
            regs.setdefault(reg)
    clobbers = [f'"{reg}"' for reg in regs.keys()]
    lines = ["asm volatile("] + asm_lines + ["  : : : " + ", ".join(clobbers) + ");"]
    return "\n".join(lines)


def generate_kernel(reg_class: RegClass, values: GprValuesList) -> str:
    """Generate a hip kernels to set register in a class to expected values."""
    gpr = f"{reg_class.gpr}gpr"
    kernel = f"set_known_{gpr}_values_kernel"
    asm = generate_gpr_asm([(reg_class, values)])

    outs = StringIO()
    print(f"__global__ void {kernel}() {{", file=outs)
    print(asm, file=outs)
    print(f"  // GPU BREAKPOINT - {gpr.upper()}", file=outs)
    print("}", file=outs)

    return outs.getvalue()


def generate_kernels() -> str:
    """Generate a string containing hip kernels to set registers to expected values."""
    kernels = []
    for reg_class, values in reg_classes:
        kernels.append(generate_kernel(reg_class, values))
    return "\n".join(kernels)


def generate_expected_values(reg_classes: RegClassesList) -> str:
    """Generate a string containing python lists of expected values."""
    outs = StringIO()
    expected_values = []
    for reg_class, values in reg_classes:
        print(f"{reg_class.gpr.upper()}GPR_VALUES = [", file=outs)
        for value in values:
            print(f"    0x{value:08X},", file=outs)
        print("]", file=outs)

    return outs.getvalue()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GPR values and assembly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--asm", action="store_true", help="Generate only inline assembly"
    )
    parser.add_argument("--kernels", action="store_true", help="Generate kernels")
    parser.add_argument(
        "--values", action="store_true", help="Generate python expected values"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    vgpr_values = generate_gpr_values(256)
    sgpr_values = generate_gpr_values(102)
    agpr_values = generate_gpr_values(256)
    reg_classes = [
        (RegClass("v", "v_mov_b32"), vgpr_values),
        (RegClass("s", "s_mov_b32"), sgpr_values),
        (RegClass("a", "v_accvgpr_write_b32"), agpr_values),
    ]
    if args.asm:
        asm = generate_gpr_asm(reg_classes)
        print(asm)

    if args.kernels:
        kernels = generate_kernels()
        print(kernels)

    if args.values:
        python = generate_expected_values(reg_classes)
        print(python)
