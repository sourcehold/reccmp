"""Converts x86 machine code into text (i.e. assembly). The end goal is to
compare the code in the original and recomp binaries, using longest common
subsequence (LCS), i.e. difflib.SequenceMatcher.
The capstone library takes the raw bytes and gives us the mnemonic
and operand(s) for each instruction. We need to "sanitize" the text further
so that virtual addresses are replaced by symbol name or a generic
placeholder string."""

import re
from collections.abc import Collection
from functools import cache
from typing_extensions import Buffer
from .const import JUMP_MNEMONICS, SINGLE_OPERAND_INSTS
from .instgen import DisasmLiteTuple, InstructGen, SectionType
from .replacement import AddrTestProtocol, NameReplacementProtocol

AsmExcerpt = list[tuple[int | None, str]]

ptr_replace_regex = re.compile(r"(?<=\[)(0x[0-9a-f]+)(?=\])")

displace_replace_regex = re.compile(r"(?<= )(0x[0-9a-f]+)(?=\])")

# For matching an immediate value operand
immediate_replace_regex = re.compile(r"(?<=, )(0x[0-9a-f]+)")

# Size in bytes of a near JMP with a 32-bit relative displacement (opcode 0xE9).
# A tail call uses this encoding; a jump inside the function usually fits in the
# 2-byte short form.
TAIL_CALL_JMP_SIZE = 5


@cache
def from_hex(string: str) -> int | None:
    try:
        return int(string, 16)
    except ValueError:
        pass

    return None


class ParseAsm:
    def __init__(
        self,
        addr_test: AddrTestProtocol | None = None,
        name_lookup: NameReplacementProtocol | None = None,
        is_32bit: bool = True,
    ) -> None:
        self.addr_test = addr_test
        self.name_lookup = name_lookup
        self.is_32bit = is_32bit

        self.replacements: dict[int, str] = {}
        self.indirect_replacements: dict[int, str] = {}
        # Target address of each direct CALL in the function, mapped to the
        # operand text used for it. Consumers use this to detect calls that
        # are routed through a wrapper function in the other binary.
        self.call_targets: dict[int, str] = {}
        # Target address of each 5-byte near JMP in the function, mapped to the
        # operand text used for it. These are the tail calls: a JMP to another
        # function in place of a CALL followed by a RET.
        self.jump_targets: dict[int, str] = {}
        # Every value we weighed up as a possible address, whether or not we
        # went on to replace it. Comparing these between the two binaries is
        # what tells a constant apart from an address: see `shared_constants`.
        self.tested_values: set[int] = set()
        # Of those, the ones we replaced without being able to name the entity
        # they start, so the evidence that they are addresses is only that they
        # fall somewhere inside something we know about.
        self.uncertain_values: set[int] = set()
        # Values to leave alone because we have decided they are constants.
        self.constant_veto: frozenset[int] = frozenset()
        self.number_placeholders = True

    def reset(self):
        self.replacements = {}
        self.indirect_replacements = {}
        self.call_targets = {}
        self.jump_targets = {}
        self.tested_values = set()
        self.uncertain_values = set()

    def replacement_names(self) -> set[str]:
        """Every name or placeholder substituted into the asm of the last parse."""
        return set(self.replacements.values()) | set(
            self.indirect_replacements.values()
        )

    def is_addr(self, value: int) -> bool:
        """Wrapper for user-provided address test"""
        if callable(self.addr_test):
            return self.addr_test(value)

        return False

    def lookup(
        self, addr: int, exact: bool = False, indirect: bool = False
    ) -> str | None:
        """Wrapper for user-provided name lookup"""
        if callable(self.name_lookup):
            return self.name_lookup(addr, exact=exact, indirect=indirect)

        return None

    def _next_placeholder(self) -> str:
        """The placeholder number corresponds to the number of addresses we have
        already replaced. This is so the number will be consistent across the diff
        if we can replace some symbols with actual names in recomp but not orig."""
        number = len(self.replacements) + len(self.indirect_replacements) + 1
        return f"<OFFSET{number}>" if self.number_placeholders else "<OFFSET>"

    def replace(self, addr: int, exact: bool = False) -> str:
        """Provide a replacement name for the given address."""
        if addr in self.replacements:
            return self.replacements[addr]

        if (name := self.lookup(addr, exact=exact)) is not None:
            self.replacements[addr] = name
            return name

        placeholder = self._next_placeholder()
        self.replacements[addr] = placeholder
        return placeholder

    def address_operand(self, value: int) -> str | None:
        """A replacement for an operand we think holds the address `value`, or
        None if we have decided the value is a constant.

        Unlike a pointer in a memory operand, a bare number in the code can
        just as easily be an integer the program works with. We only have a
        name for the thing it would point at, which is weak evidence: a large
        struct or a function covers a wide range of addresses, and any constant
        that happens to land inside one looks like a pointer into it. So record
        how sure we are, for `shared_constants` to settle.

        The caller records the value in `tested_values` first, because a value
        this side does *not* take for an address is exactly the evidence the
        other side needs.
        """
        if value in self.constant_veto:
            return None

        # Naming the exact start of an entity is good evidence. Landing inside
        # one, or not being able to name it at all, is not.
        if self.lookup(value, exact=True) is None:
            self.uncertain_values.add(value)

        return self.replace(value)

    def indirect_replace(self, addr: int) -> str:
        if addr in self.indirect_replacements:
            return self.indirect_replacements[addr]

        if (name := self.lookup(addr, exact=True, indirect=True)) is not None:
            self.indirect_replacements[addr] = name
            return name

        placeholder = self._next_placeholder()
        self.indirect_replacements[addr] = placeholder
        return placeholder

    def hex_replace_always(self, match: re.Match) -> str:
        """If a pointer value was matched, always insert a placeholder"""
        value = int(match.group(1), 16)
        return self.replace(value)

    def hex_replace_relocated(self, match: re.Match) -> str:
        """For replacing immediate value operands. We only want to
        use the placeholder if we are certain that this is a valid address.
        We can check the relocation table to find out."""
        value = int(match.group(1), 16)
        self.tested_values.add(value)
        if self.is_addr(value):
            replacement = self.address_operand(value)
            if replacement is not None:
                return replacement

        return match.group(0)

    def hex_replace_annotated(self, match: re.Match) -> str:
        """For replacing immediate value operands. Here we replace the value
        only if the name lookup returns something. Do not use a placeholder."""
        value = int(match.group(1), 16)
        self.tested_values.add(value)
        if value in self.constant_veto:
            return match.group(0)

        # Ask about the exact address first, so that the lookup for the name we
        # actually use is the last one we make.
        names_an_entity = self.lookup(value, exact=True) is not None

        placeholder = self.lookup(value)
        if placeholder is not None:
            if not names_an_entity:
                self.uncertain_values.add(value)

            return placeholder

        return match.group(0)

    def hex_replace_indirect(self, match: re.Match) -> str:
        """Edge case for hex_replace_always. The context of the instruction
        tells us that the pointer value is an absolute indirect.
        So we go to that location in the binary to get the address.
        If we cannot identify the indirect address, fall back to a lookup
        on the original pointer value so we might display something useful."""
        value = int(match.group(1), 16)
        return self.indirect_replace(value)

    def sanitize(self, inst: DisasmLiteTuple) -> tuple[str, str]:
        # For jumps or calls, if the entire op_str is a hex number, the value
        # is a relative offset.
        # Otherwise (i.e. it looks like `dword ptr [address]`) it is an
        # absolute indirect that we will handle below.
        # Providing the starting address of the function to capstone.disasm has
        # automatically resolved relative offsets to an absolute address.
        # We will have to undo this for some of the jumps or they will not match.
        inst_address, inst_size, inst_mnemonic, inst_op_str = inst

        if (
            inst_mnemonic in SINGLE_OPERAND_INSTS
            and (op_str_address := from_hex(inst_op_str)) is not None
        ):
            if inst_mnemonic == "call":
                replacement = self.replace(op_str_address, exact=True)
                self.call_targets[op_str_address] = replacement
                return (inst_mnemonic, replacement)

            if inst_mnemonic == "push":
                self.tested_values.add(op_str_address)
                if self.is_addr(op_str_address):
                    pushed = self.address_operand(op_str_address)
                    if pushed is not None:
                        return (inst_mnemonic, pushed)

                # To avoid falling into jump handling
                return (inst_mnemonic, inst_op_str)

            if inst_mnemonic == "jmp":
                # The unwind section contains JMPs to other functions.
                # If we have a name for this address, use it. If not,
                # do not create a new placeholder. We will instead
                # fall through to generic jump handling below.
                potential_name = self.lookup(op_str_address, exact=True)
                if potential_name is not None:
                    if inst_size == TAIL_CALL_JMP_SIZE:
                        self.jump_targets[op_str_address] = potential_name
                    return (inst_mnemonic, potential_name)

            # Else: this is any jump
            # Show the jump offset rather than the absolute address
            jump_displacement = op_str_address - (inst_address + inst_size)
            replacement = hex(jump_displacement)
            if inst_mnemonic == "jmp" and inst_size == TAIL_CALL_JMP_SIZE:
                # A tail call to a function we have no name for. Record it
                # anyway: the other binary may reach the same function via a
                # wrapper whose name embeds this address.
                self.jump_targets[op_str_address] = replacement
            return (inst_mnemonic, replacement)

        if inst_mnemonic == "call":
            # Special handling for absolute indirect CALL.
            op_str = ptr_replace_regex.sub(self.hex_replace_indirect, inst_op_str)
        else:
            op_str = ptr_replace_regex.sub(self.hex_replace_always, inst_op_str)

            # We only want relocated addresses for pointer displacement.
            # i.e. ptr [register + something]
            # Otherwise we would use a placeholder for every stack variable,
            # vtable call, or this->member access.
            op_str = displace_replace_regex.sub(self.hex_replace_relocated, op_str)

        # In the event of pointer comparison, only replace the immediate value
        # if it is a known address.
        if inst_mnemonic == "cmp":
            op_str = immediate_replace_regex.sub(self.hex_replace_annotated, op_str)
        else:
            op_str = immediate_replace_regex.sub(self.hex_replace_relocated, op_str)

        return (inst_mnemonic, op_str)

    def parse_asm(
        self,
        data: Buffer,
        start_addr: int,
        constant_veto: Collection[int] = (),
    ) -> AsmExcerpt:
        self.reset()
        self.constant_veto = frozenset(constant_veto)
        asm: AsmExcerpt = []

        ig = InstructGen(bytes(data), start_addr, self.is_32bit)

        for section in ig.sections:
            if section.type == SectionType.CODE:
                for inst in section.contents:
                    inst_address, inst_size, inst_mnemonic, inst_op_str = inst
                    # If there is no pointer or immediate value in the op_str,
                    # there is nothing to sanitize.
                    # This leaves us with cases where a small immediate value or
                    # small displacement (this.member or vtable calls) appears.
                    # If we assume that instructions we want to sanitize need to be 5
                    # bytes -- 1 for the opcode and 4 for the address -- exclude cases
                    # where the hex value could not be an address.
                    # The exception is jumps which are as small as 2 bytes
                    # but are still useful to sanitize.
                    if "0x" in inst_op_str and (
                        inst_mnemonic in JUMP_MNEMONICS
                        or inst_size > 4
                        or not self.is_32bit
                    ):
                        result = self.sanitize(inst)
                    else:
                        result = (inst_mnemonic, inst_op_str)

                    # mnemonic + " " + op_str
                    asm.append((inst_address, " ".join(result)))
            elif section.type == SectionType.ADDR_TAB:
                asm.append((None, "Jump table:"))
                for ofs, target in section.contents:
                    # Jumps in jump tables are absolute, which can lead to false positives
                    # when the function does not have the same address in orig and recomp.
                    # Therefore, we compute where the jump will go relative to the start of the function.
                    target_relative_to_function_start = target - start_addr
                    asm.append(
                        (ofs, f"start + 0x{(target_relative_to_function_start):x}")
                    )

            elif section.type == SectionType.DATA_TAB:
                asm.append((None, "Data table:"))
                for ofs, b in section.contents:
                    asm.append((ofs, hex(b)))

        return asm


def shared_constants(orig: "ParseAsm", recomp: "ParseAsm") -> frozenset[int]:
    """The values that both binaries use as a bare number, where at least one
    of them took the number for an address without being able to name what it
    starts.

    Two matched variables do not live at the same address: the two images are
    laid out independently, so the orig code and the recomp code hold different
    numbers for the same variable and we rely on naming them to line the two
    sides up. A number that is *identical* in both is therefore not an address
    but a constant from the source, and naming it on one side only is what
    makes an otherwise perfect function fall short of 100%.

    Leaving such a value alone is safe either way: if it really were the same
    address in both images, the two sides would still read the same.
    """
    shared = orig.tested_values & recomp.tested_values
    if not shared:
        return frozenset()

    return frozenset(shared & (orig.uncertain_values | recomp.uncertain_values))
