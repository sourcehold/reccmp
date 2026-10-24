"""Annotating the recomp assembly with the names of local variables and
parameters, for the benefit of a human reading the diff.

The PDB tells us where each local of a function lives. `--verbose` output is
much easier to follow if we say so:

    mov eax, dword ptr [ebp - 0x1c:local_x]
    mov ecx:this, eax

The annotation is cosmetic. It is applied to the text we print, never to the
text we diff, so it cannot change a match percentage.

Two kinds of location appear in the PDB:

* `S_BPREL32` gives an offset into the stack frame. Where the function keeps a
  frame pointer, that offset is the `ebp` displacement as it appears in the
  instruction, and every access can be named. Where the function has no frame
  pointer (MSVC omits it when optimizing), the offset is relative to `esp` *on
  entry to the function*, so we have to follow `esp` through the function to
  know what displacement an instruction would use for a given slot. We read the
  function as a straight line, which is right until the first branch target, so
  we stop naming stack slots at the first instruction whose stack depth we can
  no longer be sure of -- a `ret` or `jmp` partway through the function, or a
  write to `esp` we cannot follow.

* `S_REGISTER` says the variable lives in a register. This record has no live
  range in the PDB formats we read, so it tells us where the variable lives in
  general, not that the register holds it at a particular instruction: an
  optimized function will reuse the register for other things in between.
  Treat a register annotation as a hint, unlike a stack annotation.

A variable declared inside a lexical block (`S_BLOCK32`) only exists in that
part of the function, and sibling blocks may share a stack slot, so such a
variable is only named within the range of its block.
"""

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from reccmp.compare.asm.parse import AsmExcerpt

# `[ebp - 0x1c]`, `[esp + 4]`, `[ebp]`. Capstone writes a displacement under 10
# in decimal and the rest in hex. A second register in the expression (such as
# `[esi + ecx + 2]`) is not a frame access, and does not match.
_FRAME_ACCESS = re.compile(
    r"\[(?P<base>ebp|esp)(?:\s*(?P<sign>[+-])\s*(?P<disp>0x[0-9a-fA-F]+|[0-9]+))?\]"
)

# The x86 registers a PDB may name as the home of a variable. We match the
# spelling the PDB uses and no other: that `ecx` holds a variable says nothing
# about what `cl` holds at a given instruction.
_REGISTERS = frozenset(
    (
        "eax",
        "ebx",
        "ecx",
        "edx",
        "esi",
        "edi",
        "ebp",
        "esp",
        "ax",
        "bx",
        "cx",
        "dx",
        "si",
        "di",
        "bp",
        "sp",
        "al",
        "bl",
        "cl",
        "dl",
        "ah",
        "bh",
        "ch",
        "dh",
    )
)

# `sub esp, 0x10` and friends.
_ESP_ARITHMETIC = re.compile(r"^(?P<op>add|sub) esp, (?P<imm>0x[0-9a-fA-F]+|[0-9]+)$")

# After one of these, the next instruction is a branch target rather than the
# next step, so following esp in a straight line stops working.
_ENDS_FLOW = frozenset(("jmp", "int3", "iret", "iretd", "ud2", "hlt"))

# Anything else that writes to esp leaves us unable to say how deep the stack is.
_ESP_WRITE = re.compile(r"^(?:mov|lea|and|or|xor|adc|sbb|xchg)\s+esp\b")

# The base register of a memory operand, as in `[esi + 4]` or `[edi]`.
_MEMORY_BASE = re.compile(r"\[(?P<base>[a-z]{2,3})(?P<rest>[^\]]*)\]")

_SIZE_PREFIX = re.compile(
    r"^(?:byte|word|dword|qword|tbyte|xmmword|ymmword)\s+ptr\s+", re.IGNORECASE
)


@dataclass(frozen=True)
class LocalSlot:
    """A named stack slot of one function."""

    offset: int
    """Frame-base-relative offset, exactly as the PDB reports it."""

    name: str

    start: int | None = None
    """Together with `end`, the range of recomp addresses where this variable
    exists. Both are None for a variable at function scope."""

    end: int | None = None


@dataclass(frozen=True)
class LocalRegister:
    """A named register of one function."""

    register: str
    name: str
    start: int | None = None
    end: int | None = None


def _in_scope(start: int | None, end: int | None, addr: int | None) -> bool:
    if start is None or end is None:
        return True

    if addr is None:
        return False

    return start <= addr < end


@dataclass
class FunctionLocals:
    """Where the locals and parameters of one recomp function live."""

    frame_pointer: bool = False
    """True if the function keeps a frame pointer, in which case the offset of
    each slot is an `ebp` displacement. If not, it is relative to `esp` on
    entry to the function."""

    slots: list[LocalSlot] = field(default_factory=list)
    registers: list[LocalRegister] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.slots) or bool(self.registers)

    def slot_name(self, offset: int, addr: int | None) -> str | None:
        """The name of the slot at the given frame-base-relative offset."""
        for slot in self.slots:
            if slot.offset == offset and _in_scope(slot.start, slot.end, addr):
                return slot.name

        return None

    def register_name(self, register: str, addr: int | None) -> str | None:
        for reg in self.registers:
            if reg.register == register and _in_scope(reg.start, reg.end, addr):
                return reg.name

        return None

    def named_registers(self) -> list[str]:
        return [reg.register for reg in self.registers]


class _StackDepth:
    """How far `esp` has moved from its value on entry to the function.

    Only needed for a function without a frame pointer, where the PDB reports
    each slot relative to `esp` on entry. `None` means we lost track, and we do
    not guess from there on.
    """

    def __init__(self) -> None:
        self.depth: int | None = 0

    def frame_base_offset(self, disp: int) -> int | None:
        """The offset relative to `esp` on entry of an `[esp + disp]` access,
        or None if we no longer know how deep the stack is."""
        if self.depth is None:
            return None

        return disp - self.depth

    def advance(self, instruction: str):
        """Apply the effect of an instruction on `esp`."""
        if self.depth is None:
            return

        mnemonic, _, _ = instruction.partition(" ")

        if mnemonic == "push":
            self.depth += 4
        elif mnemonic == "pop":
            self.depth -= 4
        elif mnemonic in ("pusha", "pushad"):
            self.depth += 32
        elif mnemonic in ("popa", "popad"):
            self.depth -= 32
        elif (match := _ESP_ARITHMETIC.match(instruction)) is not None:
            imm = int(match.group("imm"), 0)
            self.depth += imm if match.group("op") == "sub" else -imm
        elif mnemonic in ("leave", "enter"):
            self.depth = None
        elif mnemonic in _ENDS_FLOW or mnemonic.startswith("ret"):
            # Whatever follows is reached by a branch from somewhere else, and
            # the stack is not necessarily as deep there as it is here. We are
            # reading the function as a straight line, so from this point on we
            # have to stop naming stack slots rather than guess.
            self.depth = None
        elif _ESP_WRITE.match(instruction) is not None:
            # `mov esp, ebp` and the like. We no longer know the depth, and
            # must not put a name on anything esp-relative after this.
            self.depth = None


# Instructions whose first operand is written without being read. One of these
# rebinds the register to something else, which ends the variable's stay in it.
# A read-modify-write such as `add ecx, 1` is left out on purpose: that updates
# the variable rather than replacing it.
_REBINDS_DEST = frozenset(
    ("mov", "movzx", "movsx", "movsxd", "lea", "pop", "xchg", "lods", "lodsb")
)

# Caller-saved registers in the MSVC calling conventions: a CALL may leave
# anything in these.
_VOLATILE_REGISTERS = frozenset(
    ("eax", "ecx", "edx", "ax", "cx", "dx", "al", "cl", "dl")
)


def _rebinds(instruction: str, register: str) -> bool:
    """True if this instruction leaves something other than our variable in the
    given register."""
    mnemonic, _, operands = instruction.partition(" ")

    if mnemonic.startswith("call"):
        return register in _VOLATILE_REGISTERS

    if mnemonic not in _REBINDS_DEST and not mnemonic.startswith("cmov"):
        return False

    dest = operands.split(", ")[0]
    return dest == register


def _reads(instruction: str, register: str) -> bool:
    """True if the register appears as a value this instruction consumes."""
    mnemonic, _, operands = instruction.partition(" ")

    # A bare `push esi` / `pop esi` is register save/restore as often as it is
    # a use of the variable, so it tells us nothing about who owns the register.
    if mnemonic in ("push", "pop") and operands in _REGISTERS:
        return False

    parts = operands.split(", ")
    if _rebinds(instruction, register) and parts:
        parts = parts[1:]

    return any(
        re.search(rf"(?<![0-9A-Za-z_]){re.escape(register)}(?![0-9A-Za-z_])", part)
        is not None
        for part in parts
    )


class _RegisterLiveness(NamedTuple):
    """Where we are willing to say a register holds its variable."""

    indices: range
    """Instruction indices, as a half-open range."""

    definition: int | None
    """The index of the instruction that puts the variable in the register, if
    the variable is produced inside the function rather than passed in. Only
    there may the destination operand carry the name."""


def _register_liveness(
    instructions: list[str], register: str
) -> _RegisterLiveness | None:
    """Work out over which instructions the register holds its variable.

    The PDB records which register a variable lives in but not for how long, so
    we have to read it off the instructions. If the register is consumed before
    anything rebinds it, the variable was passed in and stays until the first
    rebind. Otherwise the variable is produced by that rebind and stays until
    the next one. Either way we stop where the register certainly holds
    something else, rather than naming every occurrence of it and being wrong
    for most of the function.
    """
    rebinds = [i for i, text in enumerate(instructions) if _rebinds(text, register)]
    reads = [i for i, text in enumerate(instructions) if _reads(text, register)]

    first_rebind = rebinds[0] if rebinds else None
    first_read = reads[0] if reads else None

    if first_read is not None and (first_rebind is None or first_read <= first_rebind):
        # Passed in by the caller. The rebind itself is included because its
        # source operands still read the old value; its destination is not
        # named because `definition` is None.
        end = len(instructions) if first_rebind is None else first_rebind + 1
        return _RegisterLiveness(range(0, end), None)

    if first_rebind is None:
        return None

    later = next((i for i in rebinds if i > first_rebind), None)
    end = len(instructions) if later is None else later + 1

    # Nothing read the register before it was rebound, so either the variable
    # is produced here, or it was passed in and discarded without being used.
    # Only the first of those is worth naming, and it is the one where the
    # register is read again afterwards.
    if not any(first_rebind < i < end for i in reads):
        return None

    return _RegisterLiveness(range(first_rebind, end), first_rebind)


def _annotate_operand(
    operand: str,
    addr: int | None,
    locals_: FunctionLocals,
    depth: _StackDepth,
    *,
    live_registers: frozenset,
    rebound_dest: bool,
) -> str:
    """Add the variable name to one operand of an instruction, if we have it."""
    prefix = ""
    rest = operand
    if (match := _SIZE_PREFIX.match(rest)) is not None:
        prefix = match.group(0)
        rest = rest[match.end() :]

    # `mov eax, ecx` -- a variable that lives in a register.
    if rest in _REGISTERS:
        if rest not in live_registers or rebound_dest:
            return operand

        name = locals_.register_name(rest, addr)
        return operand if name is None else f"{prefix}{rest}:{name}"

    # `mov eax, dword ptr [ebp - 0x1c]` -- a variable on the stack.
    if (match := _FRAME_ACCESS.fullmatch(rest)) is not None:
        base = match.group("base")
        disp_text = match.group("disp")
        disp = 0 if disp_text is None else int(disp_text, 0)
        if match.group("sign") == "-":
            disp = -disp

        # With a frame pointer the PDB offset is the ebp displacement itself.
        # Without one it is relative to esp on entry, so only an esp access can
        # be resolved -- ebp is then just another register.
        offset: int | None = None
        if locals_.frame_pointer:
            if base == "ebp":
                offset = disp
        elif base == "esp":
            offset = depth.frame_base_offset(disp)

        if offset is not None:
            name = locals_.slot_name(offset, addr)
            if name is not None:
                return f"{prefix}[{rest[1:-1]}:{name}]"

    # `mov eax, dword ptr [esi + 4]` -- a pointer variable that lives in a
    # register, dereferenced. Name the base register so the field access reads
    # as `variable->field`.
    if (match := _MEMORY_BASE.fullmatch(rest)) is not None:
        base = match.group("base")
        # ebp and esp address the frame here, they are not the variable; the
        # rule above is the one that names a frame slot.
        if base not in ("ebp", "esp") and base in _REGISTERS and base in live_registers:
            name = locals_.register_name(base, addr)
            if name is not None:
                return f"{prefix}[{base}:{name}{match.group('rest')}]"

    return operand


def annotate_locals(excerpt: AsmExcerpt, locals_: FunctionLocals) -> AsmExcerpt:
    """Name the locals and parameters used by each instruction of a recomp
    function. For display only: the text we diff is not annotated."""
    if not locals_:
        return excerpt

    instructions = [text for _, text in excerpt]

    # Work out where each named register actually holds its variable.
    live: dict[int, set[str]] = {}
    definitions: dict[int, set[str]] = {}
    for register in locals_.named_registers():
        liveness = _register_liveness(instructions, register)
        if liveness is None:
            continue

        for index in liveness.indices:
            live.setdefault(index, set()).add(register)

        if liveness.definition is not None:
            definitions.setdefault(liveness.definition, set()).add(register)

    depth = _StackDepth()
    output: AsmExcerpt = []

    for index, (addr, instruction) in enumerate(excerpt):
        mnemonic, space, operands = instruction.partition(" ")
        if space:
            live_registers = frozenset(live.get(index, ()))
            defined_here = definitions.get(index, set())
            # Only look at the operands of an instruction, so that a register
            # name appearing inside a symbol is left alone.
            # Rejoining with the same separator is lossless, so an operand
            # list we split in an unexpected place is returned unchanged.
            annotated = ", ".join(
                _annotate_operand(
                    operand,
                    addr,
                    locals_,
                    depth,
                    live_registers=live_registers,
                    rebound_dest=(
                        position == 0
                        and operand not in defined_here
                        and _rebinds(instruction, operand)
                    ),
                )
                for position, operand in enumerate(operands.split(", "))
            )
            instruction = f"{mnemonic} {annotated}"

        output.append((addr, instruction))
        depth.advance(instruction)

    return output
