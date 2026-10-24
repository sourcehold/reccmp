"""Normalization of CALL instructions for recompilation projects that route
calls to original functions through a wrapper (a "resolver").

Some projects cannot call an original function directly. They instead generate
a template instantiation that performs the call, so the recomp assembly reads

    call FunctionResolver::Resolver<void (__thiscall X::*)(int),0,4657152,...>::...::call

where the original assembly reads

    call <OFFSET12>      (or the name of the annotated function)

The two instructions are equivalent, but the diff counts them as a mismatch,
which makes it impossible for such a function to reach a 100% match.

The same applies to a *tail call*, where the compiler emits a 5-byte near JMP
to the function instead of a CALL followed by a RET. The recomp side then reads

    jmp FunctionResolver::Resolver<...>::call

and the handling is identical apart from the mnemonic.

Two strategies are provided here:

* `resolve_wrapped_calls`: the original address of the call target appears
  inside the wrapper's (mangled) name as a decimal or hex number, because that
  is how the resolver template is parameterized. If the number matches the
  target address of the call on the original side, we know the wrapper calls
  that function and we rewrite the recomp instruction to use the same text as
  the original. This keeps real call mismatches visible.

* `ignore_call_targets`: the blunt instrument. Replace the target of every
  non-register CALL with a constant placeholder on both sides so that calls
  never contribute to the diff.
"""

import re
from typing import Iterable, Mapping

from reccmp.compare.asm.parse import AsmExcerpt

# Any number that could plausibly be a virtual address. We check the value
# against the set of call targets in the function, so the pattern only has to
# avoid picking up numbers that are part of a larger identifier.
_NUMBER_TOKEN = re.compile(
    r"(?<![0-9A-Za-z_$])(0x[0-9a-fA-F]+|[0-9]+)(?![0-9A-Za-z_$])"
)

# Lowest value we will consider to be an address. Resolver templates also carry
# small integer parameters (indices, flags) that we must not interpret as one.
_MIN_PLAUSIBLE_ADDR = 0x1000

# An integer template argument in an *undecorated* MSVC name. MSVC encodes a
# non-negative value as "$0", the value in base 16 with 'A'..'P' standing for
# the nibbles 0..15, and a terminating "@" ("$0EBMDBA@" == 0x41C310). A PDB
# gives us the mangled name, so a resolver's address parameter never appears as
# a plain decimal number there.
_MANGLED_NUMBER_TOKEN = re.compile(r"\$0([A-P]+)@")


def _from_mangled(digits: str) -> int:
    value = 0
    for char in digits:
        value = value * 16 + (ord(char) - ord("A"))
    return value


CALL_PLACEHOLDER = "<CALL>"


def _is_call(instruction: str) -> bool:
    return instruction.startswith("call ")


def _call_operand(instruction: str) -> str:
    return instruction[len("call ") :].strip()


def _is_tail_jump(instruction: str) -> bool:
    """A JMP whose operand is a symbol name rather than a displacement.
    The parser only names the target of a JMP that goes to a known function,
    so this is the tail call we are interested in."""
    if not instruction.startswith("jmp "):
        return False

    operand = instruction[len("jmp ") :].strip()
    # A plain (possibly negative) hex number is a jump displacement inside the
    # function, not a named target.
    return _JUMP_DISPLACEMENT.fullmatch(operand) is None


def _jump_operand(instruction: str) -> str:
    return instruction[len("jmp ") :].strip()


_JUMP_DISPLACEMENT = re.compile(r"-?0x[0-9a-fA-F]+")


def numbers_in_name(name: str) -> Iterable[int]:
    """Every number embedded in the given symbol name that could be an address."""
    for match in _NUMBER_TOKEN.finditer(name):
        text = match.group(1)
        try:
            value = int(text, 16) if text.startswith("0x") else int(text, 10)
        except ValueError:  # pragma: no cover - the regex guarantees this parses
            continue

        if value >= _MIN_PLAUSIBLE_ADDR:
            yield value

    for match in _MANGLED_NUMBER_TOKEN.finditer(name):
        value = _from_mangled(match.group(1))
        if value >= _MIN_PLAUSIBLE_ADDR:
            yield value


def _resolve(
    mnemonic: str, operand: str, orig_targets: Mapping[int, str]
) -> str | None:
    """The replacement instruction for a wrapped CALL/JMP, or None if the
    operand does not name a wrapper for one of `orig_targets`."""
    # Don't touch a target we could not give a name to: there is no wrapper
    # name to read an original address from.
    if operand.startswith("<OFFSET"):
        return None

    for value in numbers_in_name(operand):
        orig_text = orig_targets.get(value)
        if orig_text is not None:
            return f"{mnemonic} {orig_text}"

    return None


def resolve_wrapped_calls(
    recomp: AsmExcerpt,
    orig_call_targets: Mapping[int, str],
    orig_jump_targets: Mapping[int, str] | None = None,
) -> AsmExcerpt:
    """Rewrite each recomp CALL whose target name embeds the original address of
    one of the CALL targets in the original function. The replacement is the
    text used by the original instruction, so the diff lines become identical.

    Tail calls are handled the same way: a recomp 5-byte JMP to a wrapper is
    rewritten using the text of the JMP at that point in the original function.

    `orig_call_targets` maps the (absolute) target address of each direct CALL
    in the original function to the operand text used for it in the diff.
    `orig_jump_targets` does the same for each 5-byte JMP.
    """
    if not orig_call_targets and not orig_jump_targets:
        return recomp

    output: AsmExcerpt = []
    for addr, instruction in recomp:
        if orig_call_targets and _is_call(instruction):
            replacement = _resolve(
                "call", _call_operand(instruction), orig_call_targets
            )
            if replacement is not None:
                instruction = replacement
        elif orig_jump_targets and _is_tail_jump(instruction):
            replacement = _resolve("jmp", _jump_operand(instruction), orig_jump_targets)
            if replacement is not None:
                instruction = replacement

        output.append((addr, instruction))

    return output


def ignore_call_targets(excerpt: AsmExcerpt, names: Iterable[str]) -> AsmExcerpt:
    """Replace the target of each CALL with a constant placeholder.

    Only the names and placeholders that the sanitizer substituted into this
    excerpt are replaced, so `call eax` and the like are left alone.
    """
    name_set = set(names)
    if not name_set:
        return excerpt

    output: AsmExcerpt = []
    for addr, instruction in excerpt:
        if _is_call(instruction):
            operand = _call_operand(instruction)
            if operand in name_set:
                # Direct call: `call name`
                instruction = "call " + CALL_PLACEHOLDER
            else:
                # Indirect call: `call dword ptr [name]`
                prefix, bracket, rest = operand.partition("[")
                inner, closing, suffix = rest.partition("]")
                if bracket and closing and inner in name_set:
                    instruction = f"call {prefix}[{CALL_PLACEHOLDER}]{suffix}"

        output.append((addr, instruction))

    return output
