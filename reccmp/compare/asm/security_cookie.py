"""Normalization of the MSVC stack security cookie (`/GS`) instructions.

A function compiled with `/GS` that needs a stack guard reads the global
cookie in its prologue and validates it in its epilogue:

    mov eax, dword ptr [__security_cookie]
    xor eax, esp
    mov dword ptr [esp + 0x1c], eax
    ...
    mov ecx, dword ptr [esp + 0x1c]
    xor ecx, esp
    call __security_check_cookie

The cookie global and the check function are emitted by the compiler and the
CRT, not by the source. The recomp binary therefore has its own copy of both,
living at its own address, while the original binary reads its own. reccmp
names each side after the entity it finds at that address, so the two
instructions read e.g.

    -mov eax, dword ptr [StructResolver::Instance<unsigned int,12141088>::instance]
    +mov eax, dword ptr [__security_cookie]

Nothing is wrong with the recompiled code here, but every `/GS` function (any
function with a local array, in practice) loses those lines to the diff and
cannot reach a 100% match.

`ignore_security_cookie` replaces the cookie operand with a constant
placeholder on both sides, so that those instructions are equal. The idiom is
recognized by its shape, not by the name of the symbol, because the two
binaries do not agree on the name:

* a load from a *named* memory location that is the last thing to assign
  `<reg>` before `xor <reg>, esp` (or `ebp`) is the cookie read;
* the first direct CALL after that XOR is the call to the cookie check
  function, as long as nothing in between reassigns `<reg>`.

Neither half has to be adjacent to the XOR. The compiler schedules unrelated
work into the gap, and an epilogue of

    xor ecx, esp
    mov dword ptr [SomeGlobal + 4], 1
    call __security_check_cookie

is common enough that requiring adjacency leaves the call in the diff and
costs the function its 100%.

Only the operand is replaced, so the instructions themselves, the register
allocation and the stack slot of the cookie all still count towards the diff.
"""

import re
from typing import Iterable

from reccmp.compare.asm.parse import AsmExcerpt

COOKIE_PLACEHOLDER = "<SECURITY_COOKIE>"
CHECK_PLACEHOLDER = "<SECURITY_CHECK_COOKIE>"

_REGISTER = r"[er][a-z0-9]{2}"

# `mov eax, dword ptr [name]`: a load from a memory location given by name
# (i.e. an absolute address that the sanitizer replaced). The operand of a
# stack read such as `[esp + 0x1c]` is not a name, so it never matches.
_LOAD = re.compile(
    rf"^(?P<prefix>mov (?P<reg>{_REGISTER}), (?:[a-z]+ ptr )?\[)"
    r"(?P<name>[^\[\]]+)(?P<suffix>\])$"
)

# `xor eax, esp`: mixing the stack pointer into the cookie. This is what tells
# the cookie apart from any other global that a function reads.
_XOR_FRAME = re.compile(rf"^xor (?P<reg>{_REGISTER}), (?:esp|ebp|rsp|rbp)$")

_CALL = re.compile(r"^call (?P<name>.+)$")

# Instructions that end a basic block. The idiom is only the idiom while the
# load, the XOR and the check call are on one straight-line path, so the search
# stops at any branch and at an unrelated CALL.
_BRANCH = re.compile(r"^(j[a-z]+|call|ret|loop[a-z]*|int3?)\b")

# How far the two halves may drift apart. The compiler is free to schedule
# unrelated work between them, but it keeps the cookie close to the frame it
# guards, so a small window separates a real idiom from a coincidence.
_MAX_SCHEDULING_DISTANCE = 4

# Intel syntax: the destination is the first operand.
_DESTINATION = re.compile(r"^[a-z][a-z0-9]*\s+(?P<dest>[^,]+)")


def _register_parts(register: str) -> set[str]:
    """The names that alias `register`, so that a write to any of them counts
    as reassigning it: `mov cl, 0` clobbers part of `ecx`."""
    name = register.lower()
    parts = {name}
    if len(name) == 3 and name[0] in "er":
        base = name[1:]
        parts |= {"e" + base, "r" + base, base}
        if base.endswith("x"):
            parts |= {base[0] + "l", base[0] + "h"}
    return parts


def _writes_register(instruction: str, parts: set[str]) -> bool:
    """Whether `instruction` assigns the register named by `parts`. A memory
    destination such as `[ecx + 4]` only reads it, so it does not count."""
    match = _DESTINATION.match(instruction)
    return match is not None and match.group("dest").strip() in parts


def ignore_security_cookie(excerpt: AsmExcerpt, names: Iterable[str]) -> AsmExcerpt:
    """Replace the security cookie global and the security check function with
    a placeholder wherever the `/GS` idiom appears in the given function.

    `names` are the names and placeholders that the sanitizer substituted into
    this excerpt. An operand that is not one of them is not an address we know
    about, so it is left alone.
    """
    name_set = set(names)
    if not name_set:
        return excerpt

    output: AsmExcerpt = list(excerpt)

    for index, (_, instruction) in enumerate(output):
        xor_match = _XOR_FRAME.match(instruction)
        if xor_match is None:
            continue

        register = xor_match.group("reg")
        parts = _register_parts(register)

        # Backwards from the XOR: the cookie read of the prologue. The nearest
        # instruction that assigns the register has to be that load - if
        # something else assigned it in between, the value being mixed with the
        # stack pointer did not come from the global.
        is_prologue = False
        for candidate in range(index - 1, max(index - 1 - _MAX_SCHEDULING_DISTANCE, -1), -1):
            addr, previous = output[candidate]
            if _BRANCH.match(previous):
                break
            if not _writes_register(previous, parts):
                continue
            load = _LOAD.match(previous)
            if load is not None and load.group("name") in name_set:
                output[candidate] = (
                    addr,
                    load.group("prefix") + COOKIE_PLACEHOLDER + load.group("suffix"),
                )
                is_prologue = True
            # Either way this was the nearest write, so there is nothing
            # further back to attribute the XOR's operand to.
            break

        # A prologue XOR mixes the *global* into the stack pointer and is
        # followed by the store into the guard slot, never by the check call.
        # Searching forward from it would claim whatever the function calls
        # next, so stop here.
        if is_prologue:
            continue

        # Forwards from the XOR: the check call of the epilogue, whose operand
        # came from the guard slot rather than from the global. Anything
        # scheduled in between has to leave the register alone, since the
        # callee takes the cookie in it.
        for candidate in range(index + 1, min(index + 1 + _MAX_SCHEDULING_DISTANCE, len(output))):
            addr, following = output[candidate]
            call = _CALL.match(following)
            if call is not None:
                if call.group("name") in name_set:
                    output[candidate] = (addr, "call " + CHECK_PLACEHOLDER)
                break
            if _BRANCH.match(following) or _writes_register(following, parts):
                break

    return output
