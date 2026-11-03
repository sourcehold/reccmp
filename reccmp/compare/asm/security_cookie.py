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

* a load from a *named* memory location immediately followed by
  `xor <same register>, esp` (or `ebp`) is the cookie read;
* a direct CALL immediately preceded by `xor <register>, esp` (or `ebp`) is
  the call to the cookie check function.

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

        # The instruction before the XOR: the cookie read of the prologue.
        if index > 0:
            addr, previous = output[index - 1]
            load = _LOAD.match(previous)
            if (
                load is not None
                and load.group("reg") == register
                and load.group("name") in name_set
            ):
                output[index - 1] = (
                    addr,
                    load.group("prefix") + COOKIE_PLACEHOLDER + load.group("suffix"),
                )

        # The instruction after the XOR: the check call of the epilogue.
        if index + 1 < len(output):
            addr, following = output[index + 1]
            call = _CALL.match(following)
            if call is not None and call.group("name") in name_set:
                output[index + 1] = (addr, "call " + CHECK_PLACEHOLDER)

    return output
