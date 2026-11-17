"""Tests for the symbolic equivalence check behind `--semantic-match`."""

import pytest
from reccmp.compare.asm.equivalence import (
    CalleeInfo,
    find_semantic_match,
    return_kind_from_symbol,
)


def listing(text: str, base: int = 0x1000) -> list[tuple[int, str]]:
    """Assemble a listing from one instruction per line. A line ending in `:` is a
    label; a jump names its target as `@label`. A jump is taken to be a short jump
    (two bytes) and every other instruction four bytes long, which is all the
    checker needs: it reads a displacement relative to the next instruction."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    def size(line: str) -> int:
        return 2 if "@" in line else 4

    addrs: dict[str, int] = {}
    addr = base
    for line in lines:
        if line.endswith(":"):
            addrs[line[:-1]] = addr
        else:
            addr += size(line)

    out = []
    addr = base
    for line in lines:
        if line.endswith(":"):
            continue
        end = addr + size(line)
        if "@" in line:
            mnemonic, label = line.split("@")
            line = f"{mnemonic.strip()} {(addrs[label] - end) & 0xFFFFFFFF:#x}"
        out.append((addr, line))
        addr = end
    return out


INT = return_kind_from_symbol("?f@@YAHXZ")
VOID = return_kind_from_symbol("?f@@YAXXZ")


def equivalent(orig: str, recomp: str, **kwargs) -> bool:
    kwargs.setdefault("returns", INT)
    ok, _ = find_semantic_match(listing(orig), listing(recomp, 0x5000), **kwargs)
    return ok


def test_identical():
    code = """
        mov eax, dword ptr [ecx + 4]
        add eax, 1
        ret
    """
    assert equivalent(code, code)


def test_register_swap():
    orig = """
        mov eax, dword ptr [ecx + 4]
        mov edx, dword ptr [ecx + 8]
        add eax, edx
        ret
    """
    recomp = """
        mov edx, dword ptr [ecx + 4]
        mov eax, dword ptr [ecx + 8]
        add eax, edx
        ret
    """
    assert equivalent(orig, recomp)


def test_different_constant():
    orig = """
        mov eax, dword ptr [ecx + 4]
        add eax, 1
        ret
    """
    recomp = """
        mov eax, dword ptr [ecx + 4]
        add eax, 2
        ret
    """
    assert not equivalent(orig, recomp)


def test_different_field():
    orig = """
        mov eax, dword ptr [ecx + 4]
        ret
    """
    recomp = """
        mov eax, dword ptr [ecx + 8]
        ret
    """
    assert not equivalent(orig, recomp)


def test_lea_against_add():
    orig = """
        mov eax, dword ptr [esp + 4]
        lea eax, [eax + eax*2 + 5]
        ret
    """
    recomp = """
        mov ecx, dword ptr [esp + 4]
        mov eax, ecx
        add eax, ecx
        add eax, ecx
        add eax, 5
        ret
    """
    assert equivalent(orig, recomp)


def test_stack_slot_swap():
    """Two locals that trade places in the frame."""
    orig = """
        sub esp, 8
        mov eax, dword ptr [esp + 0xc]
        mov dword ptr [esp], eax
        mov eax, dword ptr [esp + 0x10]
        mov dword ptr [esp + 4], eax
        mov eax, dword ptr [esp]
        sub eax, dword ptr [esp + 4]
        add esp, 8
        ret
    """
    recomp = """
        sub esp, 8
        mov eax, dword ptr [esp + 0xc]
        mov dword ptr [esp + 4], eax
        mov eax, dword ptr [esp + 0x10]
        mov dword ptr [esp], eax
        mov eax, dword ptr [esp + 4]
        sub eax, dword ptr [esp]
        add esp, 8
        ret
    """
    assert equivalent(orig, recomp)


def test_inverted_branch_and_swapped_blocks():
    orig = """
        mov eax, dword ptr [esp + 4]
        test eax, eax
        je @zero
        mov eax, 1
        ret
    zero:
        mov eax, 2
        ret
    """
    recomp = """
        mov eax, dword ptr [esp + 4]
        test eax, eax
        jne @nonzero
        mov eax, 2
        ret
    nonzero:
        mov eax, 1
        ret
    """
    assert equivalent(orig, recomp)


def test_off_by_one_condition():
    orig = """
        cmp dword ptr [esp + 4], 5
        jl @less
        xor eax, eax
        ret
    less:
        mov eax, 1
        ret
    """
    recomp = """
        cmp dword ptr [esp + 4], 5
        jle @less
        xor eax, eax
        ret
    less:
        mov eax, 1
        ret
    """
    assert not equivalent(orig, recomp)


def test_signedness_of_condition():
    orig = """
        cmp dword ptr [esp + 4], 5
        jl @less
        xor eax, eax
        ret
    less:
        mov eax, 1
        ret
    """
    recomp = """
        cmp dword ptr [esp + 4], 5
        jb @less
        xor eax, eax
        ret
    less:
        mov eax, 1
        ret
    """
    assert not equivalent(orig, recomp)


def test_loop_with_renamed_registers():
    orig = """
        mov ecx, dword ptr [esp + 4]
        xor eax, eax
        xor edx, edx
    top:
        add eax, dword ptr [ecx + edx*4]
        inc edx
        cmp edx, 10
        jl @top
        ret
    """
    recomp = """
        mov edx, dword ptr [esp + 4]
        xor eax, eax
        xor ecx, ecx
    top:
        add eax, dword ptr [edx + ecx*4]
        inc ecx
        cmp ecx, 10
        jl @top
        ret
    """
    assert equivalent(orig, recomp)


def test_loop_with_different_bound():
    orig = """
        mov ecx, dword ptr [esp + 4]
        xor eax, eax
        xor edx, edx
    top:
        add eax, dword ptr [ecx + edx*4]
        inc edx
        cmp edx, 10
        jl @top
        ret
    """
    recomp = """
        mov ecx, dword ptr [esp + 4]
        xor eax, eax
        xor edx, edx
    top:
        add eax, dword ptr [ecx + edx*4]
        inc edx
        cmp edx, 11
        jl @top
        ret
    """
    assert not equivalent(orig, recomp)


def test_callee_saved_register():
    """Using esi and saving it matches using ebx and saving it, but not using
    ebx without saving it."""
    orig = """
        push esi
        mov esi, dword ptr [esp + 8]
        mov eax, dword ptr [esi]
        add eax, dword ptr [esi + 4]
        pop esi
        ret
    """
    saved = """
        push ebx
        mov ebx, dword ptr [esp + 8]
        mov eax, dword ptr [ebx]
        add eax, dword ptr [ebx + 4]
        pop ebx
        ret
    """
    clobbered = """
        mov ebx, dword ptr [esp + 4]
        mov eax, dword ptr [ebx]
        add eax, dword ptr [ebx + 4]
        ret
    """
    assert equivalent(orig, saved)
    assert not equivalent(orig, clobbered)


def test_call_arguments():
    orig = """
        push 2
        push 1
        call func (FUNCTION)
        add esp, 8
        ret
    """
    same = """
        push 2
        push 1
        call func (FUNCTION)
        add esp, 8
        ret
    """
    swapped = """
        push 1
        push 2
        call func (FUNCTION)
        add esp, 8
        ret
    """
    assert equivalent(orig, same)
    assert not equivalent(orig, swapped)


def test_call_argument_through_register():
    """An argument pushed from a register is the same as the value pushed directly."""
    orig = """
        mov eax, dword ptr [ecx + 4]
        push eax
        call func (FUNCTION)
        ret
    """
    recomp = """
        push dword ptr [ecx + 4]
        call func (FUNCTION)
        ret
    """
    info = CalleeInfo(pops=4)
    assert equivalent(orig, recomp, callee=lambda _: info)


def test_store_order_around_call():
    """A store cannot move across a call: the callee may read it."""
    orig = """
        mov dword ptr [ecx], 1
        call func (FUNCTION)
        ret
    """
    recomp = """
        call func (FUNCTION)
        mov dword ptr [ecx], 1
        ret
    """
    assert not equivalent(orig, recomp)


def test_return_value_ignored_for_void():
    """A void function may leave anything in eax."""
    orig = """
        mov eax, 1
        mov dword ptr [ecx], eax
        ret
    """
    recomp = """
        mov edx, 1
        mov dword ptr [ecx], edx
        ret
    """
    void = return_kind_from_symbol("?f@@YAXXZ")
    assert equivalent(orig, recomp, returns=void)
    assert not equivalent(orig, recomp, returns=return_kind_from_symbol("?f@@YAHXZ"))


@pytest.mark.parametrize(
    "symbol, eax, edx",
    [
        ("?f@@YAXXZ", False, False),  # void
        ("?f@@YAHXZ", True, False),  # int
        ("?f@@YA_JXZ", True, True),  # __int64
        ("??0Class@@QAE@XZ", True, False),  # constructor
        (None, True, True),  # unknown
    ],
)
def test_return_kind_from_symbol(symbol, eax, edx):
    kind = return_kind_from_symbol(symbol)
    assert kind.eax == eax
    assert kind.edx == edx


def test_tail_call_by_name():
    """A jump out of the function is a tail call, compared by the name of its target."""
    code = """
        mov ecx, dword ptr [ecx + 4]
        jmp @far
        ret
    far:
    """
    orig, recomp = listing(code), listing(code, 0x5000)
    orig_target, recomp_target = orig[-1][0] + 4, recomp[-1][0] + 4

    def naming(names: dict[int, str]):
        return lambda _img, addr: names.get(addr)

    same = naming({orig_target: "f", recomp_target: "f"})
    different = naming({orig_target: "f", recomp_target: "g"})
    unknown = naming({orig_target: "f"})
    assert find_semantic_match(orig, recomp, tail_name=same)[0]
    assert not find_semantic_match(orig, recomp, tail_name=different)[0]
    # Some target has no name: all are compared by displacement, like the text diff.
    assert find_semantic_match(orig, recomp, tail_name=unknown)[0]
    assert find_semantic_match(orig, recomp)[0]
    # ... and a different displacement is a different target.
    farther = listing(code.replace("@far", "@farther") + "nop\nfarther:", 0x5000)
    assert not find_semantic_match(orig, farther)[0]


def test_partly_overwritten_slot():
    """A byte written into a dword local, then the dword read back, with the local at
    a different place in each frame."""
    orig = """
        sub esp, 8
        mov eax, dword ptr [esp + 0xc]
        mov dword ptr [esp], eax
        mov byte ptr [esp + 1], 7
        mov eax, dword ptr [esp]
        add esp, 8
        ret
    """
    recomp = """
        sub esp, 8
        mov eax, dword ptr [esp + 0xc]
        mov dword ptr [esp + 4], eax
        mov byte ptr [esp + 5], 7
        mov eax, dword ptr [esp + 4]
        add esp, 8
        ret
    """
    other_byte = recomp.replace("[esp + 5], 7", "[esp + 6], 7")
    assert equivalent(orig, recomp)
    assert not equivalent(orig, other_byte)


def test_local_struct_passed_by_pointer():
    """A callee reads a local struct through the pointer it is given: its contents are
    compared, wherever each side keeps it in the frame."""
    orig = """
        sub esp, 8
        mov dword ptr [esp], 1
        mov dword ptr [esp + 4], 2
        mov eax, esp
        push eax
        call func (FUNCTION)
        add esp, 0xc
        ret
    """
    moved = """
        sub esp, 0xc
        mov dword ptr [esp + 4], 1
        mov dword ptr [esp + 8], 2
        lea eax, [esp + 4]
        push eax
        call func (FUNCTION)
        add esp, 0x10
        ret
    """
    different = orig.replace("[esp + 4], 2", "[esp + 4], 3")
    info = CalleeInfo(pops=0)
    assert equivalent(orig, moved, callee=lambda _: info)
    assert not equivalent(orig, different, callee=lambda _: info)


def test_return_value_through_cookie_check():
    """__security_check_cookie keeps eax: the value set before it is the return value."""
    code = """
        sub esp, 8
        mov eax, dword ptr [<SECURITY_COOKIE>]
        xor eax, esp
        mov dword ptr [esp + 4], eax
        mov eax, dword ptr [esp + 0xc]
        mov ecx, dword ptr [esp + 4]
        xor ecx, esp
        call <SECURITY_CHECK_COOKIE>
        add esp, 8
        ret
    """
    other = code.replace(
        "mov eax, dword ptr [esp + 0xc]", "mov eax, dword ptr [esp + 0x10]"
    )
    assert equivalent(code, code)
    assert not equivalent(code, other)


def test_argument_pushed_before_a_nested_call():
    """f(g(1), 7): 7 is pushed before g is called and is still f's argument afterwards."""
    orig = """
        push 7
        push 1
        call g (FUNCTION)
        add esp, 4
        push eax
        call f (FUNCTION)
        add esp, 8
        ret
    """
    other = orig.replace("push 7", "push 8")
    info = CalleeInfo(pops=0, reads_ecx=False, reads_edx=False)
    assert equivalent(orig, orig, callee=lambda _: info)
    assert not equivalent(orig, other, callee=lambda _: info)


def test_tail_call_takes_our_arguments():
    """A tail call passes on the argument slots, including ones this function overwrote."""
    orig = """
        mov eax, dword ptr [ecx + 4]
        mov dword ptr [esp + 4], eax
        jmp dword ptr [USER32.dll::SetCursor (IMPORT)]
    """
    other = orig.replace("mov dword ptr [esp + 4], eax", "nop")
    assert equivalent(orig, orig)
    assert not equivalent(orig, other)


def test_global_store_after_frame_pointer_escapes():
    """A store through a pointer loaded from memory is a store, even once the frame escaped."""
    orig = """
        sub esp, 4
        lea eax, [esp]
        push eax
        call g (FUNCTION)
        mov ecx, dword ptr [G (DATA)]
        mov dword ptr [ecx + 8], 1
        add esp, 4
        ret
    """
    other = orig.replace("mov dword ptr [ecx + 8], 1", "mov dword ptr [ecx + 8], 2")
    info = CalleeInfo(pops=4, reads_ecx=False, reads_edx=False)
    assert equivalent(orig, orig, callee=lambda _: info, returns=VOID)
    assert not equivalent(orig, other, callee=lambda _: info, returns=VOID)
