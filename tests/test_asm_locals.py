"""Tests for naming the locals and parameters of a function in the diff output.

Unless noted otherwise, the assembly and the cvdump text in these tests were
taken from a real MSVC 14.00.50727.762 object file compiled from:

    struct Thing { int a; int b; void method(int p_first, int p_second); };

    void Thing::method(int p_first, int p_second)
    {
        int local_x = p_first + 1;
        int local_y = p_second + 2;
        char buf[16];
        buf[0] = (char)local_x;
        a = local_x + local_y + buf[0];
        b = p_first;
    }
"""

from unittest.mock import Mock
import pytest
from reccmp.compare.asm.locals import (
    FunctionLocals,
    LocalRegister,
    LocalSlot,
    annotate_locals,
)
from reccmp.compare.asm.parse import AsmExcerpt
from reccmp.compare.db import EntityDb
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.ingest import load_cvdump_locals
from reccmp.compare.lines import LinesDb
from reccmp.cvdump import CvdumpAnalysis, CvdumpParser
from reccmp.cvdump.symbols import CvdumpSymbolsParser
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.types import EntityType, ImageId
from .raw_image import RawImage


def excerpt(*instructions: str) -> AsmExcerpt:
    """An excerpt with made-up addresses, 4 bytes apart."""
    return [(0x1000 + 4 * i, text) for i, text in enumerate(instructions)]


def texts(annotated: AsmExcerpt) -> list[str]:
    return [text for _, text in annotated]


def test_no_locals_changes_nothing():
    """Without any locals there is nothing to add."""
    code = excerpt("mov eax, dword ptr [ebp + 8]", "ret")
    assert annotate_locals(code, FunctionLocals()) == code


def test_frame_pointer_slots():
    """With a frame pointer, the offset the PDB reports is the ebp
    displacement, so it can be used as is."""
    locals_ = FunctionLocals(
        frame_pointer=True,
        slots=[
            LocalSlot(-0x20, "this"),
            LocalSlot(8, "p_first"),
            LocalSlot(0xC, "p_second"),
            LocalSlot(-0x1C, "local_x"),
            LocalSlot(-0x18, "buf"),
            LocalSlot(-4, "local_y"),
        ],
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "push ebp",
                "mov ebp, esp",
                "sub esp, 0x20",
                "mov dword ptr [ebp - 0x20], ecx",
                "mov eax, dword ptr [ebp + 8]",
                "add eax, 1",
                "mov dword ptr [ebp - 0x1c], eax",
                "mov ecx, dword ptr [ebp + 0xc]",
                "mov dword ptr [ebp - 4], ecx",
                "mov dl, byte ptr [ebp - 0x1c]",
                "mov byte ptr [ebp - 0x18], dl",
                "mov esp, ebp",
                "pop ebp",
            ),
            locals_,
        )
    )

    assert annotated == [
        "push ebp",
        "mov ebp, esp",
        "sub esp, 0x20",
        "mov dword ptr [ebp - 0x20:this], ecx",
        "mov eax, dword ptr [ebp + 8:p_first]",
        "add eax, 1",
        "mov dword ptr [ebp - 0x1c:local_x], eax",
        "mov ecx, dword ptr [ebp + 0xc:p_second]",
        "mov dword ptr [ebp - 4:local_y], ecx",
        "mov dl, byte ptr [ebp - 0x1c:local_x]",
        "mov byte ptr [ebp - 0x18:buf], dl",
        "mov esp, ebp",
        "pop ebp",
    ]


def test_frame_pointer_unknown_slot():
    """A displacement that belongs to no variable is left alone. ebp is the
    frame pointer here, not a variable, so it is not named either."""
    locals_ = FunctionLocals(frame_pointer=True, slots=[LocalSlot(-4, "local_y")])
    annotated = texts(
        annotate_locals(excerpt("mov edx, dword ptr [ebp - 0x40]"), locals_)
    )
    assert annotated == ["mov edx, dword ptr [ebp - 0x40]"]


def test_no_frame_pointer_tracks_esp():
    """Without a frame pointer the PDB reports each slot relative to esp on
    entry, so the same variable appears at a different displacement once
    something has been pushed. Here `push esi` moves p_second from
    `[esp + 8]` to `[esp + 0xc]`."""
    locals_ = FunctionLocals(
        frame_pointer=False,
        slots=[LocalSlot(4, "p_first"), LocalSlot(8, "p_second")],
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov edx, dword ptr [esp + 4]",
                "push esi",
                "mov ecx, dword ptr [esp + 0xc]",
                "pop esi",
                "mov eax, dword ptr [esp + 8]",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov edx, dword ptr [esp + 4:p_first]",
        "push esi",
        "mov ecx, dword ptr [esp + 0xc:p_second]",
        "pop esi",
        "mov eax, dword ptr [esp + 8:p_second]",
    ]


def test_no_frame_pointer_sub_and_add_esp():
    """Reserving and releasing stack space shifts the displacements too."""
    locals_ = FunctionLocals(frame_pointer=False, slots=[LocalSlot(4, "p_one")])

    annotated = texts(
        annotate_locals(
            excerpt(
                "sub esp, 0x10",
                "mov eax, dword ptr [esp + 0x14]",
                "add esp, 0x10",
                "mov ecx, dword ptr [esp + 4]",
            ),
            locals_,
        )
    )

    assert annotated == [
        "sub esp, 0x10",
        "mov eax, dword ptr [esp + 0x14:p_one]",
        "add esp, 0x10",
        "mov ecx, dword ptr [esp + 4:p_one]",
    ]


def test_no_frame_pointer_stack_depth_lost():
    """Once esp is set to something we cannot follow, we stop naming stack
    slots rather than print a name that is probably wrong."""
    locals_ = FunctionLocals(frame_pointer=False, slots=[LocalSlot(4, "p_one")])

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov eax, dword ptr [esp + 4]",
                "and esp, 0xfffffff0",
                "mov ecx, dword ptr [esp + 4]",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov eax, dword ptr [esp + 4:p_one]",
        "and esp, 0xfffffff0",
        "mov ecx, dword ptr [esp + 4]",
    ]


def test_no_frame_pointer_ignores_ebp():
    """Without a frame pointer, an ebp displacement is not a frame access."""
    locals_ = FunctionLocals(frame_pointer=False, slots=[LocalSlot(4, "p_one")])
    annotated = texts(annotate_locals(excerpt("mov eax, dword ptr [ebp + 4]"), locals_))
    assert annotated == ["mov eax, dword ptr [ebp + 4]"]


def test_register_passed_in_by_caller():
    """`this` arrives in ecx. The PDB does not say for how long, so we name it
    until the first instruction that puts something else in ecx -- naming every
    later ecx would be wrong, as the register is reused here."""
    locals_ = FunctionLocals(
        frame_pointer=False, registers=[LocalRegister("ecx", "this")]
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov eax, ecx",
                "lea ecx, [edx + 1]",
                "add esi, ecx",
                "mov dword ptr [eax], ecx",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov eax, ecx:this",
        # ecx is rebound here, so the destination is not named...
        "lea ecx, [edx + 1]",
        # ...and nothing after it is either.
        "add esi, ecx",
        "mov dword ptr [eax], ecx",
    ]


def test_register_defined_inside_function():
    """A variable the function computes into a register is named from the
    instruction that produces it until the next one that overwrites it. The
    prologue `push esi` is register save/restore, not a use of the variable."""
    locals_ = FunctionLocals(
        frame_pointer=False, registers=[LocalRegister("esi", "ptr")]
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "push esi",
                "mov esi, dword ptr [esp + 8]",
                "test esi, esi",
                "mov eax, dword ptr [esi + 4]",
                "mov esi, 0",
                "pop esi",
            ),
            locals_,
        )
    )

    assert annotated == [
        "push esi",
        "mov esi:ptr, dword ptr [esp + 8]",
        "test esi:ptr, esi:ptr",
        # The base register of a dereference is named too, so the access
        # reads as ptr->field.
        "mov eax, dword ptr [esi:ptr + 4]",
        "mov esi, 0",
        "pop esi",
    ]


def test_register_read_modify_write_keeps_name():
    """`inc`/`add` update the variable rather than replace it."""
    locals_ = FunctionLocals(
        frame_pointer=False, registers=[LocalRegister("esi", "count")]
    )

    annotated = texts(
        annotate_locals(excerpt("inc esi", "add esi, 2", "mov edx, esi"), locals_)
    )
    assert annotated == ["inc esi:count", "add esi:count, 2", "mov edx, esi:count"]


def test_register_call_clobbers_volatiles():
    """A CALL may leave anything in eax/ecx/edx, but esi/edi/ebx survive it."""
    locals_ = FunctionLocals(
        frame_pointer=False,
        registers=[LocalRegister("ecx", "this"), LocalRegister("esi", "count")],
    )

    annotated = texts(
        annotate_locals(
            excerpt("mov eax, ecx", "mov edx, esi", "call Foo", "mov eax, ecx"),
            locals_,
        )
    )

    assert annotated == [
        "mov eax, ecx:this",
        "mov edx, esi:count",
        "call Foo",
        "mov eax, ecx",
    ]


def test_register_name_inside_symbol_is_left_alone():
    """Only operands are considered, so a register name that happens to appear
    inside a symbol is not touched."""
    locals_ = FunctionLocals(
        frame_pointer=False, registers=[LocalRegister("ecx", "this")]
    )

    annotated = texts(
        annotate_locals(
            excerpt("mov eax, ecx", "call ?get@?$Resolver@ecx@@AEXH@Z"), locals_
        )
    )
    assert annotated == ["mov eax, ecx:this", "call ?get@?$Resolver@ecx@@AEXH@Z"]


def test_block_scoped_variable():
    """A variable declared in a lexical block only exists in that part of the
    function, so it is only named there. Two sibling blocks may reuse one
    stack slot."""
    locals_ = FunctionLocals(
        frame_pointer=True,
        slots=[
            LocalSlot(-8, "j", start=0x1004, end=0x100C),
            LocalSlot(-8, "infoman", start=0x100C, end=0x1014),
        ],
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov eax, dword ptr [ebp - 8]",
                "mov ecx, dword ptr [ebp - 8]",
                "mov edx, dword ptr [ebp - 8]",
                "mov esi, dword ptr [ebp - 8]",
                "mov edi, dword ptr [ebp - 8]",
                "mov ebx, dword ptr [ebp - 8]",
            ),
            locals_,
        )
    )

    assert annotated == [
        # Before either block: in no scope.
        "mov eax, dword ptr [ebp - 8]",
        "mov ecx, dword ptr [ebp - 8:j]",
        "mov edx, dword ptr [ebp - 8:j]",
        "mov esi, dword ptr [ebp - 8:infoman]",
        "mov edi, dword ptr [ebp - 8:infoman]",
        # After both blocks.
        "mov ebx, dword ptr [ebp - 8]",
    ]


# This is the real cvdump output for the sample above, compiled with /Z7 /Od.
METHOD_OD = """\
(0000DC) S_GPROC32: [0001:00034E90], Cb: 00000055, Type:             0x1003, Thing::method
         Parent: 00000000, End: 00000000, Next: 00000000
         Debug start: 00000013, Debug end: 00000045
         Flags: Frame Ptr Present
(00012F)  S_FRAMECOOKIE: ebp+FFFFFFF8, Type: XOR_BP, Flags: 00
(00013B)  S_BPREL32: [FFFFFFE0], Type:             0x1001, this
(00014C)  S_BPREL32: [00000008], Type:       T_INT4(0074), p_first
(000160)  S_BPREL32: [0000000C], Type:       T_INT4(0074), p_second
(000175)  S_BPREL32: [FFFFFFE4], Type:       T_INT4(0074), local_x
(000189)  S_BPREL32: [FFFFFFE8], Type:             0x1006, buf
(000199)  S_BPREL32: [FFFFFFFC], Type:       T_INT4(0074), local_y
(0001AD) S_END
"""

# The same function compiled with /Z7 /O2. There is no frame pointer, the
# offsets are relative to esp on entry, and `this` stays in a register.
METHOD_O2 = """\
(000058) S_GPROC32: [0001:00034E90], Cb: 00000020, Type:             0x1003, Thing::method
         Parent: 00000000, End: 00000000, Next: 00000000
         Debug start: 00000006, Debug end: 0000001D
(0000AB)  S_REGISTER: ecx, Type:             0x1001, this
(0000BA)  S_BPREL32: [00000004], Type:       T_INT4(0074), p_first
(0000CE)  S_BPREL32: [00000008], Type:       T_INT4(0074), p_second
(0000E3) S_END
"""


class StubImage:
    """Just enough of a PE image to turn section:offset into an address."""

    def is_valid_section(self, section: int) -> bool:
        return section == 1

    def get_abs_addr(self, section: int, offset: int) -> int:
        assert section == 1
        return 0x10000000 + offset


def load_locals(symbols_text: str) -> dict[int, FunctionLocals]:
    parser = CvdumpParser()
    parser.read_section("SYMBOLS", symbols_text)
    return load_cvdump_locals(CvdumpAnalysis(parser), StubImage())  # type: ignore[arg-type]


def test_load_cvdump_locals_frame_pointer():
    """The /Od build keeps a frame pointer. cvdump prints a negative frame
    offset as an unsigned 32-bit hex value, which has to be sign extended."""
    locals_db = load_locals(METHOD_OD)

    locals_ = locals_db[0x10034E90]
    assert locals_.frame_pointer is True
    assert {slot.offset: slot.name for slot in locals_.slots} == {
        -0x20: "this",
        8: "p_first",
        0xC: "p_second",
        -0x1C: "local_x",
        -0x18: "buf",
        -4: "local_y",
    }
    assert locals_.registers == []


def test_load_cvdump_locals_no_frame_pointer():
    """The /O2 build has no frame pointer and puts `this` in a register."""
    locals_db = load_locals(METHOD_O2)

    locals_ = locals_db[0x10034E90]
    assert locals_.frame_pointer is False
    assert {slot.offset: slot.name for slot in locals_.slots} == {
        4: "p_first",
        8: "p_second",
    }
    assert [(reg.register, reg.name) for reg in locals_.registers] == [("ecx", "this")]


# Real MSVC 4.20 output: two sibling blocks, each with its own variable.
PROC_WITH_BLOCKS = """\
(000638) S_GPROC32: [0001:000C6135], Cb: 00000361, Type:             0x10ED, RegistrationBook::ReadyWorld
         Parent: 00000000, End: 00000760, Next: 00000000
         Debug start: 0000000C, Debug end: 0000035C
         Flags: Frame Ptr Present
(00067C)  S_BPREL32: [FFFFFFD0], Type:             0x10EC, this
(000704)  S_BLOCK32: [0001:000C624F], Cb: 000001DA,
          Parent: 00000638, End: 0000072C
(00071C)   S_BPREL32: [FFFFFFD8], Type:      T_SHORT(0011), j
(00072C)  S_END
(000730)  S_BLOCK32: [0001:000C6448], Cb: 00000032,
          Parent: 00000638, End: 0000075C
(000748)   S_BPREL32: [FFFFFFD4], Type:             0x10FA, infoman
(00075C)  S_END
(000760) S_END
"""


def test_cvdump_symbols_block_scope():
    """A variable declared inside an S_BLOCK32 records the range of that
    block; one at function scope records nothing."""
    parser = CvdumpSymbolsParser()
    for line in PROC_WITH_BLOCKS.split("\n"):
        parser.read_line(line)

    (function,) = parser.symbols
    scopes = {symbol.name: symbol.scope for symbol in function.symbols}

    assert scopes["this"] is None
    assert scopes["j"] is not None
    assert (scopes["j"].offset, scopes["j"].size) == (0xC624F, 0x1DA)
    assert scopes["infoman"] is not None
    assert (scopes["infoman"].offset, scopes["infoman"].size) == (0xC6448, 0x32)


def test_cvdump_symbols_nested_block_scope():
    """The innermost enclosing block is the one that counts."""
    parser = CvdumpSymbolsParser()
    text = """\
(000638) S_GPROC32: [0001:00001000], Cb: 00000100, Type:             0x10ED, outer
         Flags: Frame Ptr Present
(000704)  S_BLOCK32: [0001:00001010], Cb: 00000080,
(00071C)   S_BPREL32: [FFFFFFFC], Type:      T_SHORT(0011), in_outer_block
(000704)   S_BLOCK32: [0001:00001020], Cb: 00000010,
(00071C)    S_BPREL32: [FFFFFFF8], Type:      T_SHORT(0011), in_inner_block
(00072C)   S_END
(00072C)  S_END
(000760) S_END
"""
    for line in text.split("\n"):
        parser.read_line(line)

    (function,) = parser.symbols
    scopes = {symbol.name: symbol.scope for symbol in function.symbols}

    assert scopes["in_outer_block"] is not None
    assert scopes["in_outer_block"].offset == 0x1010
    assert scopes["in_inner_block"] is not None
    assert scopes["in_inner_block"].offset == 0x1020


def test_load_cvdump_locals_block_scope_addresses():
    """The block range is converted to recomp addresses."""
    locals_db = load_locals(PROC_WITH_BLOCKS)

    locals_ = locals_db[0x100C6135]
    by_name = {slot.name: slot for slot in locals_.slots}

    assert (by_name["this"].start, by_name["this"].end) == (None, None)
    assert (by_name["j"].start, by_name["j"].end) == (
        0x100C624F,
        0x100C624F + 0x1DA,
    )
    assert (by_name["infoman"].start, by_name["infoman"].end) == (
        0x100C6448,
        0x100C6448 + 0x32,
    )


@pytest.fixture(name="report")
def fixture_report_mock() -> ReccmpReportProtocol:
    return Mock(spec=ReccmpReportProtocol)


def test_annotation_does_not_affect_the_diff(report: ReccmpReportProtocol):
    """The names are added to the text we print, not the text we diff, so an
    identical function is still a 100% match with them."""
    orig_base = 0x401000
    recomp_base = 0x10001000

    # mov eax, [ebp + 8]; mov [ebp - 4], eax; ret
    code = b"\x8b\x45\x08\x89\x45\xfc\xc3"

    db = EntityDb()
    with db.batch() as batch:
        for image_id, addr in (
            (ImageId.ORIG, orig_base),
            (ImageId.RECOMP, recomp_base),
        ):
            batch.set(
                image_id, addr, type=EntityType.FUNCTION, name="test", size=len(code)
            )
        batch.match(orig_base, recomp_base)

    comparator = FunctionComparator(
        db,
        LinesDb(),
        RawImage.from_memory(code, base_addr=orig_base),
        RawImage.from_memory(code, base_addr=recomp_base),
        report,
        CvdumpTypesParser(),
        locals_db={
            recomp_base: FunctionLocals(
                frame_pointer=True,
                slots=[LocalSlot(8, "p_value"), LocalSlot(-4, "result")],
            )
        },
    )

    (entity,) = list(db.get_functions())
    result = comparator.compare_function(entity)

    assert result.match_ratio == 1.0

    # The orig side has no PDB, so it is not annotated.
    orig_text = [text for _, text in result.diff.orig_inst]
    recomp_text = [text for _, text in result.diff.recomp_inst]

    assert orig_text == [
        "mov eax, dword ptr [ebp + 8]",
        "mov dword ptr [ebp - 4], eax",
        # capstone writes a trailing space for an instruction with no operands.
        "ret ",
    ]
    assert recomp_text == [
        "mov eax, dword ptr [ebp + 8:p_value]",
        "mov dword ptr [ebp - 4:result], eax",
        "ret ",
    ]


def test_register_passed_in_but_never_used():
    """A parameter that arrives in a register and is overwritten without ever
    being read tells us nothing useful, so it is not named. This is the real
    shape of OpenSHC's `queueUnitTypeCommand`, which never touches `this` and
    loads a different object into ecx for its tail call."""
    locals_ = FunctionLocals(
        frame_pointer=False,
        slots=[LocalSlot(4, "unitType")],
        registers=[LocalRegister("ecx", "this")],
    )

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov eax, dword ptr [esp + 4]",
                "mov dword ptr [<OFFSET1>], 0xffffffff",
                "mov dword ptr [<OFFSET2>], eax",
                "mov dword ptr [esp + 4], 0xf",
                "mov ecx, <OFFSET3>",
                "jmp -0xadbb5",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov eax, dword ptr [esp + 4:unitType]",
        "mov dword ptr [<OFFSET1>], 0xffffffff",
        "mov dword ptr [<OFFSET2>], eax",
        "mov dword ptr [esp + 4:unitType], 0xf",
        # ecx is loaded with something else here and never read again.
        "mov ecx, <OFFSET3>",
        "jmp -0xadbb5",
    ]


def test_no_frame_pointer_stops_at_a_branch_target():
    """We read the function as a straight line. The instruction after a `ret`
    partway through is reached by a branch, and the stack need not be as deep
    there, so we stop naming slots rather than guess. This is the shape MSVC
    emits for an early return out of an optimized function."""
    locals_ = FunctionLocals(frame_pointer=False, slots=[LocalSlot(4, "_Count")])

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov ecx, dword ptr [esp + 4]",
                "sub esp, 0xc",
                "ja 0x1a",
                "add esp, 0xc",
                "ret ",
                # Only reachable through the `ja` above, where esp is 12 lower.
                "lea ecx, [esp + 4]",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov ecx, dword ptr [esp + 4:_Count]",
        "sub esp, 0xc",
        "ja 0x1a",
        "add esp, 0xc",
        "ret ",
        "lea ecx, [esp + 4]",
    ]


def test_frame_pointer_unaffected_by_branch_targets():
    """An ebp displacement does not depend on how deep the stack is, so a
    frame-pointer function is named throughout."""
    locals_ = FunctionLocals(frame_pointer=True, slots=[LocalSlot(8, "p_one")])

    annotated = texts(
        annotate_locals(
            excerpt(
                "mov eax, dword ptr [ebp + 8]",
                "ret ",
                "mov ecx, dword ptr [ebp + 8]",
            ),
            locals_,
        )
    )

    assert annotated == [
        "mov eax, dword ptr [ebp + 8:p_one]",
        "ret ",
        "mov ecx, dword ptr [ebp + 8:p_one]",
    ]
