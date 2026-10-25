"""Tests for telling an integer constant apart from an address.

A bare number in the code is ambiguous: `push 0x1950000` is a pointer if
0x1950000 is a variable and an integer if it is not. All we have to go on is
whether we know of something at that address, which is weak evidence once a
large struct or a function covers a wide range of addresses -- any constant
that lands inside one then looks like a pointer into it, and gets named on the
orig side only, which costs the function its 100%.

The addresses here are OpenSHC's real ones: DAT_GameSynchronyState sits at
0x0191D768 in Stronghold Crusader and has a field at offset 0x7A850, so the
one struct claims some 500 KB of the original address space.
"""

from unittest.mock import Mock
import pytest
from reccmp.compare.asm.parse import ParseAsm, shared_constants
from reccmp.compare.asm.replacement import create_name_lookup
from reccmp.compare.db import EntityDb
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.types import EntityType, ImageId
from .raw_image import RawImage

ORIG_FUNC = 0x401000
RECOMP_FUNC = 0x10001000

ORIG_STRUCT = 0x0191D768
RECOMP_STRUCT = 0x10050000
STRUCT_SIZE = 0x7A858

# A constant that happens to land inside the struct in the original binary.
CONSTANT_IN_STRUCT = 0x1950000
assert ORIG_STRUCT < CONSTANT_IN_STRUCT < ORIG_STRUCT + STRUCT_SIZE


def push(value: int) -> bytes:
    return b"\x68" + value.to_bytes(4, "little")


@pytest.fixture(name="report")
def fixture_report_mock() -> ReccmpReportProtocol:
    return Mock(spec=ReccmpReportProtocol)


def compare(
    report: ReccmpReportProtocol,
    orig_code: bytes,
    recomp_code: bytes,
    *,
    orig_extra: dict[int, dict] | None = None,
):
    """Compare two functions, with the big struct matched between the images."""
    db = EntityDb()
    with db.batch() as batch:
        for image_id, addr, code in (
            (ImageId.ORIG, ORIG_FUNC, orig_code),
            (ImageId.RECOMP, RECOMP_FUNC, recomp_code),
        ):
            batch.set(
                image_id, addr, type=EntityType.FUNCTION, name="test", size=len(code)
            )
        batch.match(ORIG_FUNC, RECOMP_FUNC)

        batch.set(
            ImageId.ORIG,
            ORIG_STRUCT,
            type=EntityType.DATA,
            name="DAT_GameSynchronyState",
            size=STRUCT_SIZE,
        )
        batch.set(
            ImageId.RECOMP,
            RECOMP_STRUCT,
            type=EntityType.DATA,
            name="DAT_GameSynchronyState",
            size=STRUCT_SIZE,
        )
        batch.match(ORIG_STRUCT, RECOMP_STRUCT)

        for addr, values in (orig_extra or {}).items():
            batch.set(ImageId.ORIG, addr, **values)

    comparator = FunctionComparator(
        db,
        LinesDb(),
        RawImage.from_memory(orig_code, base_addr=ORIG_FUNC),
        RawImage.from_memory(recomp_code, base_addr=RECOMP_FUNC),
        report,
        CvdumpTypesParser(),
    )
    (entity,) = (e for e in db.get_functions() if e.orig_addr == ORIG_FUNC)
    result = comparator.compare_function(entity)
    return (
        result,
        [text for _, text in result.diff.orig_inst],
        [text for _, text in result.diff.recomp_inst],
    )


def test_constant_inside_a_struct(report: ReccmpReportProtocol):
    """The same number on both sides is a constant from the source, not a
    pointer into the struct that happens to span it."""
    code = push(CONSTANT_IN_STRUCT) + b"\xc3"
    result, orig, recomp = compare(report, code, code)

    assert orig == ["push 0x1950000", "ret "]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_constant_inside_a_function(report: ReccmpReportProtocol):
    """Same for a number that lands inside the code of a known function. We
    cannot name an offset into a function, so this one used to be replaced
    with an <OFFSET> placeholder on the orig side."""
    code = push(0x536C95) + b"\xc3"
    result, orig, recomp = compare(
        report,
        code,
        code,
        orig_extra={
            0x536C90: {"type": EntityType.FUNCTION, "name": "someFunc", "size": 0x25}
        },
    )

    assert orig == ["push 0x536c95", "ret "]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_pointer_to_a_variable_is_still_named(report: ReccmpReportProtocol):
    """A real pointer has a different value in each image, so it is not
    mistaken for a constant and still gets its name."""
    result, orig, recomp = compare(
        report, push(ORIG_STRUCT) + b"\xc3", push(RECOMP_STRUCT) + b"\xc3"
    )

    assert orig == ["push DAT_GameSynchronyState (DATA)", "ret "]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_pointer_into_a_variable_is_still_named(report: ReccmpReportProtocol):
    """The same for the address of a field, which is named with its offset.
    This is the case that stops us from simply never naming an offset."""
    result, orig, recomp = compare(
        report,
        push(ORIG_STRUCT + 0x7A850) + b"\xc3",
        push(RECOMP_STRUCT + 0x7A850) + b"\xc3",
    )

    assert orig == ["push DAT_GameSynchronyState+501840 (OFFSET)", "ret "]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_constant_and_pointer_together(report: ReccmpReportProtocol):
    """A function may well use both. Naming one must not stop us leaving the
    other alone, and the placeholder numbering has to line up afterwards."""
    orig_code = push(CONSTANT_IN_STRUCT) + push(ORIG_STRUCT) + push(0x1234) + b"\xc3"
    recomp_code = (
        push(CONSTANT_IN_STRUCT) + push(RECOMP_STRUCT) + push(0x1234) + b"\xc3"
    )
    result, orig, recomp = compare(report, orig_code, recomp_code)

    assert orig == [
        "push 0x1950000",
        "push DAT_GameSynchronyState (DATA)",
        "push 0x1234",
        "ret ",
    ]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_cmp_against_a_constant(report: ReccmpReportProtocol):
    """The immediate of a CMP is read the same way."""
    # cmp eax, 0x1950000 ; ret
    code = b"\x3d" + CONSTANT_IN_STRUCT.to_bytes(4, "little") + b"\xc3"
    result, orig, recomp = compare(report, code, code)

    assert orig == ["cmp eax, 0x1950000", "ret "]
    assert recomp == orig
    assert result.match_ratio == 1.0


def test_memory_operand_is_always_an_address(report: ReccmpReportProtocol):
    """A number used to address memory is not ambiguous, so it is left to the
    usual treatment even when both sides use the same one. Reading the same
    absolute address in both images really is a difference worth reporting."""
    # mov eax, dword ptr [0x1950000] ; ret
    code = b"\xa1" + CONSTANT_IN_STRUCT.to_bytes(4, "little") + b"\xc3"
    _, orig, recomp = compare(report, code, code)

    assert orig == [
        "mov eax, dword ptr [DAT_GameSynchronyState+207000 (OFFSET)]",
        "ret ",
    ]
    assert recomp == ["mov eax, dword ptr [<OFFSET1>]", "ret "]


def test_a_real_pointer_on_one_side_only(report: ReccmpReportProtocol):
    """A value only one side uses is not evidence of anything, so it is left
    to the usual treatment."""
    result, orig, recomp = compare(
        report, push(ORIG_STRUCT) + b"\xc3", push(0x1234) + b"\xc3"
    )

    assert orig == ["push DAT_GameSynchronyState (DATA)", "ret "]
    assert recomp == ["push 0x1234", "ret "]
    assert result.match_ratio != 1.0


def parser(db: EntityDb, image_id: ImageId) -> ParseAsm:
    types = CvdumpTypesParser()
    return ParseAsm(
        addr_test=lambda addr: db.intersects(image_id, addr),
        name_lookup=create_name_lookup(
            db, image_id, lambda _addr: None, types.get_name_for_offset
        ),
    )


def test_shared_constants_needs_an_uncertain_value():
    """A value both sides agree is the start of a named entity is not called a
    constant: they are naming the same thing, and the name is the better text
    for it."""
    db = EntityDb()
    with db.batch() as batch:
        for image_id in (ImageId.ORIG, ImageId.RECOMP):
            batch.set(
                image_id, 0x5000, type=EntityType.DATA, name="g_shared", size=0x10
            )

    code = push(0x5000) + b"\xc3"
    orig = parser(db, ImageId.ORIG)
    recomp = parser(db, ImageId.RECOMP)
    orig.parse_asm(code, 0x401000)
    recomp.parse_asm(code, 0x402000)

    assert 0x5000 in orig.tested_values & recomp.tested_values
    assert shared_constants(orig, recomp) == frozenset()


def test_shared_constants_finds_the_uncertain_value():
    """An interior address on one side and a plain number on the other is the
    case we are after."""
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x5000, type=EntityType.DATA, name="g_big", size=0x1000)

    code = push(0x5800) + b"\xc3"
    orig = parser(db, ImageId.ORIG)
    recomp = parser(db, ImageId.RECOMP)
    orig.parse_asm(code, 0x401000)
    recomp.parse_asm(code, 0x402000)

    assert shared_constants(orig, recomp) == frozenset({0x5800})
    # Parsing again with the veto leaves the number alone.
    assert orig.parse_asm(code, 0x401000, constant_veto={0x5800}) == [
        (0x401000, "push 0x5800"),
        (0x401005, "ret "),
    ]


def test_veto_is_cleared_between_parses():
    """The veto applies to the one parse it was given to."""
    db = EntityDb()
    with db.batch() as batch:
        batch.set(ImageId.ORIG, 0x5000, type=EntityType.DATA, name="g_big", size=0x1000)

    code = push(0x5800) + b"\xc3"
    orig = parser(db, ImageId.ORIG)

    assert orig.parse_asm(code, 0x401000, constant_veto={0x5800})[0][1] == "push 0x5800"
    assert orig.parse_asm(code, 0x401000)[0][1] == "push g_big+2048 (OFFSET)"
