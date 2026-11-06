"""Tests for the normalization of the MSVC stack security cookie (/GS)."""

from unittest.mock import Mock
import pytest
from reccmp.cvdump.types import CvdumpTypesParser
from reccmp.compare.asm.parse import AsmExcerpt
from reccmp.compare.asm.security_cookie import (
    CHECK_PLACEHOLDER,
    COOKIE_PLACEHOLDER,
    ignore_security_cookie,
)
from reccmp.compare.db import EntityDb
from reccmp.compare.event import ReccmpReportProtocol
from reccmp.compare.functions import CallComparisonOptions, FunctionComparator
from reccmp.compare.lines import LinesDb
from reccmp.types import EntityType, ImageId
from .raw_image import RawImage

# How the orig binary names the cookie in a project that resolves original
# globals through a template. 12141088 == 0xB94220.
ORIG_COOKIE_NAME = "StructResolver::Instance<unsigned int,12141088>::instance"
RECOMP_COOKIE_NAME = "__security_cookie"


@pytest.fixture(name="db")
def fixture_db() -> EntityDb:
    return EntityDb()


@pytest.fixture(name="lines_db")
def fixture_lines_db() -> LinesDb:
    return LinesDb()


@pytest.fixture(name="report")
def fixture_report_mock() -> ReccmpReportProtocol:
    return Mock(spec=ReccmpReportProtocol)


def _prologue(cookie_name: str) -> AsmExcerpt:
    return [
        (0x500, f"mov eax, dword ptr [{cookie_name}]"),
        (0x505, "xor eax, esp"),
        (0x507, "mov dword ptr [esp + 0x1c], eax"),
    ]


def test_ignore_security_cookie_prologue():
    """The cookie read is a load that feeds `xor <reg>, esp`."""
    assert ignore_security_cookie(_prologue(ORIG_COOKIE_NAME), {ORIG_COOKIE_NAME}) == [
        (0x500, f"mov eax, dword ptr [{COOKIE_PLACEHOLDER}]"),
        (0x505, "xor eax, esp"),
        (0x507, "mov dword ptr [esp + 0x1c], eax"),
    ]


def test_ignore_security_cookie_both_sides_agree():
    """The point of the placeholder: the two binaries name the cookie
    differently, but the normalized instruction is the same."""
    orig = ignore_security_cookie(_prologue(ORIG_COOKIE_NAME), {ORIG_COOKIE_NAME})
    recomp = ignore_security_cookie(_prologue(RECOMP_COOKIE_NAME), {RECOMP_COOKIE_NAME})
    assert orig == recomp


def test_ignore_security_cookie_epilogue():
    """The call that follows `xor <reg>, esp` is the cookie check."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "call __security_check_cookie"),
        (0x60B, "ret"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"}) == [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, f"call {CHECK_PLACEHOLDER}"),
        (0x60B, "ret"),
    ]


def test_ignore_security_cookie_epilogue_with_scheduled_store():
    """The compiler puts unrelated work between the XOR and the check call.
    Observed in OpenSHC's MenuItemRenderFunction_SingleplayerMapChoice_MapTable,
    where requiring adjacency left the call in the diff at 99.4%."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x2c]"),
        (0x604, "pop esi"),
        (0x605, "xor ecx, esp"),
        (0x607, "mov dword ptr [SomeGlobal + 4], 1"),
        (0x60E, "call __security_check_cookie"),
        (0x613, "add esp, 0x2c"),
        (0x616, "ret"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"})[4] == (
        0x60E,
        f"call {CHECK_PLACEHOLDER}",
    )


def test_ignore_security_cookie_epilogue_register_reassigned():
    """A write to the register in between means the call does not receive the
    cookie, so this is not the idiom."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "mov ecx, 1"),
        (0x60B, "call SomeFunction"),
    ]
    assert ignore_security_cookie(excerpt, {"SomeFunction"}) == excerpt


def test_ignore_security_cookie_epilogue_partial_register_reassigned():
    """`cl` is part of `ecx`, so writing it also breaks the chain."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "mov cl, 1"),
        (0x60B, "call SomeFunction"),
    ]
    assert ignore_security_cookie(excerpt, {"SomeFunction"}) == excerpt


def test_ignore_security_cookie_epilogue_memory_write_is_not_a_reassignment():
    """Writing through the register only reads it."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "mov dword ptr [ecx], 1"),
        (0x60B, "call __security_check_cookie"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"})[3] == (
        0x60B,
        f"call {CHECK_PLACEHOLDER}",
    )


def test_ignore_security_cookie_epilogue_branch_stops_the_search():
    """A jump between the two means they are not on one path."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "je 0x620"),
        (0x60B, "call __security_check_cookie"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"}) == excerpt


def test_ignore_security_cookie_epilogue_earlier_call_stops_the_search():
    """Only the first call after the XOR can be the check."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
        (0x606, "call SomeOtherFunction"),
        (0x60B, "call __security_check_cookie"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"}) == excerpt


def test_ignore_security_cookie_epilogue_too_far_away():
    """Past the scheduling window the shape is a coincidence, not the idiom."""
    excerpt: AsmExcerpt = [
        (0x600, "xor ecx, esp"),
        (0x602, "nop"),
        (0x603, "nop"),
        (0x604, "nop"),
        (0x605, "nop"),
        (0x606, "call __security_check_cookie"),
    ]
    assert ignore_security_cookie(excerpt, {"__security_check_cookie"}) == excerpt


def test_ignore_security_cookie_prologue_with_scheduled_instruction():
    """The load does not have to be adjacent to the XOR either."""
    excerpt: AsmExcerpt = [
        (0x500, f"mov eax, dword ptr [{ORIG_COOKIE_NAME}]"),
        (0x506, "push ebx"),
        (0x507, "xor eax, esp"),
        (0x509, "mov dword ptr [esp + 0x1c], eax"),
    ]
    assert ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME})[0] == (
        0x500,
        f"mov eax, dword ptr [{COOKIE_PLACEHOLDER}]",
    )


def test_ignore_security_cookie_prologue_nearest_write_wins():
    """If something else assigned the register after the load, the value mixed
    with the stack pointer is not the global."""
    excerpt: AsmExcerpt = [
        (0x500, f"mov eax, dword ptr [{ORIG_COOKIE_NAME}]"),
        (0x506, "mov eax, 7"),
        (0x50B, "xor eax, esp"),
    ]
    assert ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME}) == excerpt


def test_ignore_security_cookie_prologue_does_not_claim_the_next_call():
    """The prologue XOR must not be searched forward: the next call belongs to
    the function, not to the stack guard. Relaxing the adjacency requirement
    without distinguishing the two halves erases a real call target."""
    excerpt: AsmExcerpt = [
        (0x500, f"mov eax, dword ptr [{ORIG_COOKIE_NAME}]"),
        (0x505, "xor eax, esp"),
        (0x507, "mov dword ptr [esp + 0x1c], eax"),
        (0x50B, "push ebx"),
        (0x50C, "call SomeRealFunction"),
    ]
    result = ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME, "SomeRealFunction"})
    assert result[4] == (0x50C, "call SomeRealFunction")
    # ... while the cookie read itself is still normalized.
    assert result[0] == (0x500, f"mov eax, dword ptr [{COOKIE_PLACEHOLDER}]")


def test_ignore_security_cookie_stack_read_is_not_the_cookie():
    """The epilogue reads the saved cookie from the stack. That operand is not
    a name, so it must be left alone."""
    excerpt: AsmExcerpt = [
        (0x600, "mov ecx, dword ptr [esp + 0x1c]"),
        (0x604, "xor ecx, esp"),
    ]
    assert ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME}) == excerpt


def test_ignore_security_cookie_other_global_untouched():
    """A global that is not mixed with the stack pointer is a real variable."""
    excerpt: AsmExcerpt = [
        (0x500, f"mov eax, dword ptr [{ORIG_COOKIE_NAME}]"),
        (0x505, "xor eax, ebx"),
        (0x507, "call something"),
    ]
    assert ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME, "something"}) == excerpt


def test_ignore_security_cookie_register_must_match():
    """`xor` on a different register than the one just loaded is not the
    cookie idiom."""
    excerpt: AsmExcerpt = [
        (0x500, f"mov eax, dword ptr [{ORIG_COOKIE_NAME}]"),
        (0x505, "xor edx, esp"),
    ]
    assert ignore_security_cookie(excerpt, {ORIG_COOKIE_NAME}) == excerpt


def test_ignore_security_cookie_unknown_name():
    """Nothing to do if we did not substitute a name there."""
    excerpt = _prologue("eax")
    assert ignore_security_cookie(excerpt, set()) == excerpt


ORIG_BASE = 0x401000
RECOMP_BASE = 0x402000
# mov eax, dword ptr [0x500000] ; xor eax, esp ; ret
CODE = b"\xa1\x00\x00\x50\x00\x33\xc4\xc3"
COOKIE_ADDR = 0x500000


def _compare_cookie_function(
    db: EntityDb,
    lines_db: LinesDb,
    report: ReccmpReportProtocol,
    call_options: CallComparisonOptions,
):
    """Both binaries read a cookie global from the same address, but the
    entity found there has a different name in each one."""
    with db.batch() as batch:
        for image_id, base in (
            (ImageId.ORIG, ORIG_BASE),
            (ImageId.RECOMP, RECOMP_BASE),
        ):
            batch.set(
                image_id,
                base,
                type=EntityType.FUNCTION,
                name="test",
                size=len(CODE),
            )

        batch.set(
            ImageId.ORIG,
            COOKIE_ADDR,
            type=EntityType.DATA,
            name=ORIG_COOKIE_NAME,
            size=4,
        )
        batch.set(
            ImageId.RECOMP,
            COOKIE_ADDR,
            type=EntityType.DATA,
            name=RECOMP_COOKIE_NAME,
            size=4,
        )
        batch.match(ORIG_BASE, RECOMP_BASE)

    orig_bin = RawImage.from_memory(CODE, base_addr=ORIG_BASE)
    recomp_bin = RawImage.from_memory(CODE, base_addr=RECOMP_BASE)

    comp = FunctionComparator(
        db,
        lines_db,
        orig_bin,
        recomp_bin,
        report,
        CvdumpTypesParser(),
        call_options=call_options,
    )
    (entity,) = list(db.get_functions())
    return comp.compare_function(entity)


def test_compare_function_security_cookie_mismatch(
    db: EntityDb, lines_db: LinesDb, report: ReccmpReportProtocol
):
    """Without the option, the differing cookie names are a mismatch."""
    result = _compare_cookie_function(db, lines_db, report, CallComparisonOptions())
    assert result.match_ratio != 1.0


def test_compare_function_security_cookie_ignored(
    db: EntityDb, lines_db: LinesDb, report: ReccmpReportProtocol
):
    """With --ignore-security-cookie the function is a 100% match."""
    result = _compare_cookie_function(
        db, lines_db, report, CallComparisonOptions(ignore_security_cookie=True)
    )
    assert result.match_ratio == 1.0
