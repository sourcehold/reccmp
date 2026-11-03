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
