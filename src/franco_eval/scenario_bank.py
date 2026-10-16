"""The Dataset B scenario bank: data/stimulus/scenarios_v2.tsv.

A fusion job is (anchor, scenario): every teacher and the fusor for that job
get the same one-line situation (see fusion.SCENARIOS for why). Sixteen
scenarios cannot carry tens of thousands of sentences without the output
collapsing onto sixteen templates, so the bank holds a few hundred.

File contract (other code reads it, so it is fixed):
    scenario_id <TAB> domain <TAB> text <TAB> status     (header exactly this)
  - ids 0-15 are fusion.SCENARIOS verbatim, in order, always "accepted":
    sentence ids fs_<word>_<NN> and cached prompts are keyed on them.
  - ids 16+ start "pending_review"; the native-speaker pass in
    scenarios_v2_review.csv moves them to "accepted" / "rejected" via
    apply_review(). Rejected rows stay in the file as the audit trail and
    are never returned by load_scenarios(). Consumers must key on
    scenario_id, never on list position: rejected ids leave gaps.

Provenance (the 4-column contract has no field for it): ids 0-15 were
written for the fusion pilot; ids 16-330 were drafted by Claude on
2026-09-25 (30 reworded on 2026-09-26 to remove same-situation
paraphrases) and are model-authored. The native speaker accepted the bank
on 2026-09-26 ("the new scenarios seem fine", reworded ones included by
instruction). They are generation prompts for training data, not
benchmark text, and never enter data/benchmark/.

Stdlib only. Does not import fusion.py, so fusion.py may import this one.
"""

import csv
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BANK = ROOT / "data" / "stimulus" / "scenarios_v2.tsv"
DEFAULT_REVIEW = ROOT / "data" / "stimulus" / "scenarios_v2_review.csv"

HEADER = ("scenario_id", "domain", "text", "status")
REVIEW_HEADER = ("scenario_id", "domain", "text", "reviewer")
STATUSES = ("accepted", "pending_review", "rejected")
PINNED_IDS = 16  # ids 0..15 == fusion.SCENARIOS (tests enforce it)

# Same accept words as build_dataset_a.load_overrides: the first word of the
# reviewer cell decides; anything else non-empty is a rejection.
ACCEPT_WORDS = frozenset({"ok", "accept", "accepted", "yes"})

DOMAINS = (
    "family", "friends", "neighbours", "food_cooking", "shopping_prices",
    "transport_traffic", "work", "school_university", "health_doctors",
    "bureaucracy_paperwork", "housing_repairs", "money", "weather",
    "sports_football", "tv_phones", "holidays_occasions", "weddings_funerals",
    "errands", "emotions_complaints", "plans_invitations", "daily_life",
)

# Near-duplicate check: Jaccard over CONTENT tokens. Every scenario shares
# "you're ... a ... the ...", so counting function words would make short
# scenarios look alike and long ones look distinct. At 0.5, a one-word swap
# ("asking a friend for help" / "asking a neighbour for help" = 0.6) is
# caught; tests/test_scenario_bank.py runs it over every pair in the bank.
NEAR_DUP_JACCARD = 0.5

FUNCTION_WORDS = frozenset("""
a an the and or but nor of to in on at for from with by about into onto over
under after before during until since up out off down as is are was were be
been being am it its it's this that these those there here than then so if
when while because not no do does did done doing don't doesn't didn't can
can't could couldn't should would won't will shall may might must have has
had having haven't hasn't isn't aren't wasn't you you're you've you'll you'd
your yours yourself i me my we our they them their they're he him his he's
she her she's who who's what which whose how why where some something
someone somewhere any anything anyone all every each just still again also
very too more less much many few one own same other another back even only
ever never now already yet
""".split())

_TOKEN = re.compile(r"[a-z]+(?:['-][a-z]+)*")


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    domain: str
    text: str
    status: str


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN.findall(text.lower()) if t not in FUNCTION_WORDS)


def jaccard(a: str, b: str) -> float:
    left, right = content_tokens(a), content_tokens(b)
    union = left | right
    if not union:
        return 1.0 if a.lower().split() == b.lower().split() else 0.0
    return len(left & right) / len(union)


def near_duplicates(texts: Sequence[str], threshold: float = NEAR_DUP_JACCARD
                    ) -> list[tuple[int, int, float]]:
    """Every (i, j, score) pair with content-token Jaccard >= threshold,
    highest first."""
    token_sets = [content_tokens(t) for t in texts]
    hits = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            union = token_sets[i] | token_sets[j]
            score = (len(token_sets[i] & token_sets[j]) / len(union) if union
                     else jaccard(texts[i], texts[j]))
            if score >= threshold:
                hits.append((i, j, round(score, 4)))
    return sorted(hits, key=lambda hit: -hit[2])


def _parse_row(fields: list[str], line_no: int) -> Scenario:
    if len(fields) != len(HEADER):
        raise ValueError(f"line {line_no}: expected {len(HEADER)} fields, got {len(fields)}")
    raw_id, domain, text, status = fields
    if not raw_id.isdigit():
        raise ValueError(f"line {line_no}: scenario_id {raw_id!r} is not a non-negative int")
    if status not in STATUSES:
        raise ValueError(f"line {line_no}: unknown status {status!r} (want one of {STATUSES})")
    if not domain or not text.strip():
        raise ValueError(f"line {line_no}: empty domain or text")
    return Scenario(int(raw_id), domain, text, status)


def read_bank(path: str | os.PathLike) -> tuple[Scenario, ...]:
    """Every row, in file order, validated. Raises ValueError on any breach
    of the file contract rather than returning a partial bank."""
    path = pathlib.Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            raise ValueError(f"{path}: header must be {HEADER}, got {header}")
        rows = tuple(_parse_row(fields, n) for n, fields in enumerate(reader, start=2)
                     if fields)
    ids = [row.scenario_id for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate scenario_id {duplicates}")
    by_id = {row.scenario_id: row for row in rows}
    for pinned in range(PINNED_IDS):
        if pinned not in by_id:
            raise ValueError(f"{path}: pinned scenario_id {pinned} missing")
        if by_id[pinned].status != "accepted":
            raise ValueError(f"{path}: pinned scenario_id {pinned} must stay accepted")
    return rows


def load_scenarios(path: str | os.PathLike, accepted_only: bool = True
                   ) -> list[tuple[int, str]]:
    """[(scenario_id, text)] sorted by id. accepted_only=False adds the
    pending_review rows (for pilots before the review); rejected rows are
    never returned."""
    allowed = {"accepted"} if accepted_only else {"accepted", "pending_review"}
    return sorted((row.scenario_id, row.text) for row in read_bank(path)
                  if row.status in allowed)


def _atomic_write(path: pathlib.Path, lines: Iterable[str]) -> None:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False)
    try:
        with handle:
            handle.writelines(lines)
        os.replace(handle.name, path)
    except BaseException:
        pathlib.Path(handle.name).unlink(missing_ok=True)
        raise


def write_bank(path: str | os.PathLike, rows: Sequence[Scenario]) -> None:
    for row in rows:
        if any(ch in value for value in (row.domain, row.text) for ch in "\t\r\n"):
            raise ValueError(f"scenario {row.scenario_id}: tab or newline in a field")
    lines = ["\t".join(HEADER) + "\n"]
    lines += [f"{r.scenario_id}\t{r.domain}\t{r.text}\t{r.status}\n" for r in rows]
    _atomic_write(pathlib.Path(path), lines)


def verdict_status(reviewer: str) -> str:
    words = reviewer.strip().lower().split()
    if not words:
        return "pending_review"
    return "accepted" if words[0].rstrip(":,.;!") in ACCEPT_WORDS else "rejected"


def _read_review(path: pathlib.Path) -> dict[int, tuple[str, str]]:
    """{scenario_id: (text, reviewer)}. utf-8-sig and a ';' delimiter are
    tolerated because a spreadsheet re-save (a German-locale Excel in
    particular) produces both."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        delimiter = ";" if first.count(";") > first.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = {"scenario_id", "text", "reviewer"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: review sheet lacks columns {sorted(missing)}")
        out: dict[int, tuple[str, str]] = {}
        for n, row in enumerate(reader, start=2):
            raw = (row["scenario_id"] or "").strip()
            if not raw.isdigit():
                raise ValueError(f"{path} line {n}: scenario_id {raw!r} is not an int")
            if int(raw) in out:
                raise ValueError(f"{path} line {n}: scenario_id {raw} reviewed twice")
            out[int(raw)] = ((row["text"] or "").strip(), row["reviewer"] or "")
    return out


def apply_review(tsv: str | os.PathLike, review_csv: str | os.PathLike) -> None:
    """Set each reviewed row's status from its reviewer cell (empty ->
    pending_review, accept word -> accepted, anything else -> rejected).

    The review sheet is the source of truth for ids 16+, so re-running is
    idempotent. Ids 0-15 stay accepted whatever the sheet says. An id the
    bank lacks, or a row whose text no longer matches the bank, aborts the
    whole update before anything is written: a verdict on edited text is a
    verdict on a different scenario, and the edit would otherwise be lost.
    """
    tsv, review_csv = pathlib.Path(tsv), pathlib.Path(review_csv)
    bank = read_bank(tsv)
    reviews = _read_review(review_csv)
    by_id = {row.scenario_id: row for row in bank}
    unknown = sorted(set(reviews) - set(by_id))
    if unknown:
        raise ValueError(f"{review_csv}: scenario_id {unknown} not in {tsv}")
    edited = sorted(sid for sid, (text, _v) in reviews.items()
                    if sid >= PINNED_IDS and text != by_id[sid].text)
    if edited:
        raise ValueError(f"{review_csv}: text differs from the bank for scenario_id "
                         f"{edited}; put rewrites in a new row, not over a reviewed one")

    def reviewed(row: Scenario) -> Scenario:
        if row.scenario_id < PINNED_IDS or row.scenario_id not in reviews:
            return row
        return replace(row, status=verdict_status(reviews[row.scenario_id][1]))

    write_bank(tsv, tuple(reviewed(row) for row in bank))


def write_review_sheet(tsv: str | os.PathLike, review_csv: str | os.PathLike) -> int:
    """Create the review CSV (pending rows, empty reviewer). Refuses to
    overwrite: the existing sheet may already hold the reviewer's verdicts.
    Returns the number of rows written."""
    review_csv = pathlib.Path(review_csv)
    if review_csv.exists():
        raise FileExistsError(f"{review_csv} exists; it may hold verdicts")
    pending = [row for row in read_bank(tsv)
               if row.scenario_id >= PINNED_IDS and row.status == "pending_review"]
    with review_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)  # CRLF, as data/training/a_v1/rejected_for_review.csv
        writer.writerow(REVIEW_HEADER)
        writer.writerows((r.scenario_id, r.domain, r.text, "") for r in pending)
    return len(pending)
