"""Split a filing's plain text into its Items (Item 1A, Item 7, ...) and paragraphs.

Input is the text from finsight.datastore.api.html_to_text. No network, no LLM:
pure functions, so they are easy to unit-test.

Each section gets a `topic` that means the same thing in every form type, so the
router can ask for "risk_factors" without knowing that it is Item 1A in a 10-K
but Part II Item 1A in a 10-Q.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SPLITTER_VERSION = 3   # bump when splitting rules change, so filings get re-indexed

TENK_TOPICS = {
    "1": "business", "1A": "risk_factors", "1B": "unresolved_staff_comments",
    "1C": "cybersecurity", "2": "properties", "3": "legal", "4": "mine_safety",
    "5": "market_for_equity", "6": "reserved", "7": "mdna", "7A": "market_risk",
    "8": "financial_statements", "9": "accountant_changes", "9A": "controls",
    "9B": "other_information", "9C": "foreign_inspections", "10": "directors",
    "11": "executive_compensation", "12": "ownership", "13": "relationships",
    "14": "accountant_fees", "15": "exhibits", "16": "summary",
}
TENQ_TOPICS = {
    ("I", "1"): "financial_statements", ("I", "2"): "mdna", ("I", "3"): "market_risk",
    ("I", "4"): "controls", ("II", "1"): "legal", ("II", "1A"): "risk_factors",
    ("II", "2"): "equity_sales", ("II", "3"): "defaults", ("II", "4"): "mine_safety",
    ("II", "5"): "other_information", ("II", "6"): "exhibits",
}
EIGHTK_TOPICS = {
    "1.01": "material_agreement", "2.02": "results", "2.05": "restructuring",
    "5.02": "officer_changes", "5.07": "shareholder_vote", "7.01": "reg_fd",
    "8.01": "other_events", "9.01": "exhibits",
}

PART_RE = re.compile(r"^\s*PART\s+(IV|I{1,3})\b", re.IGNORECASE)
ITEM_RE = re.compile(
    r"^\s*ITEM\s+(\d{1,2}(?:\.\d{2})?[A-C]?)\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
TOC_TAIL_RE = re.compile(r"(\s\d{1,3}|\.{4,}\s*\d*)\s*$")      # "... Risk Factors 21"
FOOTER_RE = re.compile(r"^(.{0,80}\|\s*\d{1,3}|\d{1,3}|page \d+( of \d+)?|table of contents|index to (consolidated )?financial statements)$",
                       re.IGNORECASE)

MAX_WORDS = 300        # longer paragraphs are cut at sentence boundaries
TARGET_WORDS = 220
MIN_WORDS = 30         # shorter paragraphs are merged (see split_paragraphs)
MERGE_INTO_NEXT = 60   # up to this many words of short lines are joined to the next paragraph


@dataclass
class Passage:
    seq: int
    text: str
    heading: str = ""      # nearest short heading above it, e.g. "Gross Margin"
    kind: str = "text"     # "text" or "table" (tables are not searched: numbers come from XBRL)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Section:
    part: str              # "I"/"II" for 10-Q, "" otherwise
    item: str              # "1A", "7", "2.02", "EX-99.1", "ALL"
    topic: str
    title: str
    text: str
    passages: list[Passage] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(p.n_words for p in self.passages if p.kind == "text")


def _topic(form: str, part: str, item: str) -> str | None:
    if form.startswith("10-K"):
        return TENK_TOPICS.get(item)
    if form.startswith("10-Q"):
        return TENQ_TOPICS.get((part, item))
    if form.startswith("8-K"):
        return EIGHTK_TOPICS.get(item, f"8k_item_{item}")
    return None


def find_items(text: str, form: str) -> list[tuple[int, int, str, str, str]]:
    """Heading positions: (heading_start, body_start, part, item, title), in document order.

    Table-of-contents lines (ending in a page number) are ignored. If an Item heading
    still appears more than once, the LAST one wins: the body comes after the TOC.
    """
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1

    part, found = "", []
    for i, (start, line) in enumerate(lines):
        m = PART_RE.match(line)
        if m and len(line) < 80:
            part = m.group(1).upper()
            continue
        m = ITEM_RE.match(line)
        if not m or len(line) > 160:
            continue
        item, title = m.group(1).upper(), m.group(2).strip()
        if TOC_TAIL_RE.search(line):
            continue
        body_start = start + len(line) + 1
        if not title:                       # "Item 1A." on its own line, title on the next
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j][1].strip()
                if nxt:
                    if len(nxt) < 100 and not TOC_TAIL_RE.search(nxt):
                        title, body_start = nxt, lines[j][0] + len(lines[j][1]) + 1
                    break
        p = part if form.startswith("10-Q") else ""
        if _topic(form, p, item) is None:
            continue
        found.append((start, body_start, p, item, title))

    if form.startswith("10-Q") and any(f[2] for f in found):
        found = [f for f in found if f[2]]  # TOC entries seen before any "PART" line
    last: dict[tuple[str, str], tuple] = {}
    for f in found:
        last[(f[2], f[3])] = f
    return sorted(last.values())


def _clean_title(title: str, topic: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .:-–—")
    return title[:120] or topic.replace("_", " ").title()


def _is_table(para: str) -> bool:
    """Mostly numbers. html_to_text glues table cells together ("Net sales$94,036$85,777"),
    so count number runs against real words rather than splitting on spaces."""
    nums = len(re.findall(r"\d[\d,.]*", para))
    words = len(re.findall(r"[A-Za-z]{2,}", para))
    return nums >= 2 and nums >= 0.5 * words


def _is_heading(para: str) -> bool:
    words = para.split()
    return 0 < len(words) <= 12 and not para.rstrip().endswith((".", ":", ";", ","))


def _split_long(text: str) -> list[str]:
    if len(text.split()) <= MAX_WORDS:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(“\"])", text)
    out, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        if out and sum(len(x.split()) for x in cur) < 60:
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    return out


def split_paragraphs(text: str) -> list[Passage]:
    """Body text -> passages. Short headings are attached to the paragraph below them.

    Very short paragraphs (< MIN_WORDS), such as a risk factor's one-line title or the
    bullet list that summarises a 10-K's risk factors, are not kept on their own: a few
    words score high in similarity search but carry little evidence. A short title is
    joined to the paragraph after it; a run of bullets becomes one passage.
    """
    out: list[Passage] = []
    heading = ""
    buf: list[str] = []                       # short paragraphs waiting to be merged

    def emit(t: str, kind: str = "text"):
        out.append(Passage(len(out), t, heading, kind))

    def flush():
        if not buf:
            return
        chunk: list[str] = []
        for b in buf:                        # a long bullet run is cut into ~TARGET_WORDS pieces
            if chunk and sum(len(c.split()) for c in chunk) + len(b.split()) > TARGET_WORDS:
                emit(" ".join(chunk)); chunk = []
            chunk.append(b)
        emit(" ".join(chunk))
        buf.clear()

    for raw in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", raw).strip()
        if not para or FOOTER_RE.match(para):
            continue
        if _is_table(para):
            flush()
            emit(para, "table")
            continue
        if _is_heading(para):
            flush()
            heading = para
            continue
        if len(para.split()) < MIN_WORDS:
            buf.append(para)
            if sum(len(b.split()) for b in buf) >= TARGET_WORDS:
                flush()
            continue
        pieces = _split_long(para)
        if buf and sum(len(b.split()) for b in buf) <= MERGE_INTO_NEXT:
            pieces[0] = " ".join(buf + [pieces[0]])       # e.g. risk-factor title + its explanation
            buf.clear()
        else:
            flush()
        for piece in pieces:
            emit(piece)
    flush()
    return out


def split_filing(text: str, form: str) -> list[Section]:
    """Plain text of one filing -> its sections with paragraphs.

    If no Item headings are found (unusual layout), the whole document becomes one
    section with topic "full_text", so search still works.
    """
    heads = find_items(text, form)
    if not heads:
        return [Section("", "ALL", "full_text", "Full text", text, split_paragraphs(text))]
    sections = []
    for i, (_start, body, part, item, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        topic = _topic(form, part, item)
        body_text = text[body:end]
        sections.append(Section(part, item, topic, _clean_title(title, topic), body_text,
                                split_paragraphs(body_text)))
    return sections


def split_exhibit(text: str, name: str = "EX-99.1") -> Section:
    """An 8-K press release (Exhibit 99.1) has no Items: one section, topic "press_release"."""
    return Section("", name, "press_release", "Press release (Exhibit 99.1)", text,
                   split_paragraphs(text))
