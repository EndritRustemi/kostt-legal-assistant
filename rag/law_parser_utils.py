"""
law_parser_utils.py -- Kosovo Legal Document Regex Utilities

Single canonical source for all structural-marker patterns used in Kosovo
legal texts. Provides detection, classification, normalization, and cleaning
as composable, importable utilities.

Languages covered
-----------------
  Albanian : Neni, Artikulli, Paragrafi, Pika
  English  : Article, Paragraph
  Serbian  : Clan  (minority-community documents and bilateral agreements)

Public surface
--------------
  Compiled patterns  ARTICLE_RE, PARA_PAREN_RE, PARA_DOT_RE,
                     PARA_BRACE_RE, PARA_WORD_RE, INLINE_REF_RE

  Result types       ArticleMatch, ParagraphMatch, ParagraphBlock,
                     ArticleBlock

  Normalization      normalize_article(kw, num, title) -> str
                     normalize_paragraph(num)           -> str

  Detection          find_articles(text)                -> list[ArticleMatch]
                     find_paragraphs(text)              -> list[ParagraphMatch]
                     find_inline_references(text)       -> list[dict]

  Classification     is_article_header(line)            -> bool
                     is_paragraph_start(line)           -> bool
                     is_noise_line(line)                -> bool

  Cleaning           remove_page_numbers(text)          -> tuple[str, int]
                     remove_noise_lines(text, ...)      -> tuple[str, int]
                     remove_artifacts(text)             -> tuple[str, int]
                     clean(text)                        -> str

  High-level         extract_structure(text)            -> list[ArticleBlock]
                     iter_chunks(text)                  -> Iterator[...]
                     count_articles(text)               -> int
                     count_paragraphs(text)             -> int
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator


# -- Keyword tables -----------------------------------------------------------

# Maps every written variant to its canonical display form.
KW_CANON: dict[str, str] = {
    # Albanian
    "NENI":       "Neni",
    "Neni":       "Neni",
    "neni":       "Neni",
    "ARTIKULLI":  "Artikulli",
    "Artikulli":  "Artikulli",
    "artikulli":  "Artikulli",
    # English
    "ARTICLE":    "Article",
    "Article":    "Article",
    "article":    "Article",
    "Art.":       "Article",
    "Art":        "Article",
    "ART.":       "Article",
    # Serbian / Bosnian
    "CLAN":       "Clan",
    "Clan":       "Clan",
    "clan":       "Clan",
    # Explicit forms used in Energy Community documents
    "MEMBER":     "Article",
}

# Canonical keywords for Albanian paragraph words
PARA_KW_CANON: dict[str, str] = {
    "PARAGRAFI": "Paragraph",
    "Paragrafi": "Paragraph",
    "paragrafi": "Paragraph",
    "PARAGRAPH": "Paragraph",
    "Paragraph": "Paragraph",
    "paragraph": "Paragraph",
    "PIKA":      "Pika",
    "Pika":      "Pika",
    "pika":      "Pika",
}

ARTICLE_KEYWORDS: frozenset[str] = frozenset(KW_CANON.values())
PARAGRAPH_KEYWORDS: frozenset[str] = frozenset(PARA_KW_CANON.values())


# -- Compiled regex patterns --------------------------------------------------
#
# Naming convention:
#   ARTICLE_RE      -- article header anchored to start AND end of line
#   PARA_*_RE       -- paragraph-start marker anchored to start of line
#   _*_RE           -- private noise / artifact patterns

# Article header.
# Anchored to BOL (^) and EOL ($) so inline references like
# "sipas nenit 5 te ketij ligji" are NOT matched (they appear mid-line).
# Groups: indent, kw, num, sep (optional), title (optional)
ARTICLE_RE: re.Pattern[str] = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kw>ARTICLE|Article|article"
    r"|NENI|Neni|neni"
    r"|ARTIKULLI|Artikulli|artikulli"
    r"|CLAN|Clan|clan"
    r"|Art\.?|ART\.?)"
    r"\s+"
    r"(?P<num>\d+[a-zA-Z]?(?:/[a-zA-Z])?)"   # 5 | 5a | 5/a
    r"(?:"
    r"[ \t]*(?P<sep>[.:\-–—]+)[ \t]*"  # separator: . : - -- - --
    r"(?P<title>[^\n]*)"                          # title to end of line
    r"|[ \t]*"                                    # or nothing
    r")$",
    re.MULTILINE,
)

# Paragraph: "(1)" style
PARA_PAREN_RE: re.Pattern[str] = re.compile(
    r"(?m)^(?P<indent>[ \t]*)\((?P<num>\d+)\)(?=[ \t])",
)

# Paragraph: "1." style
PARA_DOT_RE: re.Pattern[str] = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?P<num>\d+)\.(?=[ \t])",
)

# Paragraph: "1)" style
PARA_BRACE_RE: re.Pattern[str] = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?P<num>\d+)\)(?=[ \t])",
)

# Paragraph: "Paragrafi 1" / "Paragraph 1" / "Pika 1" style
PARA_WORD_RE: re.Pattern[str] = re.compile(
    r"(?m)^(?P<indent>[ \t]*)"
    r"(?P<kw>PARAGRAFI|Paragrafi|paragrafi"
    r"|PARAGRAPH|Paragraph|paragraph"
    r"|PIKA|Pika|pika)"
    r"[ \t]+(?P<num>\d+)"
    r"(?=[ \t\n]|$)",
)

# Convenience: matches ANY paragraph-start marker at BOL (for classification)
PARA_ANY_RE: re.Pattern[str] = re.compile(
    r"(?m)^[ \t]*"
    r"(?:"
    r"\(\d+\)"                                        # (1)
    r"|\d+\."                                         # 1.
    r"|\d+\)"                                         # 1)
    r"|(?:PARAGRAFI|Paragrafi|PARAGRAPH|Paragraph|PIKA|Pika)[ \t]+\d+"
    r")"
    r"(?=[ \t])",
)

# Inline cross-reference to another article (NOT a header, appears in body text)
# e.g. "sipas nenit 5", "pursuant to Article 12", "per nenin 3"
INLINE_REF_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r"sipas|per|shih|see|of|under|pursuant\s+to"
    r"|ne\s+kuptim\s+te"
    r")\s+"
    r"(?:nenit|nenin|artikullit|article|artikullin|Art\.?|clana?|member)\s+"
    r"(?P<num>\d+[a-zA-Z]?)",
    re.IGNORECASE,
)


# -- Private noise patterns ---------------------------------------------------

# Standalone page number: optional word "Faqe/Page" + 1-4 digits + optional "of N"
_PAGE_NUM_RE = re.compile(
    r"^\s*"
    r"(?:(?:Faqe|Page|Strana|faqe|page)\s*)?"
    r"[-–—]?\s*\d{1,4}\s*[-–—]?"
    r"(?:\s*(?:of|nga|von|od)\s*\d+)?"
    r"\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Lines containing only punctuation / symbols (noise from PDF column rules etc.)
_PUNCT_ONLY_RE = re.compile(
    r"^\s*[^a-zA-ZÀ-ɏ\d\n]{4,}\s*$",
    re.MULTILINE,
)

# Kosovo Official Gazette and equivalent official publication headers
_GAZETTE_RE = re.compile(
    r"Gazetat?\s+Zyrtare|Official\s+Gazette|Buletin\s+Zyrtar"
    r"|Fletorja\s+Zyrtare|Rregullorja\s+Zyrtare",
    re.IGNORECASE,
)

# Lone integer line (detached list/page number embedded in extracted body text)
_LONE_INT_RE = re.compile(
    r"^\s*\d{1,3}\s*$",
    re.MULTILINE,
)

# Roman numeral only line (table-of-contents or annex section markers)
_ROMAN_ONLY_RE = re.compile(
    r"^\s*"
    r"(?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{1,3})"
    r"\s*$",
    re.MULTILINE,
)

# Table-of-contents entry: "Neni 5 .............. 12"
_TOC_ENTRY_RE = re.compile(
    r"^[ \t]*(?:Neni|Article|Artikulli|Clan)\s+\d+\S*\s+[.]{4,}\s*\d+\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Trailing separator used to strip noise from title groups
_TITLE_TRIM_RE = re.compile(r"^[.\-–—:\s]+|[.\-–—:\s]+$")


# -- Result data types --------------------------------------------------------

@dataclass
class ArticleMatch:
    """
    A single article header detected in source text.

    Attributes
    ----------
    keyword : canonical keyword ("Neni", "Article", "Clan", "Artikulli")
    number  : article number as string ("5" or "5a" or "5/a")
    title   : same-line title text, stripped of separators (may be "")
    raw     : exact text of the matched article header line
    start   : character offset in the source text (inclusive)
    end     : character offset in the source text (exclusive)
    """
    keyword: str
    number:  str
    title:   str
    raw:     str
    start:   int
    end:     int

    @property
    def label(self) -> str:
        """Canonical label: 'Neni 5 -- Qellimi' or 'Article 3'."""
        return normalize_article(self.keyword, self.number, self.title)

    def __str__(self) -> str:
        return self.label


@dataclass
class ParagraphMatch:
    """
    A single paragraph-start marker detected in source text.

    Attributes
    ----------
    number : paragraph number (1, 2, 3, ...)
    style  : marker style: "paren" | "dot" | "brace" | "word"
    raw    : exact text of the marker, e.g. "(1)", "1.", "Paragrafi 2"
    start  : character offset in the source text (inclusive)
    end    : character offset in the source text (exclusive, includes trailing space)
    """
    number: int
    style:  str   # "paren" | "dot" | "brace" | "word"
    raw:    str
    start:  int
    end:    int

    @property
    def label(self) -> str:
        """Canonical label: 'Paragraph 1'."""
        return normalize_paragraph(self.number)

    def __str__(self) -> str:
        return self.label


@dataclass
class ParagraphBlock:
    """
    One paragraph's content within an article.

    Attributes
    ----------
    match : the ParagraphMatch for the marker, or None if the paragraph
            was inferred (whole article body, or blank-line split)
    text  : clean paragraph text (marker removed)
    """
    match: ParagraphMatch | None
    text:  str

    @property
    def number(self) -> int:
        """Paragraph number (1 if implicit)."""
        return self.match.number if self.match else 1

    @property
    def label(self) -> str:
        return normalize_paragraph(self.number)


@dataclass
class ArticleBlock:
    """
    One article and its paragraph breakdown.

    Attributes
    ----------
    match      : the ArticleMatch for the article header
    body       : full raw body text of the article (paragraphs concatenated)
    paragraphs : list of ParagraphBlock, one per paragraph
    """
    match:      ArticleMatch
    body:       str
    paragraphs: list[ParagraphBlock] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.match.label

    def __str__(self) -> str:
        return (
            f"{self.label}  [{len(self.paragraphs)} paragraph(s), "
            f"{len(self.body)} chars]"
        )


# -- Normalization ------------------------------------------------------------

def normalize_article(keyword: str, number: str, title: str = "") -> str:
    """
    Return a canonical article label.

    Parameters
    ----------
    keyword : any supported keyword variant ("NENI", "Art.", "ARTICLE", ...)
    number  : article number ("5", "5a", "5/a")
    title   : optional title text (stripped of leading/trailing separators)

    Returns
    -------
    str
        "Neni 5 -- Qellimi i ligjit"  (with title)
        "Article 3"                    (without title)

    Examples
    --------
    >>> normalize_article("NENI", "5", "Qellimi")
    'Neni 5 -- Qellimi'
    >>> normalize_article("Art.", "3", "")
    'Article 3'
    >>> normalize_article("ARTICLE", "12", "  -- Penalties: ")
    'Article 12 -- Penalties'
    """
    kw    = KW_CANON.get(keyword, keyword)
    title = _TITLE_TRIM_RE.sub("", title.strip())
    if title:
        return f"{kw} {number} -- {title}"
    return f"{kw} {number}"


def normalize_paragraph(number: int | str) -> str:
    """
    Return a canonical paragraph label.

    Parameters
    ----------
    number : paragraph number (int or digit string)

    Returns
    -------
    str
        "Paragraph 1", "Paragraph 2", ...

    Examples
    --------
    >>> normalize_paragraph(1)
    'Paragraph 1'
    >>> normalize_paragraph("3")
    'Paragraph 3'
    """
    return f"Paragraph {number}"


# -- Detection ----------------------------------------------------------------

def find_articles(
    text:          str,
    max_title_len: int = 120,
) -> list[ArticleMatch]:
    """
    Find all article headers in *text*, in document order.

    Parameters
    ----------
    text          : source document text
    max_title_len : if the matched title is longer than this, the match is
                    likely inline prose rather than a real article header
                    and is discarded (default 120)

    Returns
    -------
    list[ArticleMatch]
        One entry per detected article header, sorted by position.
    """
    results: list[ArticleMatch] = []
    for m in ARTICLE_RE.finditer(text):
        kw    = KW_CANON.get(m.group("kw"), m.group("kw"))
        num   = m.group("num")
        title = _TITLE_TRIM_RE.sub("", (m.group("title") or "").strip())

        if len(title) > max_title_len:
            continue   # inline prose, not a real header

        results.append(ArticleMatch(
            keyword=kw,
            number=num,
            title=title,
            raw=m.group(0),
            start=m.start(),
            end=m.end(),
        ))
    return results


def find_paragraphs(text: str) -> list[ParagraphMatch]:
    """
    Find all paragraph-start markers in *text*, in document order.

    All four marker styles are detected:
      - "(1)" paren style
      - "1."  dot style
      - "1)"  brace style
      - "Paragrafi 1" / "Paragraph 1" / "Pika 1" word style

    Each position is reported at most once (overlapping matches are skipped).

    Returns
    -------
    list[ParagraphMatch]
        Sorted by position in the source text.
    """
    found:  list[ParagraphMatch] = []
    seen:   set[int]             = set()   # start offsets already claimed

    def _collect(pattern: re.Pattern[str], style: str) -> None:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos in seen:
                continue
            seen.add(pos)
            num_str = m.group("num")
            found.append(ParagraphMatch(
                number=int(num_str),
                style=style,
                raw=m.group(0).strip(),
                start=pos,
                end=m.end(),
            ))

    _collect(PARA_PAREN_RE, "paren")
    _collect(PARA_DOT_RE,   "dot")
    _collect(PARA_BRACE_RE, "brace")
    _collect(PARA_WORD_RE,  "word")

    found.sort(key=lambda x: x.start)
    return found


def find_inline_references(text: str) -> list[dict[str, Any]]:
    """
    Find inline cross-references to other articles within body text.

    These are phrases like "sipas nenit 5", "pursuant to Article 12",
    "per nenin 3" — as opposed to article header lines.

    Returns
    -------
    list[dict]
        Each dict has keys: number (str), raw (str), start (int), end (int).
    """
    return [
        {
            "number": m.group("num"),
            "raw":    m.group(0),
            "start":  m.start(),
            "end":    m.end(),
        }
        for m in INLINE_REF_RE.finditer(text)
    ]


# -- Classification -----------------------------------------------------------

def is_article_header(line: str) -> bool:
    """
    Return True if *line* (a single line of text) is an article header.

    Requires the keyword to start the line (optional leading whitespace)
    and the rest of the line to form a valid article header pattern.

    Examples
    --------
    >>> is_article_header("Neni 5 -- Qellimi")
    True
    >>> is_article_header("sipas nenit 5 te ketij ligji")
    False
    >>> is_article_header("Article 12")
    True
    """
    return bool(ARTICLE_RE.match(line.strip()))


def is_paragraph_start(line: str) -> bool:
    """
    Return True if *line* begins with a recognized paragraph marker.

    Examples
    --------
    >>> is_paragraph_start("(1) KOSTT is responsible...")
    True
    >>> is_paragraph_start("1. The operator shall...")
    True
    >>> is_paragraph_start("Paragrafi 2 ...")
    True
    >>> is_paragraph_start("Some normal sentence.")
    False
    """
    return bool(PARA_ANY_RE.match(line))


def is_noise_line(line: str) -> bool:
    """
    Return True if *line* is structural noise that should be removed:

    - Standalone page numbers ("12", "- 12 -", "Faqe 12")
    - Lines with only punctuation / rule characters
    - Short Official Gazette / publication headers
    - Lone integers (detached list/page numbers embedded in body text)
    - Roman-numeral only lines (ToC markers)
    - Table-of-contents entries ("Neni 5 ........ 12")

    Blank lines are NOT classified as noise -- they carry structural meaning.

    Examples
    --------
    >>> is_noise_line("12")
    True
    >>> is_noise_line("- 5 -")
    True
    >>> is_noise_line("Gazeta Zyrtare e Republikes se Kosoves")
    True
    >>> is_noise_line("Neni 3 ............. 7")
    True
    >>> is_noise_line("(1) KOSTT is responsible...")
    False
    >>> is_noise_line("")
    False
    """
    stripped = line.strip()
    if not stripped:
        return False

    if _PAGE_NUM_RE.match(line):
        return True
    if _PUNCT_ONLY_RE.match(line):
        return True
    if _LONE_INT_RE.match(line):
        return True
    if _ROMAN_ONLY_RE.match(line):
        return True
    if _TOC_ENTRY_RE.match(line):
        return True
    # Gazette header: must be short enough to be a standalone header line
    if _GAZETTE_RE.search(stripped) and len(stripped) < 80:
        return True

    return False


# -- Cleaning -----------------------------------------------------------------

def remove_page_numbers(text: str) -> tuple[str, int]:
    """
    Remove all standalone page-number lines from *text*.

    Returns
    -------
    (cleaned_text, lines_removed)
    """
    matches = [m for m in _PAGE_NUM_RE.finditer(text) if m.group(0).strip()]
    count   = len(matches)
    cleaned = _PAGE_NUM_RE.sub("", text)
    return cleaned, count


def remove_noise_lines(
    text:        str,
    min_repeats: int = 3,
) -> tuple[str, int]:
    """
    Remove all lines classified as noise and repeated header/footer boilerplate.

    A line is removed if:
      - ``is_noise_line(line)`` returns True, OR
      - it appears verbatim >= *min_repeats* times in the document
        (typical of running page headers and footers)

    Only lines longer than 8 characters are counted for the repeat check
    to avoid removing legitimate short lines (e.g. section labels).

    Returns
    -------
    (cleaned_text, lines_removed)
    """
    lines    = text.splitlines()
    counts   = Counter(ln.strip() for ln in lines if len(ln.strip()) > 8)
    repeated = {ln for ln, cnt in counts.items() if cnt >= min_repeats}

    cleaned:  list[str] = []
    removed:  int       = 0
    for ln in lines:
        if is_noise_line(ln) or ln.strip() in repeated:
            removed += 1
        else:
            cleaned.append(ln)

    return "\n".join(cleaned), removed


def remove_artifacts(text: str) -> tuple[str, int]:
    """
    Remove numbering artifacts that are not legal structural markers:

    - Lone integer lines (detached page/list numbers in extracted body text)
    - Roman-numeral-only lines (from table of contents or annex headers)
    - Table-of-contents dot-leader entries ("Neni 5 ........ 12")

    Sub-alphabetical markers (a), b), i), ii)) are intentionally preserved
    as they may be legitimate sub-items within legal provisions.

    Returns
    -------
    (cleaned_text, count_removed)
    """
    count = 0

    def _strip(pattern: re.Pattern[str], t: str) -> str:
        nonlocal count
        matches = [m for m in pattern.finditer(t) if m.group(0).strip()]
        count += len(matches)
        return pattern.sub("", t)

    text = _strip(_LONE_INT_RE,   text)
    text = _strip(_ROMAN_ONLY_RE, text)
    text = _strip(_TOC_ENTRY_RE,  text)

    return text, count


def clean(
    text:              str,
    min_repeats:       int = 3,
    collapse_blanks:   bool = True,
) -> str:
    """
    Apply all three cleaning passes in sequence and return the result.

    Passes applied:
      1. remove_page_numbers   -- standalone digit lines
      2. remove_noise_lines    -- noise + repeated header/footer boilerplate
      3. remove_artifacts      -- lone integers, Roman numerals, ToC entries

    Optionally collapses 3+ consecutive blank lines to 2.

    Returns
    -------
    str
        Cleaned text, stripped of leading/trailing whitespace.
    """
    text, _ = remove_page_numbers(text)
    text, _ = remove_noise_lines(text, min_repeats=min_repeats)
    text, _ = remove_artifacts(text)

    if collapse_blanks:
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# -- High-level ---------------------------------------------------------------

def _split_body(body: str) -> list[ParagraphBlock]:
    """
    Internal: split article body text into ParagraphBlock objects.

    Priority:
      1. Explicit numeric markers (paren > dot > brace > word)
      2. Blank-line separated blocks >= 50 characters
      3. Whole body as a single implicit Paragraph 1
    """
    # Collect all paragraph markers, deduplicating by position
    spans:  list[tuple[int, int, int, str, str]] = []
    seen:   set[int] = set()

    def _gather(pat: re.Pattern[str], style: str) -> None:
        for m in pat.finditer(body):
            pos = m.start()
            if pos not in seen:
                seen.add(pos)
                spans.append((pos, m.end(), int(m.group("num")), style, m.group(0).strip()))

    _gather(PARA_PAREN_RE, "paren")
    _gather(PARA_DOT_RE,   "dot")
    _gather(PARA_BRACE_RE, "brace")
    _gather(PARA_WORD_RE,  "word")
    spans.sort(key=lambda x: x[0])

    if spans:
        blocks: list[ParagraphBlock] = []
        for i, (pos, end, num, style, raw) in enumerate(spans):
            text_start = end
            text_end   = spans[i + 1][0] if i + 1 < len(spans) else len(body)
            ptext      = body[text_start:text_end].strip()
            if ptext:
                pm = ParagraphMatch(number=num, style=style, raw=raw,
                                    start=pos, end=end)
                blocks.append(ParagraphBlock(match=pm, text=ptext))
        if blocks:
            return blocks

    # Fallback: blank-line separated blocks of substance
    parts = [p.strip() for p in re.split(r"\n\s*\n", body)
             if len(p.strip()) >= 50]
    if len(parts) > 1:
        return [ParagraphBlock(match=None, text=p) for p in parts]

    # Last resort: whole body as one paragraph
    if body.strip():
        return [ParagraphBlock(match=None, text=body.strip())]

    return []


def extract_structure(text: str) -> list[ArticleBlock]:
    """
    Parse a legal document text into a hierarchy of ArticleBlock objects.

    Steps
    -----
    1. Locate all article headers with find_articles()
    2. Extract each article's body (text between headers)
    3. Detect a same-next-line title if the header has no inline title
    4. Split each body into ParagraphBlock objects

    Preamble text before the first article header is discarded.

    Parameters
    ----------
    text : source document text (may be raw or pre-cleaned)

    Returns
    -------
    list[ArticleBlock]
        One entry per article, in document order.
        Empty list if no article headers are found.
    """
    article_matches = find_articles(text)
    if not article_matches:
        return []

    blocks: list[ArticleBlock] = []
    for i, art in enumerate(article_matches):
        body_start = art.end
        body_end   = article_matches[i + 1].start if i + 1 < len(article_matches) else len(text)
        body       = text[body_start:body_end].strip()

        # If the header had no title, try the first body line as the title.
        # Criteria: short, does not start with a digit, not another header,
        # not a paragraph marker.
        title = art.title
        if not title and body:
            first = body.splitlines()[0].strip()
            if (
                first
                and len(first) < 100
                and not first[0].isdigit()
                and not ARTICLE_RE.match(first)
                and not PARA_ANY_RE.match(first)
                and not is_noise_line(first)
            ):
                title = first
                body  = "\n".join(body.splitlines()[1:]).strip()

        # Rebuild match with updated title
        updated = ArticleMatch(
            keyword=art.keyword,
            number=art.number,
            title=title,
            raw=art.raw,
            start=art.start,
            end=art.end,
        )

        paras = _split_body(body)
        blocks.append(ArticleBlock(match=updated, body=body, paragraphs=paras))

    return blocks


def iter_chunks(text: str) -> Iterator[tuple[ArticleMatch, ParagraphBlock]]:
    """
    Convenience generator yielding (ArticleMatch, ParagraphBlock) pairs,
    one per paragraph in document order.

    Equivalent to iterating extract_structure() and its .paragraphs lists,
    but flattened for callers that want a stream of atomic legal provisions.

    Yields
    ------
    (ArticleMatch, ParagraphBlock)
    """
    for block in extract_structure(text):
        for para in block.paragraphs:
            yield block.match, para


def count_articles(text: str) -> int:
    """Return the number of article headers in *text*."""
    return len(find_articles(text))


def count_paragraphs(text: str) -> int:
    """Return the number of explicit paragraph markers in *text*."""
    return len(find_paragraphs(text))


# -- CLI / smoke-test ---------------------------------------------------------

if __name__ == "__main__":
    import sys

    _SAMPLE = """\
Law No. 05/L-085 on Electricity
Assembly of Kosovo
Gazeta Zyrtare e Republikes se Kosoves
3

Neni 1 -- Qellimi

(1) Ky ligj ka per qellim rregullimin e sektorit te energjise elektrike.
(2) Dispozitat e ketij ligji zbatohen per te gjithe operatoret e sistemit.

Neni 3 -- Pergjegjesite e KOSTT

1. KOSTT pergjigjet per operimin e sigurt te sistemit te transmetimit.
2. KOSTT publikon kushtet dhe termat per qasje ne rrjet.

Article 5 -- Tariffs

(1) All tariffs shall be approved by ZRRE before publication.
(2) Tariffs shall be non-discriminatory and cost-reflective.
(3) Connection refusals must be communicated in writing within 15 days.

Clan 7 -- Pravni osnov

(1) Clan definise pravni osnov za rad operatora.

sipas nenit 5 te ketij ligji procedura behet ...
"""

    print("=" * 60)
    print("ARTICLE DETECTION")
    print("=" * 60)
    for art in find_articles(_SAMPLE):
        print(f"  {art.label:<45}  offset={art.start}")

    print()
    print("=" * 60)
    print("PARAGRAPH DETECTION")
    print("=" * 60)
    for para in find_paragraphs(_SAMPLE):
        print(f"  [{para.style:<5}] {para.label:<20}  raw={para.raw!r:<12}  offset={para.start}")

    print()
    print("=" * 60)
    print("INLINE CROSS-REFERENCES")
    print("=" * 60)
    for ref in find_inline_references(_SAMPLE):
        print(f"  Article {ref['number']}  raw={ref['raw']!r}")

    print()
    print("=" * 60)
    print("CLASSIFICATION")
    print("=" * 60)
    test_lines = [
        "Neni 5 -- Qellimi",
        "Article 12",
        "sipas nenit 5 te ketij ligji",
        "(1) KOSTT is responsible...",
        "1. The operator shall...",
        "Paragrafi 2 defines...",
        "3",
        "- 5 -",
        "Gazeta Zyrtare e Republikes se Kosoves",
        "Neni 5 .............. 12",
        "",
        "Normal legal sentence.",
    ]
    for ln in test_lines:
        print(f"  article={is_article_header(ln)!s:<5}  "
              f"para={is_paragraph_start(ln)!s:<5}  "
              f"noise={is_noise_line(ln)!s:<5}  "
              f"{ln!r}")

    print()
    print("=" * 60)
    print("CLEANING")
    print("=" * 60)
    cleaned, n = remove_noise_lines(_SAMPLE)
    print(f"  remove_noise_lines: {n} line(s) removed")
    c2, n2 = remove_artifacts(_SAMPLE)
    print(f"  remove_artifacts  : {n2} artifact(s) removed")
    print(f"  count_articles    : {count_articles(_SAMPLE)}")
    print(f"  count_paragraphs  : {count_paragraphs(_SAMPLE)}")

    print()
    print("=" * 60)
    print("STRUCTURED EXTRACTION")
    print("=" * 60)
    for block in extract_structure(clean(_SAMPLE)):
        print(f"  {block}")
        for para in block.paragraphs:
            print(f"    [{para.label}]  {para.text[:60]}{'...' if len(para.text) > 60 else ''}")

    print()
    print("=" * 60)
    print("iter_chunks (flat stream)")
    print("=" * 60)
    for art, para in iter_chunks(clean(_SAMPLE)):
        print(f"  {art.label}  |  {para.label}  |  {para.text[:50]}...")
