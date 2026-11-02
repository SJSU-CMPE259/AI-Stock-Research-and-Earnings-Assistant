"""Extract sections/Items from SEC filings."""
import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """A section extracted from a filing."""

    name: str  # e.g., "Item 1A. Risk Factors"
    text: str  # Cleaned text content


class FilingParser:
    """Parse SEC filing HTML and extract sections."""

    # Item patterns for 10-K and 10-Q
    ITEM_PATTERNS = {
        "10-K": {
            "Item 1": r"Item\s+1\.?\s+Business",
            "Item 1A": r"Item\s+1A\.?\s+Risk\s+Factors",
            "Item 7": r"Item\s+7\.?\s+Management.*s\s+Discussion",
            "Item 7A": r"Item\s+7A\.?\s+Quantitative",
        },
        "10-Q": {
            "Item 2": r"Item\s+2\.?\s+Management.*s\s+Discussion",
            "Item 1A": r"Item\s+1A\.?\s+Risk\s+Factors",
        },
    }

    @staticmethod
    def _clean_text(html: str) -> str:
        """Convert HTML to clean text, removing scripts/styles."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "ix:header"]):
            script.decompose()

        # Get text and normalize whitespace
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    # The real body of an Item runs to at least this many characters; shorter
    # spans are table-of-contents entries, which we skip.
    MIN_BODY_CHARS = 2000

    @classmethod
    def extract_items(cls, filing_html: str, form: str) -> list[Section]:
        """Extract specified Items from a filing.

        An Item name (e.g. "Item 1A. Risk Factors") appears twice: once in the
        table of contents and once at the real section. We scan every occurrence
        and keep the one with the most body text, so we never return the ToC stub.

        Args:
            filing_html: Raw HTML content
            form: Form type (10-K or 10-Q)

        Returns:
            List of Section objects
        """
        text = cls._clean_text(filing_html)
        sections = []

        patterns = cls.ITEM_PATTERNS.get(form, {})
        if not patterns:
            logger.warning(f"No patterns defined for form {form}")
            return sections

        for item_name, pattern in patterns.items():
            body = cls._best_occurrence(text, pattern)
            if body is None:
                logger.warning(f"{form} {item_name} not found")
            elif len(body) < cls.MIN_BODY_CHARS:
                logger.warning(
                    f"{form} {item_name}: only {len(body)} chars (likely ToC only), skipping"
                )
            else:
                sections.append(Section(name=item_name, text=body))

        return sections

    @staticmethod
    def _best_occurrence(text: str, pattern: str) -> str | None:
        """Return the longest Item->next-Item span across all matches of `pattern`."""
        bodies = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = match.start()
            nxt = re.search(r"\nItem\s+\d", text[match.end():], re.IGNORECASE)
            end = (match.end() + nxt.start()) if nxt else len(text)
            bodies.append(text[start:end].strip())
        return max(bodies, key=len) if bodies else None


def parse_filing(filing_html: str, form: str) -> list[Section]:
    """Parse a filing and extract items.

    Args:
        filing_html: Raw HTML
        form: Form type (10-K or 10-Q)

    Returns:
        List of extracted sections
    """
    return FilingParser.extract_items(filing_html, form)
