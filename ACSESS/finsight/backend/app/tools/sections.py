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

    @classmethod
    def extract_items(cls, filing_html: str, form: str) -> list[Section]:
        """Extract specified Items from a filing.

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
            # Find the section
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                logger.warning(f"{form} {item_name} not found")
                continue

            start = match.start()

            # Find the next Item (or end of text)
            next_match = re.search(
                r"\nItem\s+\d", text[start + 1 :], re.IGNORECASE
            )
            end = (start + 1 + next_match.start()) if next_match else len(text)

            # Extract section text
            section_text = text[start:end].strip()

            # Require minimum length
            if len(section_text) > 40:
                sections.append(Section(name=item_name, text=section_text))
            else:
                logger.warning(f"{form} {item_name} text too short, skipping")

        return sections


def parse_filing(filing_html: str, form: str) -> list[Section]:
    """Parse a filing and extract items.

    Args:
        filing_html: Raw HTML
        form: Form type (10-K or 10-Q)

    Returns:
        List of extracted sections
    """
    return FilingParser.extract_items(filing_html, form)
