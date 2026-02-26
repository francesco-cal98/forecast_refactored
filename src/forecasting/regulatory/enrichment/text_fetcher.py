"""Text fetching and caching for EUR-Lex documents."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


class TextFetcher:
    """Fetches full text from EUR-Lex HTML pages with caching and rate limiting."""

    def __init__(
        self,
        cache_dir: Path | str = "data/regulatory/cache",
        rate_limit_delay: float = 1.0,
        timeout: int = 30,
        retries: int = 3,
    ):
        """Initialize text fetcher.

        Args:
            cache_dir: Directory for caching downloaded HTML
            rate_limit_delay: Seconds to wait between requests
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.retries = retries
        self.last_request_time = 0.0

        logger.info(f"TextFetcher initialized with cache_dir={self.cache_dir}")

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL hash."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.txt"

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL with retries."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Research/Educational purposes)",
            "Accept": "text/html,application/xhtml+xml",
        }

        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _extract_text_from_html(self, html: str) -> Optional[str]:
        """Extract text content from EUR-Lex HTML.

        Args:
            html: Raw HTML content

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # EUR-Lex uses various div IDs for content
            # Try multiple selectors in order of preference
            selectors = [
                {"id": "text"},  # Main text container
                {"id": "document"},
                {"class": "eli-main-title"},
                {"class": "ti-art"},  # Article text
                {"id": "texte"},  # French version
            ]

            for selector in selectors:
                content_div = soup.find("div", selector)
                if content_div:
                    # Extract text, preserve paragraph structure
                    text = content_div.get_text(separator="\n", strip=True)
                    if text and len(text) > 100:  # Minimum length check
                        return text

            # Fallback: get all text from body
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)
                if len(text) > 100:
                    return text

            logger.warning("No suitable text container found in HTML")
            return None

        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return None

    def fetch_text(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetch full text from EUR-Lex URL.

        Args:
            url: EUR-Lex document URL
            use_cache: Whether to use cached version if available

        Returns:
            Document text or None if fetch/extraction fails
        """
        cache_path = self._get_cache_path(url)

        # Check cache first
        if use_cache and cache_path.exists():
            logger.debug(f"Loading from cache: {url}")
            try:
                return cache_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Cache read error: {e}, fetching fresh")

        # Fetch from web
        try:
            self._rate_limit()
            logger.info(f"Fetching: {url}")

            html = self._fetch_html(url)
            text = self._extract_text_from_html(html)

            # Cache successful extraction
            if text:
                cache_path.write_text(text, encoding="utf-8")
                logger.debug(f"Cached to {cache_path}")

            return text

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        link_column: str = "link_html",
        text_column: str = "full_text",
        max_docs: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Enrich DataFrame with full text from URLs.

        Args:
            df: Input DataFrame with link column
            link_column: Column containing EUR-Lex URLs
            text_column: Column name for extracted text
            max_docs: Maximum documents to process (for testing)
            use_cache: Whether to use cached texts

        Returns:
            DataFrame with added text column
        """
        if max_docs:
            df = df.head(max_docs).copy()
        else:
            df = df.copy()

        texts = []
        success_count = 0
        total = len(df)

        logger.info(f"Starting enrichment of {total} documents")

        for idx, row in df.iterrows():
            url = row.get(link_column)

            if pd.isna(url) or not url:
                logger.warning(f"Row {idx}: Missing URL")
                texts.append(None)
                continue

            text = self.fetch_text(url, use_cache=use_cache)
            texts.append(text)

            if text:
                success_count += 1

            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{total} ({success_count} successful, "
                    f"{success_count / (idx + 1) * 100:.1f}% success rate)"
                )

        df[text_column] = texts

        logger.info(
            f"Enrichment complete: {success_count}/{total} documents "
            f"({success_count / total * 100:.1f}% success rate)"
        )

        return df
