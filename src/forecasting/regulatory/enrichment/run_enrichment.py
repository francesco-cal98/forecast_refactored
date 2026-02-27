"""CLI entrypoint for text enrichment."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .text_fetcher import TextFetcher


CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "conf")


@hydra.main(config_path=CONFIG_DIR, config_name="regulatory/enrichment", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Enrich metadata parquet with full document text."""
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(cfg.get("log_level", "INFO"))

    logger.info("Starting text enrichment")
    logger.info(f"Config: {dict(cfg)}")

    # Load metadata parquet
    input_file = cfg.input_file
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Run metadata extraction first:")
        logger.info("  python -m src.forecasting.regulatory.ingest.run_ingest")
        sys.exit(1)

    logger.info(f"Loading metadata from: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df)} documents")

    # Initialize fetcher
    fetcher = TextFetcher(
        cache_dir=cfg.cache_dir,
        rate_limit_delay=cfg.rate_limit_delay,
        timeout=cfg.timeout,
        retries=cfg.retries,
    )

    # Enrich with text
    max_docs = cfg.get("max_docs", None)
    if max_docs:
        logger.info(f"Processing only first {max_docs} documents (test mode)")

    df_enriched = fetcher.enrich_dataframe(
        df,
        link_column=cfg.link_column,
        text_column=cfg.text_column,
        max_docs=max_docs,
        use_cache=cfg.use_cache,
    )

    # Save enriched parquet
    output_file = hydra.utils.to_absolute_path(cfg.output_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_enriched.to_parquet(output_file, index=False)
    logger.info(f"Saved enriched data to: {output_file}")

    # Statistics
    text_col = cfg.text_column
    total = len(df_enriched)
    with_text = df_enriched[text_col].notna().sum()
    avg_length = df_enriched[text_col].str.len().mean()

    logger.info("=" * 80)
    logger.info("ENRICHMENT STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total documents: {total}")
    logger.info(f"With text: {with_text} ({with_text / total * 100:.1f}%)")
    logger.info(f"Average text length: {avg_length:.0f} characters")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
