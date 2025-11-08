"""Hydra CLI entrypoint for regulatory ingestion (Step 1 - CELLAR)."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .cellar_sparql import fetch_all
from .io_utils import ensure_dirs, get_logger, safe_to_parquet


@hydra.main(config_path="../../../../conf", config_name="regulatory/ingest.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = get_logger("regulatory.ingest", cfg.get("log_level", "INFO"))
    logger.info("Starting regulatory ingestion")

    # Resolve paths from config
    raw_dir = cfg.paths.raw_dir
    ensure_dirs(raw_dir)

    # Route by source
    source = cfg.source
    if source == "cellar":
        endpoint = cfg.regulatory.sources.endpoint
        timeout_s = int(cfg.regulatory.sources.timeout_s)
        retries = int(cfg.regulatory.sources.retries)

        df = fetch_all(
            endpoint=endpoint,
            query_terms=list(cfg.query_terms or []),
            eurovoc_uris=list(cfg.eurovoc_uris or []),
            language=str(cfg.language),
            date_from=str(cfg.date_from),
            date_to=str(cfg.date_to),
            page_size=int(cfg.page_size),
            max_pages=int(cfg.max_pages),
            timeout_s=timeout_s,
            retries=retries,
            logger=logger,
        )

        outfile = hydra.utils.to_absolute_path(cfg.outfile)
        ensure_dirs(outfile)
        safe_to_parquet(df, outfile)

        logger.info(f"Rows: {len(df)} | Out: {outfile}")
        if not df.empty:
            logger.info("Sample:")
            logger.info(df.head(5).to_string(index=False))
    elif source == "eurlex_soap":
        logger.warning("EUR-Lex SOAP source is not implemented yet. Exiting.")
    else:
        logger.error(f"Unknown source: {source}")
        sys.exit(2)


if __name__ == "__main__":
    main()

