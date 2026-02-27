"""Placeholder for EUR-Lex SOAP client integration.

EUR-Lex provides a SOAP web service that requires registration and credentials.
It returns XML responses, which can be parsed into tabular structures. This
module defines a placeholder interface to plug in future support without
coupling the ingestion CLI to SOAP today.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class EurLexSoapConfig:
    """Configuration parameters for the SOAP client."""

    wsdl_url: str
    username: str
    password: str
    enabled: bool = False


class EurLexSoapClient:
    """Interface for a future EUR-Lex SOAP client.

    Methods here should mirror the CELLAR client surface so that ingestion
    controllers can swap implementations with minimal changes.
    """

    def __init__(self, config: EurLexSoapConfig) -> None:
        self.config = config

    def search(self, query: str, page: int, page_size: int) -> pd.DataFrame:
        """Executes a search query via SOAP and returns a DataFrame.

        NOTE: Not implemented. Requires SOAP integration and credentials.
        """
        raise NotImplementedError(
            "EUR-Lex SOAP client is a placeholder. Register for access and implement."
        )

# TODO:
# - Implement SOAP bindings (e.g., with zeep) once credentials are available.
# - Map SOAP XML structures to the same tidy schema used by CELLAR ingestion.

