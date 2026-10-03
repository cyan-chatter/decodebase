from __future__ import annotations


class CflError(Exception):
    """Base exception for all CodeFlowLens errors."""


class BudgetExceeded(CflError):
    """Raised when a token or resource budget is exceeded."""


class PreflightError(CflError):
    """Raised when a preflight / doctor check fails."""


class LLMTransportError(CflError):
    """Raised on HTTP / network errors communicating with Ollama."""


class LLMValidationError(CflError):
    """Raised when the LLM returns an unexpected or unparseable response."""


class AmbiguousSymbol(CflError):
    """Raised when a symbol name matches more than one definition."""

    def __init__(self, candidates: list) -> None:
        self.candidates = candidates
        super().__init__(
            f"Ambiguous symbol: {len(candidates)} candidates found: {candidates}"
        )
