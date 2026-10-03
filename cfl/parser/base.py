from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ImportEntry:
    module: str
    name: str | None
    alias: str | None
    level: int
    line: int
    is_from: bool


@dataclass
class CallSite:
    expr: str
    line: int
    kind: str  # 'call' | 'decorator'
    control_ctx: str


@dataclass
class ParsedSymbol:
    kind: str
    qualname: str
    name: str
    parent_qualname: str | None
    signature: str
    decorators: list[str]
    docstring: str | None
    start_line: int
    end_line: int
    raw_code: str
    is_async: bool
    bases: list[str]
    init_attrs: dict[str, str]
    param_types: dict[str, str]
    local_types: dict[str, str]
    call_sites: list[CallSite]


@dataclass
class ParsedFile:
    path: str
    language: str
    imports: list[ImportEntry]
    exports: list[str]
    symbols: list[ParsedSymbol]
    parse_error: str | None


class LanguageAdapter(Protocol):
    language: str
    extensions: tuple[str, ...]

    def parse(self, path: pathlib.Path, source: str) -> ParsedFile: ...

    def statement_spans(self, raw_code: str) -> list[tuple[int, int]]: ...

    def skeleton(self, parsed: ParsedFile) -> str: ...


_adapter_registry: dict[str, LanguageAdapter] = {}


def register_adapter(adapter: LanguageAdapter) -> None:
    """Register a language adapter by its language name."""
    _adapter_registry[adapter.language] = adapter


def get_adapter(language: str) -> LanguageAdapter:
    """Return the adapter for the given language.

    Raises KeyError if no adapter is registered for that language.
    """
    if language not in _adapter_registry:
        raise KeyError(f"No adapter registered for language: {language!r}")
    return _adapter_registry[language]
