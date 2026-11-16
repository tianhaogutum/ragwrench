"""Small public value types; source line spans are inclusive and one-based."""
from dataclasses import dataclass, field
from typing import Any


class TableChunkError(ValueError):
    """Input cannot be represented safely within the requested constraints."""


@dataclass
class Block:
    text: str
    line_start: int
    line_end: int
    kind: str = "prose"
    level: int = 0


@dataclass
class HTMLNormalization:
    """Source identities for a normalized HTML grid; None means no source cell."""

    source_cells: dict[str, dict[str, Any]]
    header_cell_ids: list[list[str | None]]
    row_cell_ids: list[list[str | None]]
    diagnostics: list[dict[str, Any]]

    def required_cells(self, rows):
        needed = {cell for row in self.header_cell_ids + rows for cell in row if cell is not None}
        # Body headers can lie outside this chunk's grid, including later rows.
        return needed | {ref["cell_id"] for cell in needed
                         for ref in self.source_cells[cell].get("headers", [])}

    def for_rows(self, start: int, end: int) -> dict[str, Any]:
        rows = self.row_cell_ids[start:end]
        needed = self.required_cells(rows)
        diagnostics = []
        for diagnostic in self.diagnostics:
            if "cell_id" in diagnostic and diagnostic["cell_id"] not in needed:
                continue
            if diagnostic.get("section") == "rows" and not start <= diagnostic["row_index"] < end:
                continue
            diagnostics.append(diagnostic)
        return {"schema_version": 1, "header_cell_ids": self.header_cell_ids, "row_cell_ids": rows,
                "source_cells": {key: self.source_cells[key] for key in sorted(needed)},
                "diagnostics": diagnostics}


@dataclass
class Table:
    headers: list[list[str]]
    rows: list[list[str]]
    row_lines: list[tuple[int, int]]
    line_start: int
    line_end: int
    caption: str = ""
    warnings: list[str] = field(default_factory=list)
    normalization: HTMLNormalization | None = None

    @property
    def columns(self) -> list[str]:
        result = []
        for col in range(len(self.headers[0])):
            labels = []
            previous_cell = None
            for index, row in enumerate(self.headers):
                label = row[col] or f"column_{col + 1}"
                cell = self.normalization.header_cell_ids[index][col] if self.normalization else None
                # Only a carried copy of the same source header loses a level.
                # Equal text in distinct HTML cells is still a distinct level.
                if self.normalization:
                    repeated = cell is not None and cell == previous_cell
                else:
                    repeated = bool(labels) and labels[-1] == label
                if not repeated:
                    labels.append(label)
                previous_cell = cell
            result.append(" / ".join(labels))
        return result


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]

    def as_dict(self, adapter: str = "jsonl") -> dict[str, Any]:
        if adapter == "jsonl":
            return {"id": self.id, "text": self.text, "metadata": self.metadata}
        if adapter == "langchain":
            return {"id": self.id, "page_content": self.text, "metadata": self.metadata}
        if adapter == "llamaindex":
            return {"id_": self.id, "text": self.text, "metadata": self.metadata}
        raise TableChunkError(f"unknown adapter: {adapter}")
