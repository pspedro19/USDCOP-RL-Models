"""Typed Airflow dataset URIs and the forbidden forecast→allocator edge (BL-35)."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


class DatasetContractError(ValueError):
    pass


ALLOWED_SCHEMES = frozenset(
    {
        "asset",
        "strategy",
        "action",
        "forecast",
        "artifact",
        "portfolio",
        "exec",
        "control",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetURI:
    scheme: str
    authority: str
    path: str

    @classmethod
    def parse(cls, raw: str) -> "DatasetURI":
        if not isinstance(raw, str):
            raise DatasetContractError("dataset URI must be a string")
        parsed = urlparse(raw)
        if parsed.scheme not in ALLOWED_SCHEMES:
            raise DatasetContractError(f"unsupported dataset scheme {parsed.scheme!r}")
        if not parsed.netloc or not parsed.path.strip("/"):
            raise DatasetContractError(f"dataset URI requires authority and path: {raw!r}")
        if parsed.params or parsed.query or parsed.fragment:
            raise DatasetContractError("dataset URI cannot contain params, query or fragment")
        if parsed.username or parsed.password or parsed.port:
            raise DatasetContractError("dataset URI authority cannot contain credentials or port")
        if any(part in {"", ".", ".."} for part in parsed.path.strip("/").split("/")):
            raise DatasetContractError("dataset URI path contains an empty or traversal segment")
        return cls(parsed.scheme, parsed.netloc, parsed.path.strip("/"))

    def __str__(self) -> str:
        return f"{self.scheme}://{self.authority}/{self.path}"


def validate_dataset_edges(edges: list[dict[str, str]]) -> None:
    """Reject any direct prediction consumption by book/allocation/execution."""

    for index, edge in enumerate(edges):
        source = DatasetURI.parse(edge.get("source", ""))
        target = DatasetURI.parse(edge.get("target", ""))
        if source.scheme == "forecast" and target.scheme in {
            "strategy",
            "action",
            "portfolio",
            "exec",
        }:
            raise DatasetContractError(
                f"edge[{index}] {source} -> {target} is forbidden: "
                "forecast_output is DIAGNOSTIC and allocator accepts strategy_output only"
            )
