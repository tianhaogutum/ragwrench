"""Per-call counters only: never retain queries, predicates or backend errors."""
from dataclasses import asdict, dataclass


@dataclass
class Scan:
    candidate_budget: int | None = None
    page_size: int | None = None
    pages_fetched: int = 0
    candidates_fetched: int = 0
    candidates_evaluated: int = 0
    filter_rejected: int = 0
    threshold_rejected: int = 0
    lexical_misses: int = 0
    eligible_count: int = 0
    returned_count: int = 0
    budget_probe_points: int = 0
    stop_reason: str = ""
    native_condition_count: int = 0
    snapshot_version: int | None = None

    def finish(self, records, reason):
        self.returned_count = len(records)
        self.stop_reason = reason
        return records

    def report(self, backend, query, predicate, top_k, threshold):
        counts = asdict(self)
        native = counts.pop("native_condition_count")
        if counts["snapshot_version"] is None:
            del counts["snapshot_version"]
        return {"backend": backend, "query_characters": len(query),
                "top_k": top_k, "score_threshold": threshold,
                "filter": {"logical_operator": predicate.logical,
                           "condition_count": len(predicate.conditions),
                           "native_condition_count": native}, **counts}
