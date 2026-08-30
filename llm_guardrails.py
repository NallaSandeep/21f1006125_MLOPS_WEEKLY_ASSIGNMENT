"""Fail-closed guardrails for the Week 11 Iris LLM classification service.

The wrapper deliberately receives the *untrusted raw prompt*, rather than a
pre-parsed object.  This prevents instruction text from being hidden in a
feature field before validation takes place.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SPECIES = ("setosa", "versicolor", "virginica")
FALLBACK = "Request rejected: response did not satisfy the Iris classification policy."

# These expressions correspond to the injection and leakage probes in the
# red-team suite.  Keep patterns specific so ordinary measurement text is not
# needlessly rejected.
INJECTION_RULES: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(r"\b(ignore|disregard|override)\b.{0,80}\b(previous|prior|above|instructions?)\b", re.I | re.S),
    "role_play": re.compile(r"\b(you are now|act as|pretend to be|role[- ]?play)\b", re.I),
    "prompt_exfiltration": re.compile(r"\b(system prompt|context window|training examples?|few[- ]?shot|repeat everything above|instructions were you given)\b", re.I),
    "delimiter_escape": re.compile(r"(<\|[^|>]+\|>|\[/?INST\]|###\s*(system|instruction|assistant)|```(?:system|prompt)?)", re.I),
}

V1_SCHEMA = re.compile(
    r"^\s*sepal_length:\s*[-+]?\d+(?:\.\d+)?\s*,\s*"
    r"sepal_width:\s*[-+]?\d+(?:\.\d+)?\s*,\s*"
    r"petal_length:\s*[-+]?\d+(?:\.\d+)?\s*,\s*"
    r"petal_width:\s*[-+]?\d+(?:\.\d+)?\s*$",
    re.I,
)
V2_SCHEMA = re.compile(
    r"^\s*A flower specimen has a sepal length of [-+]?\d+(?:\.\d+)? cm, "
    r"sepal width of [-+]?\d+(?:\.\d+)? cm, petal length of [-+]?\d+(?:\.\d+)? cm, "
    r"and petal width of [-+]?\d+(?:\.\d+)? cm\. Identify the iris species\.\s*$",
    re.I,
)
OUTPUT_FORMATS = {
    "v1": re.compile(r"^\s*(?:iris[- ]?)?(setosa|versicolor|virginica)\s*\.?\s*$", re.I),
    "v2": re.compile(r"^\s*This is Iris (setosa|versicolor|virginica)\.\s*$", re.I),
}
LEAKAGE_MARKERS = re.compile(
    r"\b(system prompt|context window|few[- ]?shot|training examples?|"
    r"Classify the following Iris flower|sepal_length:|output_text|input_text|"
    r"ignore previous instructions)\b|<\|(?:system|assistant)\|>|\[INST\]",
    re.I,
)


class GuardedIrisPipeline:
    """Validate a prompt, call ``predict_fn``, then validate its text output."""

    def __init__(self, predict_fn: Callable[[str, str], str], audit_path: str | Path = "artifacts/guardrails/audit.jsonl"):
        self.predict_fn = predict_fn
        self.audit_path = Path(audit_path)

    def _audit(self, event: str, rule: str, raw_input: str | None = None, raw_response: str | None = None) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "rule": rule}
        if raw_input is not None:
            entry["raw_input"] = raw_input
        if raw_response is not None:
            entry["raw_response"] = raw_response
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _schema_matches(raw_input: str, version: str) -> bool:
        # Accept JSON only when it contains exactly the four numeric feature
        # fields; this makes the public wrapper useful for API clients too.
        try:
            data = json.loads(raw_input)
            if isinstance(data, Mapping) and set(data) == {"sepal_length", "sepal_width", "petal_length", "petal_width"}:
                return all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in data.values())
        except (TypeError, ValueError):
            pass
        return (V1_SCHEMA if version == "v1" else V2_SCHEMA).fullmatch(raw_input) is not None

    def validate_input(self, raw_input: str, version: str) -> dict[str, Any] | None:
        if version not in OUTPUT_FORMATS:
            raise ValueError("version must be 'v1' or 'v2'")
        for name, pattern in INJECTION_RULES.items():
            if pattern.search(raw_input):
                self._audit("input_blocked", name, raw_input=raw_input)
                return {"blocked": True, "reason": f"Blocked by {name}"}
        if not self._schema_matches(raw_input, version):
            self._audit("input_blocked", "invalid_iris_feature_schema", raw_input=raw_input)
            return {"blocked": True, "reason": "Blocked by invalid_iris_feature_schema"}
        return None

    def filter_output(self, raw_response: str, version: str) -> dict[str, Any]:
        if LEAKAGE_MARKERS.search(raw_response):
            self._audit("output_filtered", "context_leakage", raw_response=raw_response)
            return {"blocked": True, "reason": "context_leakage", "response": FALLBACK}
        if not OUTPUT_FORMATS[version].fullmatch(raw_response):
            self._audit("output_filtered", "format_violation", raw_response=raw_response)
            return {"blocked": True, "reason": "format_violation", "response": FALLBACK}
        return {"blocked": False, "response": raw_response.strip()}

    def predict(self, raw_input: str, version: str) -> dict[str, Any]:
        blocked = self.validate_input(raw_input, version)
        if blocked:
            return blocked
        raw_response = self.predict_fn(raw_input, version)
        return self.filter_output(str(raw_response), version)
