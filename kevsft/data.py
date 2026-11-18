"""Validate grouped route datasets or render a matching Kev HTTP request."""

import argparse
import json
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

INSTRUCTIONS = (
    "Choose the route that can handle the current request. Use earlier user "
    "messages when the current request refers to them. Choose other if no "
    "listed route applies or the request has too little context to decide."
)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Example:
    state: str
    label: int
    group: str


def _object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def load_labels(data: Path) -> dict[str, str]:
    """Read the ordered Kev criteria. The final key is the fallback route."""
    path = Path(data) / "labels.json"
    try:
        labels = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
        if not isinstance(labels, dict) or not 2 <= len(labels) <= 255:
            raise ValueError("expected 2 to 255 route keys")
        if list(labels)[-1] != "other":
            raise ValueError("put the fallback key 'other' last")
        for key in labels:
            if _text(key, "route key") != key:
                raise ValueError("route keys may not have surrounding whitespace")
        return {key: _text(description, "route description") for key, description in labels.items()}
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from error


def make_question(labels: dict[str, str]) -> dict:
    """Construct the choice question accepted by POST /v1/systemone."""
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(labels)}


def render_state(query: str, history: Sequence[str] = ()) -> str:
    """Serialize chronological prior user turns and the current user request."""
    current = _text(query, "query")
    if isinstance(history, (bytes, str)) or not isinstance(history, Sequence):
        raise ValueError("history must be an array of previous user messages")
    messages = [_text(value, "history entry") for value in history]
    if not messages:
        return current
    # JSON string quoting preserves turn boundaries even when a message has newlines.
    return json.dumps(
        {"previous_user_requests": messages, "current_user_request": current},
        ensure_ascii=False,
    )


def make_request(labels: dict[str, str], query: str, history: Sequence[str] = (), model: str = "kev-latest") -> dict:
    return {
        "model": model,
        "state": render_state(query, history),
        "questions": {"route": make_question(labels)},
    }


def _example(row: object, indices: dict[str, int]) -> Example:
    if not isinstance(row, dict):
        raise ValueError("expected a JSON object")
    if not {"query", "label", "group"} <= row.keys() <= {"query", "label", "group", "history"}:
        raise ValueError("expected query, label, group, and optional history")
    label = _text(row["label"], "label")
    if label not in indices:
        raise ValueError("label is missing from labels.json")
    return Example(
        state=render_state(row["query"], row.get("history", [])),
        label=indices[label],
        group=_text(row["group"], "group"),
    )


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_dataset(data: Path) -> tuple[list[str], dict[str, list[Example]]]:
    """Verify split isolation and return the exact options used by Kev's API."""
    data = Path(data)
    labels = load_labels(data)
    indices = {key: index for index, key in enumerate(labels)}
    options = [f"{key}: {description}" for key, description in labels.items()]
    groups: dict[str, str] = {}
    seen: dict[str, tuple[str, int]] = {}
    splits: dict[str, list[Example]] = {}
    for split in SPLITS:
        path = data / f"{split}.jsonl"
        examples = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    example = _example(json.loads(line, object_pairs_hook=_object), indices)
                    if groups.setdefault(_normalized(example.group), split) != split:
                        raise ValueError("source group occurs in multiple splits")
                    state = _normalized(example.state)
                    first_split, first_label = seen.setdefault(state, (split, example.label))
                    if first_split != split:
                        raise ValueError("equivalent input occurs in multiple splits")
                    if first_label != example.label:
                        raise ValueError("equivalent input has conflicting labels")
                    examples.append(example)
                except ValueError as error:
                    raise ValueError(f"{path.name}:{number}: {error}") from error
        if not examples:
            raise ValueError(f"{path.name}: empty split")
        splits[split] = examples
    if len({example.label for example in splits["train"]}) != len(labels):
        raise ValueError("train.jsonl needs examples for every route, including other")
    return options, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing labels.json")
    parser.add_argument("--request", help="Print a request for this query instead of validating splits")
    parser.add_argument("--history", action="append", default=[], help="Previous user turn, oldest first")
    parser.add_argument("--model", default="kev-latest", help="Model name to use in the request")
    args = parser.parse_args()
    try:
        labels = load_labels(args.data)
        if args.request is not None:
            result = make_request(labels, args.request, args.history, model=args.model)
        else:
            if args.history:
                parser.error("--history requires --request")
            _, splits = load_dataset(args.data)
            keys = list(labels)
            result = {}
            for name, examples in splits.items():
                result[name] = {
                    "rows": len(examples),
                    "groups": len({e.group for e in examples}),
                    "per_route": dict(Counter(keys[e.label] for e in examples)),
                }
    except (OSError, ValueError) as error:
        parser.exit(2, f"Dataset error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
