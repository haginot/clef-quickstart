"""Minimal command-line runner for Cloudflare Clef decision models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MODEL_IDS = {
    "clef": "Cloudflare/clef",
    "clef-flash": "Cloudflare/clef-flash",
}
QUESTION_TYPES = {"noul", "choice", "score"}


def validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise TypeError("The request must be a JSON object")
    if "state" not in request:
        raise ValueError("state is required")
    if request.get("model") not in MODEL_IDS:
        allowed = ", ".join(MODEL_IDS)
        raise ValueError(f"model must be one of: {allowed}")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise TypeError(f"{question_id}: question must be an object")
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"{question_id}: choice criteria needs at least two options")
        if question_type == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"{question_id}: score criteria needs at least two levels")
    return request


def load_request(path: Path) -> dict[str, Any]:
    return validate_request(json.loads(path.read_text(encoding="utf-8")))


def run(request: dict[str, Any], device: str) -> dict[str, Any]:
    try:
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install the dependencies in requirements.txt first") from error

    model_id = MODEL_IDS[request["model"]]
    model_path = snapshot_download(model_id)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(model_path, device=device, dtype=torch.bfloat16)
    return systemone(model, processor, request)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a typed decision with Cloudflare Clef")
    parser.add_argument("input", type=Path, help="SystemOne-compatible JSON request")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSON without downloading or loading a model",
    )
    args = parser.parse_args()
    try:
        request = load_request(args.input)
        if args.validate_only:
            print(json.dumps(request, ensure_ascii=False, indent=2))
            return 0
        result = run(request, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
