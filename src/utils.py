import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.environment import DATASET_PATHS


def read_records(path):
    p = Path(path)
    text = p.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        for rec in data:
            yield rec
    else:
        with p.open("r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def hash_dict(d):
    encoded = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _iter_records(path):
    yield from read_records(path)



def _extract_functions_from_record(rec):
    if "Functions" in rec:
        for fn_str in rec.get("Functions", []):
            try:
                yield ast.literal_eval(fn_str)
            except Exception:
                continue
    elif "function" in rec:
        fn = rec.get("function") or {}
        yield fn


def load_functions():
    funcs = defaultdict(dict)
    for path in DATASET_PATHS:
        for record in _iter_records(path):
            for fn in _extract_functions_from_record(record):
                hashed_value = hash_dict(fn)
                funcs[hashed_value]={
                    "name": fn.get("name", ""),
                    "api_name": fn.get("api_name") or fn.get("api_call") or "",
                    "parameters": normalize_parameters(fn.get("parameters")),
                    "description": fn.get("description", ""),
                }

    return funcs


def normalize_parameters(params):
    required = []
    optional = []

    if isinstance(params, dict):
        if "required" in params or "optional" in params:
            required = params.get("required", []) or []
            optional = params.get("optional", []) or []
        else:
            for name, info in params.items():
                entry = {"name": name}
                if isinstance(info, dict):
                    entry.update({k: v for k, v in info.items() if k in ("description", "type")})
                required.append(entry)
    elif isinstance(params, list):
        for item in params:
            if isinstance(item, dict):
                required.append(item)
    return {"required": required, "optional": optional}


def generate_function_as_text(fn):
    name = fn.get("name", "")
    api_name = fn.get("api_name", "") or fn.get("api_call", "")
    description = fn.get("description", "")
    return f"{name}::{api_name}\n{description}\nparams: {json.dumps(fn.get('parameters', ''))}"


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def load_llm_response_as_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()

    # Remove Markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    return json.loads(text)