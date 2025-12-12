import ast
import hashlib
import json
from pathlib import Path

from src.indexer import Indexer
from src.environment import INDEX_PATH, METADATA_PATH, MISMATCH_PATH
from src.utils import read_records


DATASETS = [
    ("manual_easy",   Path("../data/gorilla_openfunctions_v1_manual_test.json")),
    ("manual_medium", Path("../data/gorilla_openfunctions_v1_manual_test_medium.json")),
    ("manual_hard",   Path("../data/gorilla_openfunctions_v1_manual_test_hard.json")),
    ("manual_cross",  Path("../data/gorilla_openfunctions_v1_manual_test_cross.json")),
]



def load_dataset(path, limit):
    rows = list(read_records(path))
    return rows if limit is None else rows[:limit]


def normalize_parameters(params):
    required, optional = [], []

    if isinstance(params, dict):
        if "required" in params or "optional" in params:
            required = params.get("required", []) or []
            optional = params.get("optional", []) or []
        else:
            for name, info in params.items():
                entry = {"name": name}
                if isinstance(info, dict):
                    entry.update({k: v for k, v in info.items()
                                  if k in ("description", "type")})
                required.append(entry)

    elif isinstance(params, list):
        required = [p for p in params if isinstance(p, dict)]

    return {"required": required, "optional": optional}


def make_tool_id(fn):
    tool_api = fn.get("api_name") or fn.get("api_call") or ""
    norm_params = normalize_parameters(fn.get("parameters", {}))
    param_bytes = json.dumps(norm_params, sort_keys=True).encode("utf-8")
    params_hash = hashlib.sha1(param_bytes).hexdigest()[:8]
    return f"{fn.get('name', '')}::{tool_api}::v{params_hash}"


def compare_params_deeply(params1, params2):
    try:
        json1 = json.dumps(params1, sort_keys=True)
        json2 = json.dumps(params2, sort_keys=True)
        return json1 == json2
    except TypeError:
        return False


def extract_expected_function(rec):
    fn = rec.get("function")
    if fn:
        return fn

    funcs = rec.get("Functions") or rec.get("functions") or []
    for entry in funcs:
        if isinstance(entry, dict):
            return entry
        if isinstance(entry, str):
            try:
                return ast.literal_eval(entry)
            except Exception:
                continue
    return {}


def find_matching_api(target_api, hits):
    target_tool_id = make_tool_id(target_api)

    target_api_call = (target_api.get("api_call") or "").lower()
    target_name = (target_api.get("name") or "").lower()
    target_desc = (target_api.get("description") or "").lower()
    target_params = normalize_parameters(target_api.get("parameters", {}))

    for api in hits:
        if api.get("tool_id") == target_tool_id:
            return True

        api_call = (api.get("api_name") or "").lower()
        name_avail = (api.get("name") or "").lower()
        desc_avail = (api.get("description") or "").lower()
        params_avail = api.get("parameters", {})

        name_match = (
            api_call == target_api_call
            or name_avail == target_name
            or target_name in desc_avail
            or target_desc in desc_avail
        )

        if name_match and compare_params_deeply(target_params, params_avail):
            return True

    return False


def evaluate_dataset(label, path, sample_size, k, indexer):
    if not path.exists():
        print(f"[{label}] skipping (dataset not found: {path})")
        return None

    rows = load_dataset(path, sample_size)
    total = hit = 0
    misses = []

    for rec in rows:
        query = rec.get("Instruction") or rec.get("question") or ""
        expected = extract_expected_function(rec)
        if not expected:
            continue

        total += 1

        hits = indexer.search(query)
        if find_matching_api(expected, hits):
            hit += 1
        else:
            misses.append({
                "query": query,
                "expected": expected,
                "topk": hits,
            })

    recall = hit / total if total else 0.0
    print(f"[{label}] recall@{k}: {recall:.3f} ({hit}/{total})")

    if misses:
        base = Path(MISMATCH_PATH)
        mismatch_path = base.with_name(f"{base.stem}_{label}{base.suffix}")
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        with mismatch_path.open("w") as f:
            for m in misses:
                f.write(json.dumps(m, ensure_ascii=False, indent=4) + "\n")
        print(f"[{label}] {len(misses)} mismatches written → {mismatch_path}")

    return {
        "label": label,
        "total": total,
        "hit": hit,
        "recall": recall,
        "misses": len(misses),
    }


def evaluate_all(sample_size=100, k=5):
    indexer = Indexer(INDEX_PATH, METADATA_PATH)
    indexer.load()

    results = []
    for label, path in DATASETS:
        res = evaluate_dataset(label, path, sample_size, k, indexer)
        if res:
            results.append(res)

    return results


if __name__ == "__main__":
    evaluate_all(sample_size=100, k=5)
