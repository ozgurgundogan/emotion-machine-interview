import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import utils


class TestUtils(unittest.TestCase):
    def test_normalize_parameters_handles_dict(self):
        params = {
            "required": [{"name": "id", "type": "string"}],
            "optional": [{"name": "limit"}],
        }
        out = utils.normalize_parameters(params)
        self.assertEqual(out["required"], params["required"])
        self.assertEqual(out["optional"], params["optional"])

    def test_normalize_parameters_handles_list(self):
        params = [{"name": "id"}, {"name": "page"}]
        out = utils.normalize_parameters(params)
        self.assertEqual(out["required"], params)
        self.assertEqual(out["optional"], [])

    def test_hash_dict_is_deterministic(self):
        payload = {"a": 1, "b": 2}
        h1 = utils.hash_dict(payload)
        h2 = utils.hash_dict(json.loads(json.dumps(payload)))
        self.assertEqual(h1, h2)

    def test_read_records_supports_json_array(self):
        data = [{"x": 1}, {"y": 2}]
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            records = list(utils.read_records(path))
            self.assertEqual(records, data)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_read_records_supports_jsonl(self):
        recs = [{"a": 1}, {"b": 2}]
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
            f.write("\n")  # blank line should be ignored
            path = f.name

        try:
            records = list(utils.read_records(path))
            self.assertEqual(records, recs)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_functions_from_record_handles_functions_key(self):
        rec = {"Functions": ["{'name': 'fn', 'api_name': 'do.it', 'parameters': {'required': []}}", "invalid"]}
        functions = list(utils._extract_functions_from_record(rec))
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "fn")

    def test_generate_function_as_text_formats_fields(self):
        fn = {"name": "fn", "api_name": "do.it", "description": "desc", "parameters": {"required": [{"name": "x"}]}}
        text = utils.generate_function_as_text(fn)
        self.assertIn("fn::do.it", text)
        self.assertIn("desc", text)
        self.assertIn('"required": [{"name": "x"}]', text)

    def test_load_functions_uses_dataset_paths(self):
        fn_from_functions = {
            "name": "alpha",
            "api_call": "alpha.do",
            "parameters": {"required": [{"name": "foo"}]},
            "description": "Alpha function",
        }
        fn_from_function_key = {
            "name": "beta",
            "api_name": "beta.call",
            "parameters": {"required": [{"name": "id"}]},
            "description": "Beta function",
        }
        records = [
            {"Functions": [str(fn_from_functions)]},
            {"function": fn_from_function_key},
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            path = f.name

        alpha_hash = utils.hash_dict(fn_from_functions)
        beta_hash = utils.hash_dict(fn_from_function_key)
        try:
            original_paths = utils.DATASET_PATHS
            utils.DATASET_PATHS = [path]
            funcs = utils.load_functions()
            self.assertIn(alpha_hash, funcs)
            self.assertIn(beta_hash, funcs)
            self.assertEqual(funcs[alpha_hash]["api_name"], "alpha.do")
            self.assertEqual(funcs[beta_hash]["parameters"]["required"], [{"name": "id"}])
        finally:
            utils.DATASET_PATHS = original_paths
            Path(path).unlink(missing_ok=True)

    def test_normalize_vectors(self):
        vecs = np.array([[3.0, 4.0], [0.0, 5.0]])
        normed = utils.normalize(vecs)
        # First vector magnitude is 5; second is 5
        np.testing.assert_allclose(normed[0], [0.6, 0.8])
        np.testing.assert_allclose(normed[1], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
