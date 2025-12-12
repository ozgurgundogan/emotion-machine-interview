import json
import unittest

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

    def test_normalize_vectors(self):
        vecs = np.array([[3.0, 4.0], [0.0, 5.0]])
        normed = utils.normalize(vecs)
        # First vector magnitude is 5; second is 5
        np.testing.assert_allclose(normed[0], [0.6, 0.8])
        np.testing.assert_allclose(normed[1], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
