import json
import unittest

from src.context_segmenter import DeterministicSegmenter, LLMBasedSegmenter


class FakeResponse:
    def __init__(self, output_json=None, output_text=None):
        self._output_json = output_json
        if output_text is None and output_json is not None and not isinstance(output_json, Exception):
            self.output_text = json.dumps(output_json)
        else:
            self.output_text = output_text

    @property
    def output_json(self):
        if isinstance(self._output_json, Exception):
            raise self._output_json
        if self._output_json is None:
            raise AttributeError("no output_json")
        return self._output_json


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

        class Responses:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kwargs):
                self.outer.calls.append(kwargs)
                return self.outer.response

        self.responses = Responses(self)


class TestDeterministicSegmenter(unittest.TestCase):
    def test_segment_splits_and_limits(self):
        seg = DeterministicSegmenter(max_segments=2)
        query = "turn on lights and lock doors, then set alarm"
        parts = seg.segment(query)
        self.assertEqual(parts, ["turn on lights", "lock doors"])

    def test_segment_handles_empty(self):
        seg = DeterministicSegmenter()
        self.assertEqual(seg.segment(""), [])


class TestLLMBasedSegmenter(unittest.TestCase):
    def test_segment_raises_without_client(self):
        seg = LLMBasedSegmenter()
        seg.client = None

        with self.assertRaises(RuntimeError):
            seg.segment("anything")

    def test_segment_uses_output_json_when_available(self):
        response = FakeResponse(output_json={"segments": ["a", "b"]})
        client = FakeClient(response)
        seg = LLMBasedSegmenter()
        seg.client = client

        segments = seg.segment("do things")
        self.assertEqual(segments, ["a", "b"])
        self.assertEqual(client.calls[0]["input"].splitlines()[-1], "do things")

    def test_segment_falls_back_to_output_text(self):
        response = FakeResponse(output_json=ValueError("no json"), output_text='{"segments": ["x"]}')
        client = FakeClient(response)
        seg = LLMBasedSegmenter()
        seg.client = client

        segments = seg.segment("another task")
        self.assertEqual(segments, ["x"])


if __name__ == "__main__":
    unittest.main()
