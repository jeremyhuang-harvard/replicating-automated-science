import unittest

from parrot_baseline.analysis import LIMITATIONS, summarize
from parrot_baseline.schemas import Concept, Persona, Question, Response, StudyManifest

ATT = {"novelty_seeking": 3}
QS = [
    Question("appeal", "Appeal?", "rating"),
    Question("intent", "Intent?", "rating"),
    Question("why", "Why?", "text"),
]


def manifest(personas):
    return StudyManifest("s1", "2025-01-01T00:00:00+00:00", "demo", "m", 1, Concept("T", "D"), personas, QS)


def persona(pid, seg):
    return Persona(pid, pid, seg, "desc", ATT)


def resp(pid, seg, appeal, intent, why="because"):
    return Response(pid, seg, {"appeal": appeal, "intent": intent, "why": why}, "{}")


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.personas = [persona("a1", "A"), persona("a2", "A"), persona("b1", "B"), persona("b2", "B")]
        self.responses = [
            resp("a1", "A", 5, 4, "love it"), resp("a2", "A", 3, 2, "meh"),
            resp("b1", "B", 1, 1, "no"), resp("b2", "B", 2, 1, "nope"),
        ]
        self.s = summarize(manifest(self.personas), self.responses)

    def test_means_overall_and_by_segment(self):
        self.assertEqual(self.s["means"]["overall"], {"appeal": 2.75, "intent": 2.0})
        self.assertEqual(self.s["means"]["by_segment"]["A"], {"appeal": 4.0, "intent": 3.0})
        self.assertEqual(self.s["means"]["by_segment"]["B"], {"appeal": 1.5, "intent": 1.0})

    def test_rating_distributions_cover_all_scale_points(self):
        d = self.s["distributions"]["overall"]["appeal"]
        self.assertEqual(d, {"1": 1, "2": 1, "3": 1, "4": 0, "5": 1})
        self.assertEqual(self.s["distributions"]["by_segment"]["A"]["intent"], {"1": 0, "2": 1, "3": 0, "4": 1, "5": 0})

    def test_counts(self):
        self.assertEqual(self.s["counts"], {"panel": 4, "ok": 4, "failed": 0, "by_segment": {"A": 2, "B": 2}})

    def test_text_questions_not_averaged(self):
        self.assertNotIn("why", self.s["means"]["overall"])

    def test_representative_quote_is_closest_to_segment_mean(self):
        quotes = self.s["quotes"]["A"]["why"]
        self.assertEqual(quotes["persona_id"], "a1")  # appeal 5 vs 3, mean 4: tie -> lowest id
        self.assertEqual(quotes["text"], "love it")

    def test_failed_responses_excluded_and_warned(self):
        responses = self.responses[:3] + [Response("b2", "B", {}, "garbage", error="no JSON")]
        s = summarize(manifest(self.personas), responses)
        self.assertEqual(s["counts"]["failed"], 1)
        self.assertEqual(s["means"]["by_segment"]["B"]["appeal"], 1.0)
        self.assertTrue(any("b2" in w and "no JSON" in w for w in s["warnings"]))

    def test_segment_with_no_valid_responses(self):
        responses = self.responses[:2] + [Response("b1", "B", {}, "x", error="bad"), Response("b2", "B", {}, "x", error="bad")]
        s = summarize(manifest(self.personas), responses)
        self.assertNotIn("B", s["means"]["by_segment"])
        self.assertTrue(any("B" in w and "no valid" in w for w in s["warnings"]))

    def test_small_segments_and_limitations_always_reported(self):
        self.assertTrue(any("small" in w.lower() for w in self.s["warnings"]))
        self.assertEqual(self.s["limitations"], LIMITATIONS)
        self.assertTrue(any("human" in l.lower() for l in LIMITATIONS))

    def test_missing_persona_response_counted_as_failed(self):
        s = summarize(manifest(self.personas), self.responses[:3])
        self.assertEqual(s["counts"]["panel"], s["counts"]["ok"] + s["counts"]["failed"])
        self.assertEqual(s["counts"]["failed"], 1)
        self.assertTrue(any("b2" in w and "missing" in w.lower() for w in s["warnings"]))

    def test_duplicate_unknown_and_spoofed_segments_are_reconciled_to_manifest(self):
        responses = [
            resp("a1", "B", 5, 5),  # segment must come from the manifest, not the response
            resp("a2", "A", 1, 1),
            resp("a2", "A", 5, 5),  # duplicate makes a2 unusable
            resp("b1", "A", 2, 2),  # spoofed segment must still aggregate under B
            resp("unknown", "A", 5, 5),
        ]
        s = summarize(manifest(self.personas), responses)
        self.assertEqual(s["counts"], {"panel": 4, "ok": 2, "failed": 2, "by_segment": {"A": 1, "B": 1}})
        self.assertEqual(s["means"]["by_segment"]["A"]["appeal"], 5.0)
        self.assertEqual(s["means"]["by_segment"]["B"]["appeal"], 2.0)
        self.assertTrue(any("Duplicate" in w and "a2" in w for w in s["warnings"]))
        self.assertTrue(any("Unknown" in w and "unknown" in w for w in s["warnings"]))
        self.assertTrue(any("missing" in w.lower() and "b2" in w for w in s["warnings"]))


if __name__ == "__main__":
    unittest.main()
