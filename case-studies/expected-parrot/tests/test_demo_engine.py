import unittest
from statistics import mean

from parrot_baseline.engines import DemoEngine
from parrot_baseline.panel import default_panel
from parrot_baseline.parsing import parse_answers
from parrot_baseline.prompts import build_persona_prompt
from parrot_baseline.schemas import Concept
from parrot_baseline.survey import default_questions

QS = default_questions()
NOVEL = Concept("Lasagna Cola", "A zero-sugar cola with tomato, basil and parmesan notes. A bold, savory, limited-edition twist on soda.")
FAMILIAR = Concept("Classic Cherry Cola", "A classic cherry cola in the familiar red can, sweet and refreshing, same great taste as always.")


def run(engine, concept, seed=1, size=12):
    out = []
    for p in default_panel(size):
        raw = engine.generate(concept, p, QS, build_persona_prompt(concept, p, QS), seed)
        out.append((p, parse_answers(raw, QS)))
    return out


def seg_mean(results, segment, qid):
    return mean(a[qid] for p, a in results if p.segment == segment)


class DemoEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DemoEngine()

    def test_identity(self):
        self.assertEqual(self.engine.name, "demo")
        self.assertTrue(self.engine.model)

    def test_output_parses_against_survey(self):
        results = run(self.engine, NOVEL)
        self.assertEqual(len(results), 12)

    def test_deterministic_for_same_seed(self):
        self.assertEqual(run(self.engine, NOVEL, seed=5), run(DemoEngine(), NOVEL, seed=5))

    def test_seed_changes_output(self):
        self.assertNotEqual(run(self.engine, NOVEL, seed=5), run(self.engine, NOVEL, seed=6))

    def test_segments_differ_on_novel_concept(self):
        r = run(self.engine, NOVEL)
        self.assertGreater(seg_mean(r, "adventurous_foodie", "appeal") - seg_mean(r, "mainstream_drinker", "appeal"), 0.75)
        self.assertGreater(seg_mean(r, "adventurous_foodie", "purchase_intent"), seg_mean(r, "skeptical_health_conscious", "purchase_intent"))
        self.assertGreater(seg_mean(r, "adventurous_foodie", "appeal"), seg_mean(r, "skeptical_health_conscious", "appeal"))

    def test_novel_concept_rated_more_unique_than_familiar(self):
        novel, familiar = run(self.engine, NOVEL), run(self.engine, FAMILIAR)
        overall = lambda r: mean(a["uniqueness"] for _, a in r)
        self.assertGreater(overall(novel), overall(familiar) + 1)

    def test_open_text_is_concept_specific_and_segment_specific(self):
        r = dict((p.persona_id, a) for p, a in run(self.engine, NOVEL))
        self.assertIn("Lasagna Cola", r["af-01"]["first_reaction"])
        self.assertNotEqual(r["af-01"]["first_reaction"], r["sh-01"]["first_reaction"])
        self.assertNotEqual(r["af-01"]["concerns"], r["md-01"]["concerns"])

    def test_larger_panel_still_has_noise_between_copies(self):
        r = run(self.engine, NOVEL, size=48)
        texts = {a["first_reaction"] for _, a in r}
        self.assertGreater(len(texts), 12)


if __name__ == "__main__":
    unittest.main()
