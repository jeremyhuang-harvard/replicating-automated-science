import unittest

from parrot_baseline.schemas import (
    Concept,
    Persona,
    Question,
    Response,
    SchemaError,
    StudyManifest,
)


def make_persona(**overrides):
    fields = dict(
        persona_id="p01",
        name="Test Persona",
        segment="adventurous_foodie",
        description="Likes trying unusual flavors.",
        attitudes={"novelty_seeking": 5, "brand_loyalty": 2, "health_concern": 2, "price_sensitivity": 2},
    )
    fields.update(overrides)
    return Persona(**fields)


class ConceptTests(unittest.TestCase):
    def test_valid_concept_round_trips(self):
        concept = Concept(title="Lasagna Cola", description="A savory cola.")
        self.assertEqual(Concept.from_dict(concept.to_dict()), concept)

    def test_blank_title_or_description_rejected(self):
        with self.assertRaises(SchemaError):
            Concept(title="  ", description="x")
        with self.assertRaises(SchemaError):
            Concept(title="x", description="")

    def test_non_string_rejected(self):
        with self.assertRaises(SchemaError):
            Concept(title=3, description="x")


class PersonaTests(unittest.TestCase):
    def test_round_trip(self):
        persona = make_persona()
        self.assertEqual(Persona.from_dict(persona.to_dict()), persona)

    def test_attitude_out_of_range_rejected(self):
        with self.assertRaises(SchemaError):
            make_persona(attitudes={"novelty_seeking": 6})

    def test_blank_id_rejected(self):
        with self.assertRaises(SchemaError):
            make_persona(persona_id="")


class QuestionTests(unittest.TestCase):
    def test_kinds(self):
        Question(question_id="appeal", text="How appealing?", kind="rating")
        Question(question_id="why", text="Why?", kind="text")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(SchemaError):
            Question(question_id="q", text="?", kind="choice")

    def test_round_trip(self):
        q = Question(question_id="appeal", text="How appealing?", kind="rating")
        self.assertEqual(Question.from_dict(q.to_dict()), q)


class ResponseTests(unittest.TestCase):
    def test_round_trip_and_ok_flag(self):
        r = Response(persona_id="p01", segment="s", answers={"appeal": 4, "why": "yum"}, raw_output="{}")
        self.assertTrue(r.ok)
        self.assertEqual(Response.from_dict(r.to_dict()), r)

    def test_error_response_not_ok(self):
        r = Response(persona_id="p01", segment="s", answers={}, raw_output="junk", error="no JSON")
        self.assertFalse(r.ok)


class StudyManifestTests(unittest.TestCase):
    def make(self, **overrides):
        fields = dict(
            study_id="s1",
            created_at="2025-01-01T00:00:00+00:00",
            engine="demo",
            model="demo-rules-v1",
            seed=1,
            concept=Concept(title="T", description="D"),
            panel=[make_persona()],
            questions=[Question(question_id="appeal", text="?", kind="rating")],
        )
        fields.update(overrides)
        return StudyManifest(**fields)

    def test_round_trip(self):
        m = self.make()
        self.assertEqual(StudyManifest.from_dict(m.to_dict()), m)

    def test_timestamp_must_be_utc_iso(self):
        with self.assertRaises(SchemaError):
            self.make(created_at="yesterday")
        with self.assertRaises(SchemaError):
            self.make(created_at="2025-01-01T00:00:00")  # naive

    def test_empty_panel_rejected(self):
        with self.assertRaises(SchemaError):
            self.make(panel=[])

    def test_duplicate_persona_or_question_ids_rejected(self):
        with self.assertRaises(SchemaError):
            self.make(panel=[make_persona(), make_persona()])
        q = Question(question_id="a", text="?", kind="rating")
        with self.assertRaises(SchemaError):
            self.make(questions=[q, q])


if __name__ == "__main__":
    unittest.main()
