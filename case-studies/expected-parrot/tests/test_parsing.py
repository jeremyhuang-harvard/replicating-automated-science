import unittest

from parrot_baseline.parsing import ParseError, extract_json_object, parse_answers
from parrot_baseline.survey import default_questions

QS = default_questions()
GOOD = {
    "appeal": 4, "clarity": 5, "uniqueness": 3, "purchase_intent": 2,
    "first_reaction": "Curious.", "concerns": "Salt.", "improvement": "Less cheese.",
}


def dumps(d):
    import json
    return json.dumps(d)


class ExtractJsonTests(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_markdown_fence_and_chatter(self):
        raw = 'Sure! Here you go:\n```json\n{"a": 1, "b": "x"}\n```\nHope that helps.'
        self.assertEqual(extract_json_object(raw), {"a": 1, "b": "x"})

    def test_mlx_banner_noise(self):
        raw = '==========\n{"a": 2}\n==========\nPrompt: 300 tokens, 1200.0 tokens-per-sec\nPeak memory: 5 GB'
        self.assertEqual(extract_json_object(raw), {"a": 2})

    def test_think_block_is_ignored_even_with_braces(self):
        raw = '<think>maybe {"a": 99} hmm</think>\n{"a": 1}'
        self.assertEqual(extract_json_object(raw), {"a": 1})

    def test_braces_inside_strings(self):
        self.assertEqual(extract_json_object('x {"a": "} {"} y'), {"a": "} {"})

    def test_trailing_comma_repaired(self):
        self.assertEqual(extract_json_object('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_prefers_candidate_matching_expected_keys(self):
        raw = '{"unrelated": 1}\n' + dumps(GOOD) + '\n{"other": 2}'
        self.assertEqual(extract_json_object(raw, expected_keys=list(GOOD)), GOOD)

    def test_no_json_raises_with_snippet(self):
        with self.assertRaises(ParseError) as ctx:
            extract_json_object("I cannot comply with that request.")
        self.assertIn("I cannot comply", str(ctx.exception))

    def test_truncated_json_raises(self):
        with self.assertRaises(ParseError) as ctx:
            extract_json_object('{"appeal": 4, "clarity": ')
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_empty_output(self):
        with self.assertRaises(ParseError):
            extract_json_object("   ")


class ParseAnswersTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_answers(dumps(GOOD), QS), GOOD)

    def test_nested_under_answers_key(self):
        self.assertEqual(parse_answers(dumps({"answers": GOOD}), QS), GOOD)

    def test_coerces_numeric_strings_and_integral_floats(self):
        raw = dumps({**GOOD, "appeal": "4", "clarity": 5.0})
        answers = parse_answers(raw, QS)
        self.assertEqual((answers["appeal"], answers["clarity"]), (4, 5))

    def test_missing_keys_listed(self):
        bad = {k: v for k, v in GOOD.items() if k not in ("concerns", "appeal")}
        with self.assertRaises(ParseError) as ctx:
            parse_answers(dumps(bad), QS)
        self.assertIn("appeal", str(ctx.exception))
        self.assertIn("concerns", str(ctx.exception))

    def test_out_of_range_and_wrong_types_rejected(self):
        for field, value in [("appeal", 6), ("appeal", 0), ("appeal", 3.5), ("appeal", True), ("appeal", "high"), ("concerns", 7), ("concerns", "  ")]:
            with self.assertRaises(ParseError, msg=f"{field}={value!r}"):
                parse_answers(dumps({**GOOD, field: value}), QS)

    def test_extra_keys_dropped(self):
        answers = parse_answers(dumps({**GOOD, "bonus": "x"}), QS)
        self.assertNotIn("bonus", answers)

    def test_text_is_stripped_and_bounded(self):
        answers = parse_answers(dumps({**GOOD, "concerns": "  " + "x" * 5000 + " "}), QS)
        self.assertLessEqual(len(answers["concerns"]), 1000)


if __name__ == "__main__":
    unittest.main()
