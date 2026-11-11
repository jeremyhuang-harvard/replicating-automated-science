import re
import unittest

from parrot_baseline.analysis import summarize
from parrot_baseline.report import render_report
from parrot_baseline.schemas import Concept, Persona, Question, Response, StudyManifest

XSS = '<script>alert("x")</script>'
QS = [
    Question("appeal", "Appeal?", "rating"),
    Question("why", "Why?", "text"),
]


def build(title=XSS, quote=XSS):
    personas = [
        Persona("a1", "Persona <b>A1</b>", "adventurous_foodie", "d", {"novelty_seeking": 5}),
        Persona("m1", "M1", "mainstream_drinker", "d", {"novelty_seeking": 2}),
    ]
    manifest = StudyManifest(
        "study-1", "2025-03-04T05:06:07+00:00", "demo", "demo-rules-v1", 42,
        Concept(title, "Desc & <i>more</i>"), personas, QS,
    )
    responses = [
        Response("a1", "adventurous_foodie", {"appeal": 5, "why": quote}, "{}"),
        Response("m1", "mainstream_drinker", {"appeal": 2, "why": "plain"}, "{}"),
    ]
    return manifest, responses, summarize(manifest, responses)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.responses, self.summary = build()
        self.html = render_report(self.manifest, self.responses, self.summary)

    def test_user_content_is_escaped_everywhere(self):
        self.assertNotIn("<script", self.html.lower())
        self.assertNotIn("<b>A1</b>", self.html)
        self.assertNotIn("<i>more</i>", self.html)
        self.assertIn("&lt;script&gt;", self.html)

    def test_prominent_simulation_label(self):
        self.assertIn("Simulated hypotheses, not human evidence", self.html)
        head = self.html[: self.html.index("</header>")]
        self.assertIn("not human evidence", head.lower())

    def test_self_contained_no_external_resources_or_scripts(self):
        self.assertNotRegex(self.html, r"https?://")
        self.assertNotRegex(self.html, r"(?i)<script|<link|<img|@import|src=")
        self.assertTrue(self.html.startswith("<!doctype html>"))

    def test_includes_metadata_charts_quotes_methodology(self):
        for needle in ("demo-rules-v1", "2025-03-04T05:06:07+00:00", "42", "adventurous_foodie", "mainstream_drinker",
                       "Methodology", "Limitations", "Representative quotes", "plain"):
            self.assertIn(needle, self.html)
        self.assertIn("5.00", self.html)  # mean for foodie appeal
        self.assertIn("2.00", self.html)
        self.assertRegex(self.html, r'style="width:\s*100(\.0+)?%')  # CSS bar for mean 5/5 ... or dist segment
        self.assertIn('class="bar', self.html)

    def test_warnings_rendered_and_escaped(self):
        failed = self.responses[:1] + [Response("m1", "mainstream_drinker", {}, "x", error="bad <b>json</b>")]
        html = render_report(self.manifest, failed, summarize(self.manifest, failed))
        self.assertIn("bad &lt;b&gt;json&lt;/b&gt;", html)
        self.assertNotIn("<b>json</b>", html)

    def test_bar_widths_within_bounds(self):
        for width in re.findall(r"width:\s*([\d.]+)%", self.html):
            self.assertLessEqual(float(width), 100.0)


if __name__ == "__main__":
    unittest.main()
