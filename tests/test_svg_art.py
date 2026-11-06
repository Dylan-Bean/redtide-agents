import unittest
from unittest.mock import patch

from redtide.llm import LLMResponse
from redtide.svg_art import SVGGenerationError, generate_report_svg, sanitize_svg


class SanitizeSVGTests(unittest.TestCase):
    def test_adds_svg_namespace_and_renders_basic_markup(self):
        svg = sanitize_svg(
            '<svg viewBox="0 0 800 450"><text x="10" y="20">Red tide</text></svg>'
        )

        self.assertIn("<svg", svg)
        self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)
        self.assertIn("Red tide", svg)

    def test_removes_unsafe_attributes(self):
        svg = sanitize_svg(
            '<svg onload="alert(1)"><rect x="0" y="0" width="10" height="10" '
            'onclick="alert(1)" fill="#ff0000"/></svg>'
        )

        self.assertNotIn("onload", svg)
        self.assertNotIn("onclick", svg)
        self.assertIn('fill="#ff0000"', svg)

    def test_rejects_non_svg_and_active_elements(self):
        for source in ("<html/>", '<svg><script>alert(1)</script></svg>'):
            with self.subTest(source=source):
                with self.assertRaises(SVGGenerationError):
                    sanitize_svg(source)

    def test_rejects_external_resource_references(self):
        with self.assertRaisesRegex(SVGGenerationError, "外部资源引用"):
            sanitize_svg('<svg><rect fill="url(https://example.com/a.svg)"/></svg>')

    def test_rejects_entity_declarations(self):
        with self.assertRaisesRegex(SVGGenerationError, "ENTITY"):
            sanitize_svg('<!DOCTYPE svg [<!ENTITY x "unsafe">]><svg>&x;</svg>')


class GenerateReportSVGTests(unittest.TestCase):
    def test_uses_configured_chat_model_and_sanitizes_response(self):
        with patch(
            "redtide.svg_art.llm.chat",
            return_value=LLMResponse(content='```svg\n<svg><text>Report</text></svg>\n```'),
        ) as chat:
            svg = generate_report_svg("Why?", "A short report.")

        self.assertIn("Report", svg)
        self.assertEqual(chat.call_args.kwargs["temperature"], 0.2)
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 4000)


if __name__ == "__main__":
    unittest.main()
