from pathlib import Path
import unittest


class AssessmentAgentArchitectureDiagramTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diagram_path = Path("docs/assets/assessment_agent_architecture.jsx")
        self.source = self.diagram_path.read_text()

    def test_component_exists_with_expected_sections(self) -> None:
        self.assertTrue(self.diagram_path.exists())
        self.assertIn("export default function AssessmentAgentArchitectureDiagram()", self.source)
        for label in [
            "Assessment Agent Architecture",
            "Input Record",
            "Predict Loss",
            "Hybrid Controller",
            "Fallback Retrieval",
            "Reviewer Response",
            "Confidence Policy",
            "Guardrails",
            ]:
            with self.subTest(label=label):
                self.assertIn(label, self.source)

    def test_html_wrapper_exists_for_browser_preview(self) -> None:
        html_path = Path("docs/assets/assessment_agent_architecture.html")
        self.assertTrue(html_path.exists())
        html = html_path.read_text()
        self.assertIn("Assessment Agent Architecture", html)
        self.assertIn("Open this file directly in a browser", html)
        for label in [
            "Input Record",
            "Predict Loss",
            "Hybrid Controller",
            "Fallback Retrieval",
            "Reviewer Response",
        ]:
            with self.subTest(label=label):
                self.assertIn(label, html)


if __name__ == "__main__":
    unittest.main()
