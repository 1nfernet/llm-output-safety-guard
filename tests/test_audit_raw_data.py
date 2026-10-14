"""阶段 2 的小型工程测试：只用合成文本，不接触真实 XGuard 原文。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "audit_raw_data.py"


class AuditRawDataTest(unittest.TestCase):
    def run_audit(self, lines: list[str]) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.jsonl"
            report = Path(temp_dir) / "report.json"
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(report)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                check=False,
            )
            report_text = report.read_text(encoding="utf-8") if report.exists() else ""
            return result, report_text

    def test_reports_counts_and_lengths_without_copying_text(self) -> None:
        rows = [
            {"id": "1", "sample_type": "general", "stage": "r", "label": "sec", "response": "中文回复"},
            {"id": "2", "sample_type": "general", "stage": "r", "label": "ha", "response": "hello"},
            {"id": "3", "sample_type": "general", "stage": "q", "label": "sec", "prompt": "DONOTSHARE", "response": None},
        ]
        lines = [*(json.dumps(row, ensure_ascii=False) for row in rows), "{invalid", "", "[]"]
        result, report_text = self.run_audit(lines)

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_text)
        self.assertEqual(report["rows"], {
            "total_lines": 6,
            "valid_objects": 3,
            "blank_lines": 1,
            "malformed_json": 1,
            "non_object_json": 1,
        })
        self.assertEqual(report["sample_type"], {"general": 3})
        self.assertEqual(report["stage"], {"q": 1, "r": 2})
        self.assertEqual(report["label"], {"ha": 1, "sec": 2})
        self.assertEqual(report["target_reply_subset"]["count"], 2)
        self.assertEqual(report["target_reply_subset"]["label"], {"ha": 1, "sec": 1})
        self.assertEqual(report["target_reply_subset"]["response_characters"], {
            "min": 4, "p50": 4, "p90": 5, "p99": 5, "max": 5,
        })
        self.assertEqual(report["target_reply_subset"]["script_signals"]["has_han"], 1)
        self.assertEqual(report["target_reply_subset"]["script_signals"]["has_latin"], 1)
        self.assertEqual(report["target_reply_subset"].get("han_candidate_label"), {"sec": 1})
        self.assertNotIn("中文回复", report_text + result.stdout)
        self.assertNotIn("DONOTSHARE", report_text + result.stdout)
        self.assertNotIn("hello", report_text + result.stdout)

    def test_unknown_values_and_field_names_are_not_echoed(self) -> None:
        row = {
            "sample_type": "general", "stage": "r", "label": "SECRET_CATEGORY",
            "response": "PRIVATE_REPLY", "SECRET_FIELD": "PRIVATE_VALUE",
        }
        result, report_text = self.run_audit([json.dumps(row)])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(report_text)["target_reply_subset"]["label"], {"<other>": 1})
        self.assertNotIn("SECRET", report_text + result.stdout)
        self.assertNotIn("PRIVATE", report_text + result.stdout)


if __name__ == "__main__":
    unittest.main()
