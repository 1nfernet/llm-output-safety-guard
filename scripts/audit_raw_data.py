"""阶段 2：逐行审计本地 XGuard JSONL，只输出汇总统计，不复制样本正文。"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


# 只统计这些公开的结构字段。未知字段仅计数，避免把意外出现的敏感字段名写进报告。
FIELDS = ("id", "sample_type", "prompt", "response", "stage", "policy", "label", "explanation")
SAMPLE_TYPES = frozenset(("general", "dynamic_policy"))
STAGES = frozenset(("q", "r", "qr"))
# 来源：XGuard 同版本 examples/prompt_template.txt。未知标签只归入 <other>，不回显原值。
LABELS = frozenset(
    ("sec", "pc", "dc", "dw", "pi", "ec", "ac", "def", "ti", "cy", "ph", "mh",
     "se", "sci", "pp", "cs", "acc", "mc", "ha", "ps", "ter", "sd", "ext",
     "fin", "med", "law", "cm", "ma", "md")
)
SCRIPT_SIGNALS = (
    "has_han", "has_kana", "has_hangul", "has_latin", "has_cyrillic",
    "has_arabic", "has_thai", "has_devanagari", "has_other_letters",
    "han_without_kana_or_hangul",
)


def safe_bucket(record: dict, field: str, allowed: frozenset[str]) -> str:
    """已知枚举保留原值，其余只给固定标记，防止自由文本进入统计键。"""
    if field not in record:
        return "<missing>"
    value = record[field]
    if value is None:
        return "<null>"
    if isinstance(value, str) and not value.strip():
        return "<blank>"
    if isinstance(value, str) and value in allowed:
        return value
    return "<other>"


def script_signals(text: str) -> set[str]:
    """用 Unicode 字符范围观察文字混杂；这是文字系统信号，不是可靠的语言识别。"""
    found: set[str] = set()
    for char in text:
        code = ord(char)
        if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0x20000 <= code <= 0x2A6DF:
            found.add("has_han")
        elif 0x3040 <= code <= 0x30FF:
            found.add("has_kana")
        elif 0xAC00 <= code <= 0xD7AF:
            found.add("has_hangul")
        elif (0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A
              or 0x00C0 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF):
            found.add("has_latin")
        elif 0x0400 <= code <= 0x052F:
            found.add("has_cyrillic")
        elif 0x0600 <= code <= 0x06FF:
            found.add("has_arabic")
        elif 0x0E00 <= code <= 0x0E7F:
            found.add("has_thai")
        elif 0x0900 <= code <= 0x097F:
            found.add("has_devanagari")
        elif char.isalpha():
            found.add("has_other_letters")
    if "has_han" in found and "has_kana" not in found and "has_hangul" not in found:
        found.add("han_without_kana_or_hangul")
    return found


def length_summary(lengths: list[int]) -> dict[str, int | None]:
    """最近秩百分位：p90 是排序后第 ceil(0.9*n) 个长度。"""
    if not lengths:
        return {name: None for name in ("min", "p50", "p90", "p99", "max")}
    ordered = sorted(lengths)

    def percentile(ratio: float) -> int:
        return ordered[math.ceil(ratio * len(ordered)) - 1]

    return {
        "min": ordered[0], "p50": percentile(0.50), "p90": percentile(0.90),
        "p99": percentile(0.99), "max": ordered[-1],
    }


def audit_file(source: Path) -> dict:
    """按行读取，避免一次把约数百 MB 的原始数据载入内存。"""
    rows = Counter({key: 0 for key in (
        "total_lines", "valid_objects", "blank_lines", "malformed_json", "non_object_json"
    )})
    field_stats = {name: Counter() for name in FIELDS}
    sample_types: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    target_labels: Counter[str] = Counter()
    han_candidate_labels: Counter[str] = Counter()
    signals: Counter[str] = Counter({name: 0 for name in SCRIPT_SIGNALS})
    lengths: list[int] = []
    general_r_rows = 0

    # JSONL 是「一行一条 JSON 对象」；损坏的单行只计数，不让整份审计中断。
    with source.open("r", encoding="utf-8-sig") as reader:
        for line in reader:
            rows["total_lines"] += 1
            if not line.strip():
                rows["blank_lines"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rows["malformed_json"] += 1
                continue
            if not isinstance(record, dict):
                rows["non_object_json"] += 1
                continue
            rows["valid_objects"] += 1

            for name in FIELDS:
                if name not in record:
                    field_stats[name]["missing"] += 1
                    continue
                field_stats[name]["present"] += 1
                value = record[name]
                if value is None:
                    field_stats[name]["null"] += 1
                elif isinstance(value, str) and not value.strip():
                    field_stats[name]["blank_string"] += 1
                elif not isinstance(value, str):
                    field_stats[name]["non_string"] += 1

            sample_types[safe_bucket(record, "sample_type", SAMPLE_TYPES)] += 1
            stages[safe_bucket(record, "stage", STAGES)] += 1
            label = safe_bucket(record, "label", LABELS)
            labels[label] += 1

            # 当前项目只检测模型「回复」。此处仅观察候选量，不清洗或保存训练样本。
            if record.get("sample_type") == "general" and record.get("stage") == "r":
                general_r_rows += 1
                response = record.get("response")
                if isinstance(response, str) and response.strip():
                    target_labels[label] += 1
                    lengths.append(len(response))  # Python 字符数，不等于模型 Tokenizer 的 token 数。
                    found_signals = script_signals(response)
                    for signal in found_signals:
                        signals[signal] += 1
                    if "han_without_kana_or_hangul" in found_signals:
                        han_candidate_labels[label] += 1

    return {
        "source_filename": source.name,
        "rows": dict(rows),
        "fields": {
            name: {key: field_stats[name][key] for key in
                   ("present", "missing", "null", "blank_string", "non_string")}
            for name in FIELDS
        },
        "sample_type": dict(sorted(sample_types.items())),
        "stage": dict(sorted(stages.items())),
        "label": dict(sorted(labels.items())),
        "target_reply_subset": {
            "selection": "sample_type=general, stage=r, response=nonempty string",
            "general_r_rows": general_r_rows,
            "count": len(lengths),
            "label": dict(sorted(target_labels.items())),
            "han_candidate_label": dict(sorted(han_candidate_labels.items())),
            "response_characters": length_summary(lengths),
            "script_signals": dict(signals),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="审计本地 XGuard JSONL，只输出无原文的统计报告")
    parser.add_argument("--input", type=Path,
                        default=project_root / "data" / "raw" / "xguard_train_open_200k.jsonl")
    parser.add_argument("--output", type=Path,
                        default=project_root / "reports" / "stage-2-raw-audit.json")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"找不到本地 JSONL 文件：{args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("输出路径不能与原始数据路径相同")

    report = audit_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"总行数：{report['rows']['total_lines']}")
    print(f"有效 JSON 对象：{report['rows']['valid_objects']}")
    print(f"general + r 且回复非空：{report['target_reply_subset']['count']}")
    print(f"统计报告：{args.output}")


if __name__ == "__main__":
    main()
