"""阶段 3：把已审计的 XGuard 回复转换为无原文报告的二分类数据集。"""

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# 这些类别来自本项目固定的 XGuard 标签表。sec 是安全，其余已知类别视作不安全。
LABELS = frozenset(
    ("sec", "pc", "dc", "dw", "pi", "ec", "ac", "def", "ti", "cy", "ph", "mh",
     "se", "sci", "pp", "cs", "acc", "mc", "ha", "ps", "ter", "sd", "ext",
     "fin", "med", "law", "cm", "ma", "md")
)
LABEL2ID = {"unsafe": 0, "safe": 1}
SPLITS = ("train", "validation", "test")


def script_signals(text: str) -> set[str]:
    """判断字符是否含汉字、假名或韩文；这是约定的筛选规则，不是语言识别器。"""
    found: set[str] = set()
    for char in text:
        code = ord(char)
        if (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
                or 0x20000 <= code <= 0x2A6DF):
            found.add("han")
        elif 0x3040 <= code <= 0x30FF:
            found.add("kana")
        elif 0xAC00 <= code <= 0xD7AF:
            found.add("hangul")
    return found


def is_chinese_candidate(text: str) -> bool:
    """包含汉字并且没有假名、韩文，符合本阶段约定的中文候选规则。"""
    signals = script_signals(text)
    return "han" in signals and "kana" not in signals and "hangul" not in signals


def normalize_for_deduplication(text: str) -> str:
    """统一兼容字符、英文大小写和空白，减少格式差异导致的重复样本。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def collect_candidates(source: Path) -> tuple[list[dict], dict]:
    """流式读 JSONL；只在内存保留符合项目筛选条件的候选回复。"""
    groups: dict[str, dict] = {}
    stats = Counter()
    source_hash = hashlib.sha256()

    with source.open("rb") as reader:
        for line_number, raw_line in enumerate(reader, start=1):
            source_hash.update(raw_line)
            stats["source_lines"] += 1
            try:
                record = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # 只报告行号，不把可能含敏感内容的原始行或解析异常回显。
                raise ValueError(f"第 {line_number} 行不是有效 UTF-8 JSON") from None
            if not isinstance(record, dict):
                raise ValueError(f"第 {line_number} 行 JSON 顶层不是对象")

            if record.get("sample_type") != "general" or record.get("stage") != "r":
                continue
            response = record.get("response")
            if not isinstance(response, str) or not response.strip():
                continue
            stats["general_r_nonempty"] += 1

            label = record.get("label")
            if not isinstance(label, str) or label not in LABELS:
                stats["excluded_unknown_label"] += 1
                continue
            if not is_chinese_candidate(response):
                stats["excluded_by_language_rule"] += 1
                continue

            class_name = "safe" if label == "sec" else "unsafe"
            stats[f"eligible_{class_name}"] += 1
            normalized = normalize_for_deduplication(response)
            if not normalized:
                stats["excluded_empty_after_normalization"] += 1
                continue

            if normalized not in groups:
                groups[normalized] = {"text": response.strip(), "classes": {class_name}}
            else:
                group = groups[normalized]
                group["classes"].add(class_name)
                stats["duplicate_rows_collapsed"] += 1

    candidates = []
    stats["conflicting_text_groups_removed"] = 0
    for group in groups.values():
        if len(group["classes"]) != 1:
            stats["conflicting_text_groups_removed"] += 1
            continue
        class_name = next(iter(group["classes"]))
        candidates.append({"text": group["text"], "class_name": class_name})

    candidates_by_class = Counter(row["class_name"] for row in candidates)
    stats["unique_nonconflicting_safe"] = candidates_by_class["safe"]
    stats["unique_nonconflicting_unsafe"] = candidates_by_class["unsafe"]
    metadata = {
        "source_sha256": source_hash.hexdigest(),
        "source_lines": stats["source_lines"],
        "selection_counts": dict(sorted(stats.items())),
    }
    return candidates, metadata


def make_splits(candidates: list[dict], max_per_class: int, seed: int) -> dict[str, list[dict]]:
    """按安全/不安全分别抽样并切分，保证每个集合中的类别比例一致。"""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate["class_name"]].append(candidate)

    rng = random.Random(seed)
    splits = {name: [] for name in SPLITS}
    for class_name in ("unsafe", "safe"):
        pool = by_class[class_name]
        rng.shuffle(pool)
        selected = pool[:max_per_class]
        if len(selected) < 10:
            raise ValueError(
                f"{class_name} 去重后只有 {len(selected)} 条；每类至少需要 10 条，"
                "才能让 train/validation/test 都有该类样本"
            )

        train_end = int(len(selected) * 0.8)
        validation_end = train_end + int(len(selected) * 0.1)
        class_splits = {
            "train": selected[:train_end],
            "validation": selected[train_end:validation_end],
            "test": selected[validation_end:],
        }
        for split_name, split_rows in class_splits.items():
            splits[split_name].extend(
                {"text": row["text"], "label": LABEL2ID[class_name]}
                for row in split_rows
            )

    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def count_split_labels(splits: dict[str, list[dict]]) -> dict:
    """把数字编码还原为可读类别，便于检查切分后类别比例。"""
    id_to_label = {value: key for key, value in LABEL2ID.items()}
    result = {}
    for split_name, rows in splits.items():
        counts = Counter(id_to_label[row["label"]] for row in rows)
        result[split_name] = {
            "unsafe": counts["unsafe"], "safe": counts["safe"], "total": len(rows)
        }
    return result


def write_outputs(splits: dict[str, list[dict]], output_dir: Path, manifest_path: Path,
                  manifest: dict) -> None:
    """写入三个 JSONL 与无原文 manifest；训练文本只进入被忽略的数据目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        target = output_dir / f"{split_name}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as writer:
            for row in rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="制作本地 XGuard 回复二分类数据集")
    parser.add_argument("--input", type=Path,
                        default=project_root / "data" / "raw" / "xguard_train_open_200k.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "data" / "processed" / "stage-3")
    parser.add_argument("--manifest", type=Path,
                        default=project_root / "reports" / "stage-3-data-manifest.json")
    parser.add_argument("--max-per-class", type=int, default=400,
                        help="每个二分类类别最多保留多少条，默认 400")
    parser.add_argument("--seed", type=int, default=42,
                        help="可复现随机种子，默认 42")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"找不到本地 JSONL 文件：{args.input}")
    if args.max_per_class < 10:
        parser.error("--max-per-class 至少为 10，保证三个数据集合均能包含两个类别")
    if args.input.resolve() in {args.output_dir.resolve(), args.manifest.resolve()}:
        parser.error("输出路径不能覆盖原始数据文件")

    try:
        candidates, source_info = collect_candidates(args.input)
        splits = make_splits(candidates, args.max_per_class, args.seed)
    except ValueError as error:
        parser.error(str(error))

    split_counts = count_split_labels(splits)
    manifest = {
        "source_filename": args.input.name,
        "source_sha256": source_info["source_sha256"],
        "source_lines": source_info["source_lines"],
        "language_rule": "response contains Han characters and contains no Kana or Hangul characters",
        "selection": "sample_type=general, stage=r, nonempty response, known XGuard label",
        "label2id": LABEL2ID,
        "deduplication": {
            "normalization": "Unicode NFKC + casefold + collapse whitespace",
            "duplicate_rows_collapsed": source_info["selection_counts"].get(
                "duplicate_rows_collapsed", 0),
            "conflicting_text_groups_removed": source_info["selection_counts"].get(
                "conflicting_text_groups_removed", 0),
        },
        "sampling": {"max_per_class": args.max_per_class, "seed": args.seed},
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "selection_counts": source_info["selection_counts"],
        "selected_rows": sum(counts["total"] for counts in split_counts.values()),
        "split_counts": split_counts,
    }
    write_outputs(splits, args.output_dir, args.manifest, manifest)

    print(f"源文件行数：{source_info['source_lines']}")
    print(f"按规则筛选并去重后：safe={source_info['selection_counts'].get('unique_nonconflicting_safe', 0)}, "
          f"unsafe={source_info['selection_counts'].get('unique_nonconflicting_unsafe', 0)}")
    print(f"最终样本数：{manifest['selected_rows']}（每类最多 {args.max_per_class}）")
    for split_name, counts in split_counts.items():
        print(f"{split_name}: safe={counts['safe']}, unsafe={counts['unsafe']}, total={counts['total']}")
    print(f"数据文件目录：{args.output_dir}")
    print(f"无原文清单：{args.manifest}")


if __name__ == "__main__":
    main()
