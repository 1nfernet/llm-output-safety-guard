# 阶段 2：取得并审计原始数据

## 为什么先审计

真实项目不能只看到「有 20 万条」就开始训练。XGuard 混合了用户提问、模型回复和问答对，也包含多种语言和动态规则样本。我们的任务是检测**中文模型回复**，因此先核对数据文件、字段、标签、空值和语言混杂，再在阶段 3 决定如何筛选与切分。审计只汇总统计，不产生训练集。

JSONL 是 *JSON Lines*：一行是一条独立 JSON 记录。可以像 Java 逐行读日志一样处理它，不必一次把约 565 MB 数据放入内存。当前脚本用 Python 标准库 `json.loads()` 解析每行；这一步还不需要 `datasets` 包、Tokenizer 或模型。

## 来源与本阶段固定版本

- 数据集：[`Alibaba-AAIG/XGuard-Train-Open-200K`](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K)。
- 固定提交：[`88d30c0d80abde49663f97d1bdbbec519995d79a`](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K/commit/88d30c0d80abde49663f97d1bdbbec519995d79a)。
- 文件：[`xguard_train_open_200k.jsonl`](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K/blob/main/xguard_train_open_200k.jsonl)。文件页标示约 **565 MB**，SHA256 为 `58ca381b907165d08de5c5d56d9bded3103db3ee4f9361a382355ca19c69cb18`。文件页当前显示的提交是上面的 `88d30c0`；若以后 `main` 更新，仍以这个提交和 SHA256 为准。
- 上游说明：`sample_type=general` 是基础数据；`stage=q` 只有提问，`r` 只有回复，`qr` 是问答对。`response` 是本项目的候选输入，`label` 是风险类别标记。[字段说明](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K)

请在可访问外网的电脑上从文件页点击 **Download**，将实际 JSONL 文件转移到项目的 `data\raw\xguard_train_open_200k.jsonl`。不要把网页 HTML、Git LFS/Xet 指针或上游 `examples` 里的加工示例当作原始文件。原始数据位于 Git 忽略目录，不提交，也无需把正文发到对话中。

## 由学习者运行：校验文件与执行审计

在 PowerShell 中进入项目目录：

```powershell
Set-Location 'D:\projectStore\codeXProject\python-basic-learning\projects\llm-output-safety-guard'
New-Item -ItemType Directory -Force .\data\raw | Out-Null
```

将文件放好后，先查看文件大小与 SHA256。`Length` 是字节数；SHA256 是文件内容的指纹，能发现下载不完整或版本变化。

```powershell
Get-Item .\data\raw\xguard_train_open_200k.jsonl | Select-Object Name,Length
(Get-FileHash .\data\raw\xguard_train_open_200k.jsonl -Algorithm SHA256).Hash
```

哈希应与上面文件页公布的 `58ca...9c69cb18` **完整一致**。文件应约 565 MB；只得到很小文件时，先排查是否保存了网页或指针。校验通过后，再运行：

```powershell
uv run --no-sync --offline --no-python-downloads python .\scripts\audit_raw_data.py
Get-Content .\reports\stage-2-raw-audit.json
```

`uv run` 使用阶段 1 的 `.venv`；`--no-sync` 不重新安装依赖，`--offline` 阻止 uv 联网，`--no-python-downloads` 不下载 Python。脚本从 `data\raw` 逐行只读原始文件，把统计写到 `reports\stage-2-raw-audit.json`。两个目录都被 `.gitignore` 忽略。报告仅包含文件名与统计值，不含 `prompt`、`response`、`explanation`、`id` 的原值；未知字段名、未知标签值也不回显。

## 如何读报告

- `rows`：总行数、有效 JSON 对象、空行、无效 JSON 和非对象行。如果后几项不为 0，需要先判断是否拿到了正确文件。
- `fields`：八个预期字段各有多少条出现、缺失、为 `null`、空字符串或类型不是字符串。`null` 和空字符串都不能直接当训练文本。
- `sample_type`、`stage`、`label`：全文件的类别分布。已知标签保留类别代号，未知值只汇总为 `<other>`。注意全文件含提问与问答对，其标签分布不等于我们训练子集的分布。
- `target_reply_subset`：**仅观察** `general + r + 非空字符串 response` 的数量和标签分布。`response_characters` 是 Python 字符数的最小值、p50、p90、p99、最大值；它不是模型 token 数，不能直接判断 128 token 会截断多少。
- `script_signals`：候选回复里出现汉字、假名、韩文、拉丁字母等文字系统的数量。信号可能重叠；`han_without_kana_or_hangul` 只是「中文候选」的粗略代理，并非可靠语言判定。日文纯汉字、夹杂英文的中文都可能影响判断。`han_candidate_label` 统计这类候选的各标签数量，帮助判断阶段 3 每类能否取得足够样本；阶段 3 再结合真实分布确定筛选规则。

`sec` 在上游类别模板中表示 Safe；其余类别的含义也应以[上游模板](https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K/blob/main/examples/prompt_template.txt)核对。此阶段先观察真实标签与数量，不把它们提前合并成二分类标签，也不做去重、抽样和训练集切分。

## 本次真实验收结果（2026-10-14）

学习者实际下载并运行审计脚本，结果如下：

- 文件大小：`564,525,162` 字节（约 565 MB）。
- 本地 SHA256：`58ca381b907165d08de5c5d56d9bded3103db3ee4f9361a382355ca19c69cb18`，与固定版本文件页公布值一致。
- 解析：200,000 行、200,000 个有效 JSON 对象；空行、无效 JSON、非对象均为 0。
- 目标候选：`sample_type=general`、`stage=r` 且 `response` 非空，共 69,346 条。
- 目标候选标签共 29 类，分布明显不均：`sec` 为 37,015 条，而 `md` 只有 24 条、`ps` 89 条。阶段 3 需要据此处理类别数量不足的问题，不能假设每类都能抽到 400 条。
- 目标候选字符数 p50 为 851、p90 为 3,283、p99 为 6,948、最大值为 14,183。字符数不等于 Tokenizer token 数；阶段 3 会用实际 Tokenizer 观察长度与截断。
- 脚本的汉字文字系统信号为 9,138 条；进一步排除同条中的假名和韩文后为 6,456 条。这只是粗略候选，不是可靠的中文语言分类，阶段 3 仍需审查中文筛选策略。
- `prompt` 空值 71,925 条与 `stage=r` 数量一致；`response` 空值 49,819 条与 `stage=q` 数量一致；`policy` 空值 190,000 条与 `sample_type=general` 数量一致，符合本次看到的数据结构。

审计报告保存在本地忽略目录 `reports/`，包含不带原文的统计；原始 JSONL 在 `data/raw/`。两者都不提交 Git。本阶段已完成：数据来源与内容结构已由真实文件、哈希和实际审计输出核对。阶段 3 已进入数据集制作，筛选与切分规则及实操说明见[阶段 3 说明](stage-3-dataset-preparation.md)。
