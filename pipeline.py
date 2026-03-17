"""双轨提取与合并流水线。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from llm_engine import DualTrackExtractor, ExtractionError
from schemas import AnswerNode, MergedQA, QuestionNode

LOGGER = logging.getLogger(__name__)


HEADING_PATTERN = re.compile(
    r"^\s*(#{1,6}\s+.+|[-*]\s*[一二三四五六七八九十]+、.+|[一二三四五六七八九十]+、.+)\s*$"
)


def _split_long_text(text: str, max_chunk_chars: int) -> List[str]:
    """当单个段落过长时，按字符窗口做兜底切分。"""
    if len(text) <= max_chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chunk_chars, text_len)
        # 优先在结尾附近找换行断点，减少切断句子
        if end < text_len:
            pivot = text.rfind("\n", start + int(max_chunk_chars * 0.6), end)
            if pivot > start:
                end = pivot
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


def chunk_markdown_by_heading(markdown_text: str, max_chunk_chars: int = 2400) -> List[str]:
    """按“标题/大题分区”粗切，再按最大字符数合并为合适块。"""
    lines = markdown_text.splitlines()
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        if HEADING_PATTERN.match(line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: List[str] = []
    buffer = ""
    for sec in sections:
        if not sec:
            continue
        if len(sec) > max_chunk_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_text(sec, max_chunk_chars=max_chunk_chars))
            continue

        candidate = f"{buffer}\n\n{sec}".strip() if buffer else sec
        if len(candidate) <= max_chunk_chars:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = sec
    if buffer:
        chunks.append(buffer.strip())

    return chunks


def _question_sort_key(question_id: str) -> Tuple[int, ...]:
    """把 Q_18_2 变成可排序 key。"""
    nums = [int(x) for x in re.findall(r"\d+", question_id)]
    if not nums:
        return (10**9,)
    return tuple(nums)


def _question_base_num(question_id: str) -> int:
    match = re.match(r"Q_(\d+)", question_id)
    return int(match.group(1)) if match else -1


def _group_questions(questions: Sequence[QuestionNode], group_size: int = 5) -> List[List[QuestionNode]]:
    return [list(questions[i : i + group_size]) for i in range(0, len(questions), group_size)]


def _build_question_summary(questions: Sequence[QuestionNode], snippet_len: int = 20) -> str:
    """构造“通缉令”摘要：ID + 题干前若干字。"""
    lines: List[str] = []
    for q in questions:
        snippet = re.sub(r"\s+", " ", q.question_sys).strip()
        snippet = snippet[:snippet_len]
        lines.append(f"- {q.question_id}: {snippet}")
    return "\n".join(lines)


def _score_answer_chunk(chunk: str, base_numbers: Iterable[int]) -> int:
    """给答案块打分，用于挑选“最可能包含目标题”的上下文。"""
    score = 0
    for num in base_numbers:
        if num < 0:
            continue

        # 强特征：明确的题号答案开头
        if re.search(rf"(?m)^\s*#{{0,6}}\s*{num}\s*[【\.\uFF0E、(（]", chunk):
            score += 4
        if re.search(rf"(?m)^\s*#+\s*{num}\s*【答案】", chunk):
            score += 4

        # 中特征：表格中出现题号
        if re.search(rf"\|\s*{num}\s*\|", chunk):
            score += 2

        # 弱特征：普通文本出现题号
        if re.search(rf"(?<!\d){num}(?!\d)", chunk):
            score += 1

    return score


def _select_answer_context(
    answer_chunks: Sequence[str],
    group_questions: Sequence[QuestionNode],
    max_context_chars: int = 7000,
) -> str:
    """根据题号从答案分块中挑选上下文，减少无关文本。"""
    if not answer_chunks:
        return ""

    base_numbers = {_question_base_num(q.question_id) for q in group_questions}
    scored = []
    for idx, chunk in enumerate(answer_chunks):
        score = _score_answer_chunk(chunk, base_numbers)
        if score > 0:
            scored.append((score, idx, chunk))

    # 按得分选，最终再按原始顺序拼接，保证上下文自然连贯
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = scored[:3]

    if not selected:
        # 兜底：按题号估算位置取一个块
        positive_nums = sorted(n for n in base_numbers if n >= 0)
        fallback_index = 0
        if positive_nums:
            fallback_index = min(len(answer_chunks) - 1, max(0, (positive_nums[0] - 1) // 5))
        selected = [(1, fallback_index, answer_chunks[fallback_index])]

    selected.sort(key=lambda x: x[1])

    merged_parts: List[str] = []
    total_len = 0
    for _, _, chunk in selected:
        if total_len >= max_context_chars:
            break
        remain = max_context_chars - total_len
        piece = chunk[:remain]
        merged_parts.append(piece)
        total_len += len(piece)

    return "\n\n".join(merged_parts).strip()


def _append_jsonl(output_jsonl: Path, records: Sequence[MergedQA]) -> None:
    """以 JSONL 方式追加写入。"""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as fp:
        for item in records:
            fp.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


async def run_pipeline(
    text_md_path: str | Path,
    ans_md_path: str | Path,
    output_jsonl: str | Path,
    extractor: DualTrackExtractor,
    question_chunk_chars: int = 2400,
    answer_chunk_chars: int = 2600,
    group_size: int = 5,
) -> List[MergedQA]:
    """执行双轨提取与合并主流程。"""
    text_path = Path(text_md_path)
    ans_path = Path(ans_md_path)
    output_path = Path(output_jsonl)

    text_md = text_path.read_text(encoding="utf-8")
    ans_md = ans_path.read_text(encoding="utf-8")

    # ---------- 第一阶段：试卷轨提取题目 ----------
    question_chunks = chunk_markdown_by_heading(text_md, max_chunk_chars=question_chunk_chars)
    LOGGER.info("第一阶段：试卷分块完成，共 %d 块", len(question_chunks))

    question_map: Dict[str, QuestionNode] = {}
    for idx, chunk in enumerate(question_chunks, start=1):
        LOGGER.info("提取题目中：chunk %d/%d", idx, len(question_chunks))
        try:
            questions = await extractor.extract_questions(chunk)
        except ExtractionError:
            LOGGER.exception("题目提取失败：chunk %d", idx)
            continue
        except Exception:  # noqa: BLE001
            LOGGER.exception("题目提取出现未预期异常：chunk %d", idx)
            continue

        for q in questions:
            # 若重复命中同一题，保留信息更完整的版本
            if q.question_id not in question_map:
                question_map[q.question_id] = q
                continue
            if len(q.question_sys) > len(question_map[q.question_id].question_sys):
                question_map[q.question_id] = q

    question_nodes = sorted(question_map.values(), key=lambda q: _question_sort_key(q.question_id))
    LOGGER.info("第一阶段完成：共提取到 %d 道题", len(question_nodes))

    if not question_nodes:
        raise RuntimeError("未提取到任何题目，请检查试卷文本格式或提示词。")

    # ---------- 第二阶段：答案轨按5题一组对齐 ----------
    answer_chunks = chunk_markdown_by_heading(ans_md, max_chunk_chars=answer_chunk_chars)
    LOGGER.info("第二阶段：答案分块完成，共 %d 块", len(answer_chunks))

    groups = _group_questions(question_nodes, group_size=group_size)
    answer_map: Dict[str, AnswerNode] = {}

    for g_idx, group in enumerate(groups, start=1):
        group_ids = {q.question_id for q in group}
        summary = _build_question_summary(group)
        ans_context = _select_answer_context(answer_chunks, group)

        LOGGER.info(
            "对齐答案中：group %d/%d，题目=%s",
            g_idx,
            len(groups),
            ",".join(sorted(group_ids, key=_question_sort_key)),
        )

        try:
            aligned = await extractor.extract_answers(ans_context, summary)
        except ExtractionError:
            LOGGER.exception("答案对齐失败：group %d，尝试回退到全量答案上下文", g_idx)
            try:
                aligned = await extractor.extract_answers(ans_md, summary)
            except Exception:  # noqa: BLE001
                LOGGER.exception("答案对齐回退也失败：group %d", g_idx)
                aligned = []
        except Exception:  # noqa: BLE001
            LOGGER.exception("答案对齐出现未预期异常：group %d", g_idx)
            aligned = []

        for ans in aligned:
            if ans.question_id in group_ids:
                answer_map[ans.question_id] = ans

        # 若仍缺失，追加一次“全量答案 + 缺失题”兜底提取
        missing_ids = sorted(group_ids - set(answer_map.keys()), key=_question_sort_key)
        if missing_ids:
            missing_group = [q for q in group if q.question_id in missing_ids]
            missing_summary = _build_question_summary(missing_group)
            try:
                retry_aligned = await extractor.extract_answers(ans_md, missing_summary)
                for ans in retry_aligned:
                    if ans.question_id in group_ids:
                        answer_map[ans.question_id] = ans
            except Exception:  # noqa: BLE001
                LOGGER.warning(
                    "缺失题二次对齐失败：group %d，missing=%s",
                    g_idx,
                    ",".join(missing_ids),
                )

    # ---------- 第三阶段：按 question_id 合并并输出 JSONL ----------
    if output_path.exists():
        output_path.unlink()

    merged_records: List[MergedQA] = []
    for question in question_nodes:
        answer = answer_map.get(question.question_id)
        answer_sys = (
            answer.answer_sys
            if answer is not None
            else "原文解析缺失：未在答案文本中稳定定位到该题解析。"
        )
        merged = MergedQA(
            question_id=question.question_id,
            preamble=question.preamble,
            question_sys=question.question_sys,
            options=question.options,
            answer_sys=answer_sys,
        )
        merged_records.append(merged)

    _append_jsonl(output_path, merged_records)
    LOGGER.info("第三阶段完成：已输出 %d 条到 %s", len(merged_records), output_path)
    return merged_records
