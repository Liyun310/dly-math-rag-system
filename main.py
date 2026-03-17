"""双轨提取流水线入口。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_engine import DualTrackExtractor
from pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高中数学 RAG 双轨提取与合并脚本")
    parser.add_argument(
        "--text-md",
        type=str,
        default="data/md_outputs/bashu_text.md",
        help="试卷 Markdown 路径",
    )
    parser.add_argument(
        "--ans-md",
        type=str,
        default="data/md_outputs/bashu_ans.md",
        help="答案 Markdown 路径",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/output/final_qa.jsonl",
        help="最终 JSONL 输出路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glm-4-plus",
        help="智谱模型名称",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="custom",
        choices=["custom", "sdk", "auto"],
        help="LLM 后端：custom 使用本地 zhipuai_llm.py，sdk 使用官方 SDK，auto 自动选择",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    api_key = os.getenv("ZHIPUAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("未读取到 ZHIPUAI_API_KEY，请检查 .env 或系统环境变量。")

    extractor = DualTrackExtractor(
        api_key=api_key,
        model_name=args.model,
        backend=args.backend,
    )

    records = await run_pipeline(
        text_md_path=Path(args.text_md),
        ans_md_path=Path(args.ans_md),
        output_jsonl=Path(args.output_jsonl),
        extractor=extractor,
    )

    logging.info("流程结束：共生成 %d 条 QA 记录", len(records))


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
