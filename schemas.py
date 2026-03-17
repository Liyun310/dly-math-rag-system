"""双轨提取流水线的数据结构定义。

本文件只负责定义结构化数据模型，不涉及业务逻辑或网络调用。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


QUESTION_ID_PATTERN = re.compile(r"^Q_\d+(?:_\d+)*$")


class QuestionNode(BaseModel):
    """题目节点：由试卷轨道提取出的标准题目结构。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., description="标准化题目 ID，例如 Q_1、Q_18_1")
    preamble: Optional[str] = Field(
        default=None,
        description="大题公共题干。若无公共题干则为 null。",
    )
    question_sys: str = Field(..., description="题目正文，需保留数学表达式与 LaTeX。")
    options: Optional[Dict[str, str]] = Field(
        default=None,
        description="选择题选项字典，例如 {'A': '...', 'B': '...'}。",
    )

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        """限制 question_id 为统一格式，避免后续合并失败。"""
        value = value.strip()
        if not QUESTION_ID_PATTERN.match(value):
            raise ValueError(f"非法 question_id: {value}")
        return value

    @field_validator("question_sys")
    @classmethod
    def validate_question_sys(cls, value: str) -> str:
        """题干不能为空。"""
        value = value.strip()
        if not value:
            raise ValueError("question_sys 不能为空")
        return value

    @field_validator("options")
    @classmethod
    def normalize_options(cls, value: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """标准化选项键名，统一为大写 A/B/C/D。"""
        if value is None:
            return None

        normalized: Dict[str, str] = {}
        for key, option_text in value.items():
            option_key = key.strip().upper()
            if not option_key:
                continue
            normalized[option_key] = option_text.strip()
        return normalized or None


class AnswerNode(BaseModel):
    """答案节点：由答案轨道提取并强制对齐到 question_id。"""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., description="必须与 QuestionNode.question_id 严格一致。")
    answer_sys: str = Field(..., description="解析过程与最终答案。")

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        value = value.strip()
        if not QUESTION_ID_PATTERN.match(value):
            raise ValueError(f"非法 question_id: {value}")
        return value

    @field_validator("answer_sys")
    @classmethod
    def validate_answer_sys(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer_sys 不能为空")
        return value


class MergedQA(QuestionNode):
    """最终合并结构：题目结构 + 对应解析。"""

    answer_sys: str = Field(..., description="最终对齐后的解析。")


class QuestionExtractionResult(BaseModel):
    """题目提取结果根节点（配合 JSON Mode 使用）。"""

    model_config = ConfigDict(extra="forbid")
    questions: List[QuestionNode] = Field(default_factory=list)


class AnswerExtractionResult(BaseModel):
    """答案提取结果根节点（配合 JSON Mode 使用）。"""

    model_config = ConfigDict(extra="forbid")
    answers: List[AnswerNode] = Field(default_factory=list)
