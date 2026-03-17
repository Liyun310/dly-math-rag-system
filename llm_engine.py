"""大模型双轨提取引擎。

职责：
1) 调用智谱模型并强制 JSON 输出；
2) 做 JSON 解析与 Pydantic 校验；
3) 在 API 调用失败或结构化校验失败时自动重试。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import ValidationError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    # 首选官方 zhipuai SDK。
    from zhipuai import ZhipuAI  # type: ignore
except ModuleNotFoundError:
    try:
        # 兼容你现有项目中常见的 zai 客户端。
        from zai import ZhipuAiClient as ZhipuAI  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未找到 zhipuai / zai 依赖。请先安装：`python -m pip install zhipuai` "
            "（或 `python -m pip install zai`）。"
        ) from exc

try:
    # 某些 zhipuai 版本提供真正的 async client。
    from zhipuai import AsyncZhipuAI  # type: ignore
except Exception:  # noqa: BLE001  # pragma: no cover - 取决于本地 SDK 版本
    AsyncZhipuAI = None  # type: ignore

from prompts import ANSWER_ALIGN_PROMPT, QUESTION_EXTRACT_PROMPT
from schemas import AnswerExtractionResult, AnswerNode, QuestionExtractionResult, QuestionNode

LOGGER = logging.getLogger(__name__)

try:
    # 用户自定义封装（优先使用）
    from zhipuai_llm import ZhipuaiLLM  # type: ignore
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # noqa: BLE001
    ZhipuaiLLM = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore


class ExtractionError(RuntimeError):
    """统一提取异常类型，便于 tenacity 捕获重试。"""


class DualTrackExtractor:
    """双轨提取器：题目轨 + 答案轨。"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "glm-4-plus",
        temperature: float = 0.1,
        timeout: int = 120,
        backend: Literal["auto", "custom", "sdk"] = "auto",
        thinking_type: str = "enabled",
    ) -> None:
        if not api_key:
            raise ValueError("缺少 ZHIPUAI_API_KEY")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.backend = backend
        self.thinking_type = thinking_type

        self._custom_llm: Optional[ZhipuaiLLM] = None  # type: ignore

        use_custom = backend == "custom" or (backend == "auto" and ZhipuaiLLM is not None)
        if use_custom:
            if ZhipuaiLLM is None:
                raise RuntimeError("backend=custom 但未能导入 zhipuai_llm.py / langchain_core。")
            self._custom_llm = ZhipuaiLLM(
                model_name=self.model_name,
                temperature=self.temperature,
                timeout=self.timeout,
                api_key=self.api_key,
                thinking_type=self.thinking_type,
            )
            self._sync_client = None
            self._async_client = None
            LOGGER.info("DualTrackExtractor 后端: custom(zhipuai_llm.py)")
        else:
            self._sync_client = ZhipuAI(api_key=self.api_key)
            self._async_client = AsyncZhipuAI(api_key=self.api_key) if AsyncZhipuAI else None
            LOGGER.info("DualTrackExtractor 后端: sdk(zhipuai/zai)")

    async def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用模型并返回 JSON 对象。

        注意：
        - 强制启用 response_format={"type": "json_object"}；
        - 若当前 SDK 无 AsyncZhipuAI，则退化为 asyncio.to_thread 包装同步调用。
        """

        if self._custom_llm is not None:
            content = await self._chat_with_custom_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            return self._parse_json_content(content)

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self._async_client is not None:
            response = await self._async_client.chat.completions.create(**payload)
        else:
            response = await asyncio.to_thread(self._sync_client.chat.completions.create, **payload)

        try:
            content = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"模型返回结构异常: {exc}") from exc

        return self._parse_json_content(content)

    async def _chat_with_custom_llm(self, system_prompt: str, user_prompt: str) -> Any:
        """通过用户自定义 zhipuai_llm 调用模型。"""
        if self._custom_llm is None:
            raise ExtractionError("自定义 LLM 未初始化")
        if SystemMessage is None or HumanMessage is None:
            raise ExtractionError("langchain_core 消息类型不可用，无法调用自定义 LLM")

        def _invoke() -> Any:
            message = self._custom_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return getattr(message, "content", message)

        return await asyncio.to_thread(_invoke)

    @staticmethod
    def _parse_json_content(content: Any) -> Dict[str, Any]:
        """将模型输出解析为 dict。

        兼容输出：
        1) 纯 JSON 字符串；
        2) ```json fenced code block；
        3) 已经是 Python dict。
        """

        if isinstance(content, dict):
            return content

        if isinstance(content, list):
            text = "".join(DualTrackExtractor._item_to_text(item) for item in content)
        else:
            text = str(content or "").strip()

        if not text:
            raise ExtractionError("模型返回为空")

        # 去掉可能出现的 Markdown 代码块包装
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fenced_match:
            text = fenced_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            repaired_text = DualTrackExtractor._repair_json_text(text)
            try:
                parsed = json.loads(repaired_text)
            except json.JSONDecodeError:
                try:
                    # 兼容部分松散 JSON（如多余逗号、注释等）
                    import json5  # type: ignore

                    parsed = json5.loads(repaired_text)
                except Exception as final_exc:  # noqa: BLE001
                    raise ExtractionError(f"JSON 解析失败: {exc}") from final_exc

        if isinstance(parsed, list):
            # 防御性兼容：如果模型返回数组，自动包一层对象。
            return {"items": parsed}
        if not isinstance(parsed, dict):
            raise ExtractionError(f"期望 JSON 对象，实际为: {type(parsed)}")
        return parsed

    @staticmethod
    def _item_to_text(item: Any) -> str:
        """把 SDK 的分段内容结构拼接成文本。"""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("text") or item.get("content") or "")
        return str(item)

    @staticmethod
    def _repair_json_text(text: str) -> str:
        """修复模型常见的非法 JSON 细节。

        重点处理：
        - 未转义的反斜杠（LaTeX 如 \\sqrt 可能被写成 \\sqrt）；
        - 仅替换非法转义，不破坏合法 JSON 转义。
        """

        # 仅把“不是合法 JSON 转义起始”的反斜杠改成双反斜杠
        repaired = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)
        return repaired

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ExtractionError, ValidationError)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING),
        reraise=True,
    )
    async def extract_questions(self, text_chunk: str) -> List[QuestionNode]:
        """题目轨提取：从试卷块中抽取 QuestionNode 列表。"""

        payload = await self._chat_json(
            system_prompt=QUESTION_EXTRACT_PROMPT,
            user_prompt=f"请抽取以下试卷文本中的题目：\n\n{text_chunk}",
        )

        if "questions" not in payload and "items" in payload:
            payload = {"questions": payload["items"]}
        if "questions" not in payload and isinstance(payload.get("data"), dict):
            if "questions" in payload["data"]:
                payload = {"questions": payload["data"]["questions"]}

        payload = self._normalize_question_payload(payload)

        try:
            result = QuestionExtractionResult.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError(f"题目结构化校验失败: {exc}") from exc

        return result.questions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ExtractionError, ValidationError)),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING),
        reraise=True,
    )
    async def extract_answers(self, text_chunk: str, question_list_summary: str) -> List[AnswerNode]:
        """答案轨提取：按“通缉令”对齐指定 question_id 的解析。"""

        system_prompt = ANSWER_ALIGN_PROMPT.replace("{target_questions}", question_list_summary)
        payload = await self._chat_json(
            system_prompt=system_prompt,
            user_prompt=f"请在以下答案文本中对齐解析：\n\n{text_chunk}",
        )

        if "answers" not in payload and "items" in payload:
            payload = {"answers": payload["items"]}
        if "answers" not in payload and isinstance(payload.get("data"), dict):
            if "answers" in payload["data"]:
                payload = {"answers": payload["data"]["answers"]}

        payload = self._normalize_answer_payload(payload)

        try:
            result = AnswerExtractionResult.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError(f"答案结构化校验失败: {exc}") from exc

        return result.answers

    @staticmethod
    def _normalize_question_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """清洗模型返回，尽量把轻微脏数据修正为可校验结构。"""
        raw_items = payload.get("questions")
        if not isinstance(raw_items, list):
            return payload

        cleaned: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            # 兼容字段漂移
            qid = item.get("question_id") or item.get("id")
            question_sys = item.get("question_sys") or item.get("question")
            preamble = item.get("preamble")
            options = item.get("options")

            if not isinstance(qid, str) or not qid.strip():
                continue
            if not isinstance(question_sys, str) or not question_sys.strip():
                continue

            normalized_options = None
            if isinstance(options, dict):
                tmp: Dict[str, str] = {}
                for key, value in options.items():
                    if value is None:
                        continue
                    if isinstance(value, (str, int, float)):
                        option_text = str(value).strip()
                        if option_text:
                            tmp[str(key).strip().upper()] = option_text
                if tmp:
                    normalized_options = tmp

            cleaned.append(
                {
                    "question_id": qid.strip(),
                    "preamble": preamble if isinstance(preamble, str) else None,
                    "question_sys": question_sys.strip(),
                    "options": normalized_options,
                }
            )

        return {"questions": cleaned}

    @staticmethod
    def _normalize_answer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """清洗答案返回，过滤明显无效项。"""
        raw_items = payload.get("answers")
        if not isinstance(raw_items, list):
            return payload

        cleaned: List[Dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            qid = item.get("question_id") or item.get("id")
            answer_sys = item.get("answer_sys") or item.get("answer") or item.get("analysis")
            if not isinstance(qid, str) or not qid.strip():
                continue
            if not isinstance(answer_sys, str) or not answer_sys.strip():
                continue

            cleaned.append(
                {
                    "question_id": qid.strip(),
                    "answer_sys": answer_sys.strip(),
                }
            )
        return {"answers": cleaned}
