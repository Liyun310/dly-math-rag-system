from typing import Any, Dict, Iterator, List, Optional
from zhipuai import ZhipuAI
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ChatMessage,
    HumanMessage
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import time
from zai import ZhipuAiClient

class ZhipuaiLLM(BaseChatModel):
    """自定义Zhipuai聊天模型。
    """

    model_name: str = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 3
    api_key: str | None = None
    thinking_type: str = "enabled" #该参数支持两种取值：enabled（动态）和 disabled （禁用）。默认情况下开启动态思考功能。

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """通过调用智谱API从而响应输入。

        Args:
            messages: 由messages列表组成的prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        """

        messages = [_convert_message_to_dict(message) for message in messages]
        start_time = time.time()
        # response = ZhipuAI(api_key=self.api_key).chat.completions.create(
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     timeout=self.timeout,
        #     stop=stop,
        #     messages=messages
        # )

        # 使用ZhipuAiClient进行API调用
        client = ZhipuAiClient(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages,
            thinking={
                "type": self.thinking_type,    # 深度思考模式
            },
        )

        time_in_seconds = time.time() - start_time
        # 处理可能为 None 的内容
        content = response.choices[0].message.content
        if content is None:
            content = ""  # 将 None 转换为空字符串
            
        message = AIMessage(
            content=response.choices[0].message.content,
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": round(time_in_seconds, 3),
            },
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """通过调用智谱API返回流式输出。

        Args:
            messages: 由messages列表组成的prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        """
        messages = [_convert_message_to_dict(message) for message in messages]
        # response = ZhipuAI().chat.completions.create(
        #     model=self.model_name,
        #     stream=True,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     timeout=self.timeout,
        #     stop=stop,
        #     messages=messages
        # )

        #使用ZhipuAiClient进行API调用
        client = ZhipuAiClient(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages,
            thinking={
                "type": self.thinking_type,    # 深度思考模式
            },
            stream=True
        )

        start_time = time.time()
        for res in response:
            if res.usage:
                usage_metadata = UsageMetadata(
                    {
                        "input_tokens": res.usage.prompt_tokens,
                        "output_tokens": res.usage.completion_tokens,
                        "total_tokens": res.usage.total_tokens,
                    }
                )
            # 处理可能为 None 的内容

            # content = res.choices[0].delta.content #对应最终输出的真正答案
            # if content is None:
            #     content = ""  # 将 None 转换为空字符串
            delta = res.choices[0].delta
            #处理思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=delta.reasoning_content or "",
                    additional_kwargs={"is_thinking": "True"}  # 标记这是思考过程
                )
            )
            # 处理回复内容    
            elif hasattr(delta, "content"):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=delta.content or "",
                        additional_kwargs={"is_thinking": "False"}  # 标记这是最终答案
                    )
                )
            else:
                continue

            # chunk = ChatGenerationChunk(
            #     message=AIMessageChunk(content=content)
            # )

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)

            yield chunk
        time_in_sec = time.time() - start_time
        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="", 
                additional_kwargs={"is_thinking": "False"},
                response_metadata={"time_in_sec": round(time_in_sec, 3)}, 
                usage_metadata=usage_metadata
            )
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。"""
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。

        该信息由LangChain回调系统使用，用于跟踪目的，使监视llm成为可能。
        """
        return {
            "model_name": self.model_name,
        }

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """把LangChain的消息格式转为智谱支持的格式

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": message.content}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict

if __name__ == "__main__":
    # Test
    # model = ZhipuaiLLM(model_name="glm-4-plus")
    model = ZhipuaiLLM(model_name="glm-4.5", temperature=0, api_key="87dce7cbc37942b7ad7c7824efd8dd22.N98I1D5xwB0jsBic",thinking_type="enabled")
    # invoke
    answer = model.invoke("你好")
    # print(answer)
    # print("------generate-----\n")
    # answer = model.invoke(
    #         [
    #         HumanMessage(content="你好!"),
    #         AIMessage(content="人类你好!"),
    #         HumanMessage(content="介绍你自己!"),
    #     ]
    # )
    # print(answer)
    print("------stream-----\n")
    # stream
    for chunk in model.stream([
            HumanMessage(content="你好!"),
            AIMessage(content="人类你好!"),
            HumanMessage(content="介绍你自己!"),
        ]):
        print(chunk, end="|")
        print("\n")
    # # batch
    # print(model.batch(["你好", "再见"]))
