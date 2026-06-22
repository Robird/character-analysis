from dataclasses import dataclass
import json
import logging
import os
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMResponseError(Exception):
    pass


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, LLMResponseError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.RequestError)


@dataclass(frozen=True)
class RawToolCall:
    """从模型响应中原样提取出的一次工具调用。

    Attributes:
        id: 工具调用 id，回传 ``role="tool"`` 反馈时需要原样带上。
        name: 被调用的工具名。
        arguments: 工具参数的原始 JSON 字符串，尚未解析或校验。
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMChatResult:
    """一次多工具对话请求的结果，供工具调用循环（agent）使用。

    本结果不强制满足任何更高层语义约束：模型可能返回 0 个、1 个或多个工具调用，
    参数也未经解析或校验，由上层 agent / 调度器决定如何反馈。

    Attributes:
        text: 助手正文内容（可与工具调用并存）。
        reasoning: 推理内容（若模型提供）。
        tool_calls: 本轮模型发起的所有工具调用，按原样顺序排列。
        message: 原始 assistant message，保留以便调试与审计。
        finish_reason: 本次响应的 finish_reason。
    """

    text: str
    reasoning: str
    tool_calls: tuple[RawToolCall, ...]
    message: dict[str, Any]
    finish_reason: str

    def to_assistant_message(self) -> dict[str, Any]:
        """返回可直接回放到下一轮请求的标准 assistant 消息。

        基于解析后的 ``text`` / ``tool_calls`` 重建，而不是原样复用 provider
        返回的 ``message``，以避免把 ``reasoning_content`` 等供应商私有字段重新
        喂回模型。
        """

        message: dict[str, Any] = {"role": "assistant"}
        if self.tool_calls:
            message["content"] = self.text or None
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                }
                for tool_call in self.tool_calls
            ]
            return message
        message["content"] = self.text
        return message


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "deepseek-v4-pro",
        timeout: int = 240,
    ):
        resolved_base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not resolved_api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required when api_key is not provided")
        self.base_url = resolved_base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self._client = httpx.Client(headers=self.headers, timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        parallel_tool_calls: bool | None = None,
    ) -> LLMChatResult:
        # Low-level building block for tool-call loops. It only speaks the wire
        # protocol and normalizes the response; higher-level semantics such as
        # "must emit one valid structured object" belong in the agent layer.
        return self._request_chat_result(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request_chat_result(
        self,
        messages: list[dict],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        temperature: float,
        parallel_tool_calls: bool | None,
    ) -> LLMChatResult:
        raw = self._post(
            self._build_chat_body(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
            )
        )
        return self._parse_chat_result(raw)

    def _build_chat_body(
        self,
        messages: list[dict],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        temperature: float,
        parallel_tool_calls: bool | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                body["parallel_tool_calls"] = parallel_tool_calls
        return body

    def _post(self, body: dict) -> dict:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def _parse_chat_result(cls, raw_response: dict) -> LLMChatResult:
        message = cls._extract_message(raw_response)
        content = cls._normalize_message_content(message.get("content", ""))
        reasoning = cls._extract_reasoning_content(raw_response)
        finish_reason = raw_response.get("choices", [{}])[0].get("finish_reason", "")
        return LLMChatResult(
            text=content,
            reasoning=reasoning,
            tool_calls=cls._extract_raw_tool_calls(message),
            message=message,
            finish_reason=finish_reason,
        )

    @classmethod
    def _extract_raw_tool_calls(cls, message: dict) -> tuple[RawToolCall, ...]:
        raw_tool_calls = message.get("tool_calls") or []
        calls: list[RawToolCall] = []
        for tool_call in raw_tool_calls:
            function_payload = tool_call.get("function") or {}
            calls.append(
                RawToolCall(
                    id=tool_call.get("id", ""),
                    name=function_payload.get("name", ""),
                    arguments=cls._normalize_message_content(function_payload.get("arguments", "")),
                )
            )
        return tuple(calls)

    @staticmethod
    def _extract_message(raw_response: dict) -> dict[str, Any]:
        try:
            return raw_response["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMResponseError(f"Unexpected response structure: {e}")

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("content", "")
                if text:
                    parts.append(str(text).strip())
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip() if content else ""

    @staticmethod
    def _extract_reasoning_content(raw_response: dict) -> str:
        try:
            message = LLMClient._extract_message(raw_response)
        except LLMResponseError:
            return ""

        reasoning = message.get("reasoning_content", "")
        if isinstance(reasoning, list):
            parts: list[str] = []
            for item in reasoning:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(reasoning) if reasoning else ""
