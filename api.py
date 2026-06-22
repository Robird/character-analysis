from dataclasses import dataclass
import json
import logging
import os
from typing import Any
from typing import Generic
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
DEFAULT_STRUCTURED_TOOL_NAME = "submit_structured_output"


class LLMResponseError(Exception):
    pass


class MissingToolCallError(LLMResponseError):
    pass


class InvalidToolArgumentsError(LLMResponseError):
    pass


class StructuredOutputValidationError(LLMResponseError):
    pass


RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, LLMResponseError):
        # Contract violations rarely improve with blind retries.
        return not isinstance(
            exc,
            (
                MissingToolCallError,
                InvalidToolArgumentsError,
                StructuredOutputValidationError,
            ),
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.RequestError)


@dataclass(frozen=True)
class LLMTextResult:
    text: str
    reasoning: str = ""


@dataclass(frozen=True)
class LLMFillResult(Generic[StructuredModelT]):
    value: StructuredModelT
    text: str
    reasoning: str = ""


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

    def text(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> LLMTextResult:
        # Do not expose max_tokens caps on this client. In practice they often
        # burn budget on reasoning or truncate tool arguments, producing a
        # half-result that must be retried anyway.
        return self._request_text_result(
            messages,
            temperature=temperature,
        )

    def fill(
        self,
        messages: list[dict],
        *,
        schema: type[StructuredModelT],
        temperature: float = 0.0,
    ) -> LLMFillResult[StructuredModelT]:
        return self._request_fill_result(
            messages,
            schema=schema,
            temperature=temperature,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request_text_result(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> LLMTextResult:
        # Some tasks only need the model's natural-language answer while still
        # preserving reasoning capture. Keep this path separate from structured
        # contracts so callers do not force long-form text through a fake
        # one-field schema.
        raw = self._post(self._build_text_body(messages, temperature))
        return self._parse_chat_text_result(raw)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_is_retryable_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request_fill_result(
        self,
        messages: list[dict],
        *,
        schema: type[StructuredModelT],
        temperature: float = 0.0,
    ) -> LLMFillResult[StructuredModelT]:
        tool = self._build_structured_output_tool(schema)
        raw = self._post(
            self._build_tool_call_body(
                self._with_structured_output_guidance(messages),
                tool,
                temperature,
            )
        )
        arguments, reasoning, text = self._extract_expected_tool_call(
            raw,
            expected_tool_name=DEFAULT_STRUCTURED_TOOL_NAME,
        )
        payload = self._validate_structured_payload(
            schema,
            arguments,
            output_label=DEFAULT_STRUCTURED_TOOL_NAME,
        )
        return LLMFillResult(
            value=payload,
            text=text,
            reasoning=reasoning,
        )

    def _build_text_body(
        self,
        messages: list[dict],
        temperature: float,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        return body

    def _build_tool_call_body(
        self,
        messages: list[dict],
        tool: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "tools": [tool],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
        }
        return body

    @staticmethod
    def _with_structured_output_guidance(
        messages: list[dict],
    ) -> list[dict]:
        # Keep this guidance centralized so provider/protocol knowledge lives in
        # one place instead of being re-explained in every business prompt.
        guidance = {
            "role": "system",
            "content": (
                "Return the final structured result by calling the provided function "
                f"`{DEFAULT_STRUCTURED_TOOL_NAME}` exactly once. Do not place the final structured payload "
                "in assistant text."
            ),
        }
        insert_at = 0
        for message in messages:
            if message.get("role") == "system":
                insert_at += 1
                continue
            break
        return [*messages[:insert_at], guidance, *messages[insert_at:]]

    @staticmethod
    def _build_structured_output_tool(
        schema: type[BaseModel],
    ) -> dict[str, Any]:
        # Tool parameter schema is derived from the typed output model so the
        # wire contract and runtime validation stay aligned.
        schema_title = schema.model_json_schema().get("title") or schema.__name__
        return {
            "type": "function",
            "function": {
                "name": DEFAULT_STRUCTURED_TOOL_NAME,
                "description": f"Submit the structured result for {schema_title}.",
                "parameters": schema.model_json_schema(),
            },
        }

    @staticmethod
    def _validate_structured_payload(
        output_model: type[StructuredModelT],
        raw_payload: dict[str, Any],
        *,
        output_label: str,
    ) -> StructuredModelT:
        try:
            return output_model.model_validate(raw_payload)
        except ValidationError as exc:
            raise StructuredOutputValidationError(
                "Invalid structured output for "
                f"{output_label!r}: {exc}; payload_preview={str(raw_payload)[:500]}"
            ) from exc

    def _post(self, body: dict) -> dict:
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    @classmethod
    def _parse_chat_text_result(cls, raw_response: dict) -> LLMTextResult:
        message = cls._extract_message(raw_response)
        content = cls._normalize_message_content(message.get("content", ""))
        reasoning = cls._extract_reasoning_content(raw_response)
        if not content:
            finish_reason = raw_response.get("choices", [{}])[0].get("finish_reason", "unknown")
            raise LLMResponseError(
                "Empty content from model when text was expected. "
                f"finish_reason={finish_reason!r}, reasoning_preview={reasoning[:200]!r}"
            )
        return LLMTextResult(
            text=content,
            reasoning=reasoning,
        )

    @classmethod
    def _extract_expected_tool_call(
        cls,
        raw_response: dict,
        *,
        expected_tool_name: str,
    ) -> tuple[dict[str, Any], str, str]:
        message = cls._extract_message(raw_response)
        tool_calls = message.get("tool_calls") or []
        reasoning = cls._extract_reasoning_content(raw_response)
        content = cls._normalize_message_content(message.get("content", ""))
        if not tool_calls:
            finish_reason = raw_response.get("choices", [{}])[0].get("finish_reason", "unknown")
            # Keep both content and reasoning previews in the error: with
            # reasoning models this is often the fastest way to tell whether we
            # hit a token budget issue, a contract drift, or a genuine API bug.
            raise MissingToolCallError(
                "Missing tool call from model. "
                f"expected_tool={expected_tool_name!r}, finish_reason={finish_reason!r}, "
                f"content_preview={content[:200]!r}, reasoning_preview={reasoning[:200]!r}"
            )
        for tool_call in tool_calls:
            function_payload = tool_call.get("function") or {}
            tool_name = function_payload.get("name")
            if tool_name != expected_tool_name:
                continue
            arguments_text = cls._normalize_message_content(function_payload.get("arguments", ""))
            arguments = cls._parse_tool_arguments(arguments_text, tool_name=tool_name)
            return arguments, reasoning, content
        available_tools = [((tool_call.get("function") or {}).get("name")) for tool_call in tool_calls]
        raise MissingToolCallError(
            "Expected tool call not found. "
            f"expected_tool={expected_tool_name!r}, available_tools={available_tools!r}"
        )

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
    def _parse_tool_arguments(content: str, *, tool_name: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise InvalidToolArgumentsError(
                "Invalid tool arguments JSON for "
                f"{tool_name!r}: {exc}; content_preview={content[:500]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise InvalidToolArgumentsError(
                f"Expected tool arguments object for {tool_name!r}, got {type(parsed).__name__}"
            )
        return parsed

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
