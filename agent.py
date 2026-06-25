#!/usr/bin/env python3
"""任务导向的 LLM Agent 节点：执行工具调用循环，收集结构化产出。

本模块在 :mod:`api` 的 :class:`~api.LLMClient` 之上提供一个面向管线的「LLM 节点」：
给定系统提示词与若干工具，节点会驱动一轮或多轮工具调用循环，把模型发起的工具调用
分发到对应处理逻辑，并把结果反馈给模型，直到模型不再调用工具为止。

两类工具：

* 结构化产出工具（:meth:`Agent.add_output_tool`）——用 pydantic 模型描述产出结构，
  每次调用收集一条产出并强校验；校验失败时把错误反馈给模型以触发结构自愈。
* 通用动作工具（:meth:`Agent.add_function_tool`）——绑定自定义 handler，面向未来的
  查资料、读写文件等副作用操作，handler 的返回值会序列化后反馈给模型。

主任务循环结束后，节点再追加一轮询问，让模型通过工具调用自报任务执行状态
（success / failed / paused），便于上层调度器以结构化方式获知任务结果。

典型用法::

    from pydantic import BaseModel
    from api import LLMClient
    from agent import Agent

    class LifeStage(BaseModel):
        name: str
        summary: str

    agent = Agent("你是一个历史人物分析专家……", client=LLMClient())
    agent.add_output_tool("output_life_stage", LifeStage, "用此工具输出人物的一个人生阶段。")
    result = agent.run("请分析李白经历了哪些主要人生阶段，每个阶段一次工具调用。")

    for stage in result.by_tool("output_life_stage"):
        print(stage.name, stage.summary)
    print(result.status, result.status_detail)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Literal
from typing import Protocol

from pydantic import BaseModel
from pydantic import ValidationError

from api import LLMChatResult
from api import LLMResponseError
from api import RawToolCall

logger = logging.getLogger(__name__)


# 调度器用来向模型索取结构化状态报告的收尾提问。语气刻意机械化，强调「只通过工具
# 调用作答」，避免模型用自然语言闲聊，从而稳定地拿到三选一的任务状态。
_STATUS_QUERY_PROMPT = (
    "我是一个调度器程序，看不懂自然语言。"
    "请仅通过 `report_task_status` 工具，告诉我刚刚这个任务的执行状态，三选一：\n"
    "- success：任务已成功完成。\n"
    "- failed：遇到无法解决的困难，任务失败。\n"
    "- paused：尚未结束，因需要汇报进度或向我提问而暂停。\n"
    "如有补充说明，请写入 detail 字段。"
)
_STATUS_TOOL_NAME = "report_task_status"


class AgentStatus(str, Enum):
    """任务执行状态。前三者与 :class:`TaskStatusReport` 的 status 字段取值对齐。

    Attributes:
        SUCCESS: 任务成功完成。
        FAILED: 遇到无法解决的困难而失败。
        PAUSED: 可继续但未完成，因汇报进度或提问而停止。
        UNKNOWN: 未能取得状态报告（如状态查询本身失败）时的兜底值。
    """

    SUCCESS = "success"
    FAILED = "failed"
    PAUSED = "paused"
    UNKNOWN = "unknown"


class TaskStatusReport(BaseModel):
    """模型自报的任务执行状态。"""

    status: Literal["success", "failed", "paused"]
    detail: str = ""


@dataclass(frozen=True)
class StructuredOutputTool:
    """结构化产出工具：每次调用收集一条经 pydantic 校验的产出。

    Attributes:
        name: 工具名，作为模型可调用的函数名。
        schema: 产出结构的 pydantic 模型；其 JSON schema 同时用作工具参数定义与运行时校验。
        description: 工具说明，提示模型何时及如何使用。
    """

    name: str
    schema: type[BaseModel]
    description: str


@dataclass(frozen=True)
class FunctionTool:
    """通用动作工具：绑定 handler，其返回值序列化后反馈给模型。

    面向未来的查资料、读写文件等带副作用的操作。

    Attributes:
        name: 工具名，作为模型可调用的函数名。
        parameters: 工具参数的原始 JSON schema。
        description: 工具说明，提示模型何时及如何使用。
        handler: 处理一次工具调用的可调用对象，入参为解析后的参数字典，返回值将反馈给模型。
    """

    name: str
    parameters: dict[str, Any]
    description: str
    handler: Callable[[dict[str, Any]], Any]


_Tool = StructuredOutputTool | FunctionTool


@dataclass(frozen=True)
class StructuredOutput:
    """一条收集到的结构化产出。

    Attributes:
        tool_name: 产生该条产出的结构化产出工具名。
        value: 校验通过的 pydantic 模型实例。
    """

    tool_name: str
    value: BaseModel


@dataclass(frozen=True)
class AgentCoreResult:
    """一次基础 tool loop 的结果。"""

    final_text: str
    outputs: tuple[StructuredOutput, ...]
    intermediate_texts: tuple[str, ...]
    iterations: int
    max_iterations_reached: bool
    messages: tuple[dict[str, Any], ...]

    def by_tool(self, name: str) -> list[BaseModel]:
        """返回指定结构化产出工具收集到的所有产出值，保持产生顺序。"""
        return [output.value for output in self.outputs if output.tool_name == name]


@dataclass(frozen=True)
class AgentResult:
    """一次任务运行的完整结果。

    Attributes:
        status: 模型自报的任务执行状态。
        status_detail: 状态的补充说明。
        final_text: 主任务循环自然结束时模型的收尾正文（被最大轮数截断时为空串）。
        outputs: 按发生顺序排列的所有合法结构化产出。
        intermediate_texts: 循环中夹在工具调用之间的模型正文片段。
        iterations: 主任务循环实际执行的轮数。
        max_iterations_reached: 是否因达到最大轮数而被强制结束。
        messages: 完整对话历史（含状态汇报轮次，如启用），便于调试与审计。
    """

    status: AgentStatus
    status_detail: str
    final_text: str
    outputs: tuple[StructuredOutput, ...]
    intermediate_texts: tuple[str, ...]
    iterations: int
    max_iterations_reached: bool
    messages: tuple[dict[str, Any], ...]

    def by_tool(self, name: str) -> list[BaseModel]:
        """返回指定结构化产出工具收集到的所有产出值，保持产生顺序。"""
        return [output.value for output in self.outputs if output.tool_name == name]


class AgentClient(Protocol):
    """Agent 运行时依赖的最小客户端协议。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        parallel_tool_calls: bool | None = None,
    ) -> LLMChatResult: ...


class Agent:
    """任务导向的 LLM 节点：用系统提示词 + 客户端创建，驱动工具调用循环完成任务。

    实例是无状态可复用的：每次 :meth:`run` 都基于固定的系统提示词与已注册工具，从空白
    对话开始，产出独立的 :class:`AgentResult`，互不影响。
    """

    def __init__(
        self,
        system_prompt: str,
        client: AgentClient,
        *,
        max_iterations: int = 64,
    ) -> None:
        """创建一个 Agent 节点。

        Args:
            system_prompt: 角色设定与任务背景，作为对话的 system 消息。
            client: 已配置好的 LLM 客户端，提供 ``chat`` 能力。
            max_iterations: 主任务循环的最大轮数上限，防止工具调用失控。

        Raises:
            ValueError: max_iterations 小于 1。
        """
        if max_iterations < 1:
            raise ValueError("max_iterations 必须 >= 1")
        self.system_prompt = system_prompt
        self.client = client
        self.max_iterations = max_iterations
        self._tools: dict[str, _Tool] = {}

    # ── 工具注册 ──────────────────────────────────

    def add_output_tool(
        self,
        name: str,
        schema: type[BaseModel],
        description: str,
    ) -> None:
        """注册一个结构化产出工具。

        Args:
            name: 工具名。
            schema: 产出结构的 pydantic 模型。
            description: 工具说明。

        Raises:
            ValueError: 工具名与已注册工具重复。
        """
        self._register(StructuredOutputTool(name=name, schema=schema, description=description))

    def add_function_tool(
        self,
        name: str,
        parameters: dict[str, Any],
        description: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """注册一个通用动作工具。

        Args:
            name: 工具名。
            parameters: 工具参数的原始 JSON schema。
            description: 工具说明。
            handler: 处理工具调用的可调用对象，入参为参数字典，返回值反馈给模型。

        Raises:
            ValueError: 工具名与已注册工具重复。
        """
        self._register(
            FunctionTool(name=name, parameters=parameters, description=description, handler=handler)
        )

    def _register(self, tool: _Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name!r}")
        self._tools[tool.name] = tool

    # ── 运行 ──────────────────────────────────────

    def run_core(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        tools: dict[str, _Tool] | None = None,
    ) -> AgentCoreResult:
        """执行一次基础 tool loop。

        Args:
            messages: 当前对话历史。调用方可传入全新任务，也可在已有历史后继续追加一轮。
            temperature: 本轮 loop 的采样温度。
            tools: 本轮允许调用的工具集合；默认为当前 agent 已注册工具。

        Returns:
            本次 loop 的 :class:`AgentCoreResult`。
        """
        history = list(messages)
        tool_registry = tools or self._tools
        tools_schema = self._build_tools_schema(tool_registry)
        outputs: list[StructuredOutput] = []
        intermediate_texts: list[str] = []
        final_text = ""
        iterations = 0
        max_reached = False

        while True:
            if iterations >= self.max_iterations:
                max_reached = True
                logger.warning("达到最大工具调用轮数 %d，提前结束循环", self.max_iterations)
                break
            iterations += 1
            result = self.client.chat(
                history,
                tools=tools_schema or None,
                temperature=temperature,
            )
            history.append(result.to_assistant_message())
            if not result.tool_calls:
                final_text = result.text
                break
            if result.text:
                intermediate_texts.append(result.text)
            for tool_call in result.tool_calls:
                feedback = self._handle_tool_call(tool_call, outputs, tool_registry)
                history.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": feedback}
                )

        return AgentCoreResult(
            final_text=final_text,
            outputs=tuple(outputs),
            intermediate_texts=tuple(intermediate_texts),
            iterations=iterations,
            max_iterations_reached=max_reached,
            messages=tuple(history),
        )

    def run(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        query_status: bool = True,
    ) -> AgentResult:
        """执行一个任务：驱动工具调用循环，收集产出，并自报任务状态。

        Args:
            user_prompt: 本次任务的指令。
            temperature: 主任务循环的采样温度。
            query_status: 是否在循环结束后追加一轮以索取结构化任务状态；
                置为 False 可省去一次请求，此时 ``status`` 为 :attr:`AgentStatus.UNKNOWN`。

        Returns:
            本次运行的 :class:`AgentResult`。
        """
        initial_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        core_result = self.run_core(initial_messages, temperature=temperature)

        status, status_detail = AgentStatus.UNKNOWN, ""
        result_messages = core_result.messages
        if query_status:
            status, status_detail, result_messages = self._query_status(core_result.messages)

        return AgentResult(
            status=status,
            status_detail=status_detail,
            final_text=core_result.final_text,
            outputs=core_result.outputs,
            intermediate_texts=core_result.intermediate_texts,
            iterations=core_result.iterations,
            max_iterations_reached=core_result.max_iterations_reached,
            messages=result_messages,
        )

    # ── 内部实现 ──────────────────────────────────

    def _build_tools_schema(self, tools: dict[str, _Tool]) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for tool in tools.values():
            if isinstance(tool, StructuredOutputTool):
                parameters = tool.schema.model_json_schema()
            else:
                parameters = tool.parameters
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                    },
                }
            )
        return schemas

    def _handle_tool_call(
        self,
        tool_call: RawToolCall,
        outputs: list[StructuredOutput],
        tools: dict[str, _Tool],
    ) -> str:
        tool = tools.get(tool_call.name)
        if tool is None:
            return self._feedback(ok=False, error=f"未知工具 {tool_call.name!r}")
        arguments = self._parse_arguments(tool_call.arguments)
        if arguments is None:
            return self._feedback(ok=False, error="参数必须是合法的 JSON 对象")
        if isinstance(tool, StructuredOutputTool):
            return self._handle_output_tool(tool, arguments, outputs)
        return self._handle_function_tool(tool, arguments)

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any] | None:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _handle_output_tool(
        self,
        tool: StructuredOutputTool,
        arguments: dict[str, Any],
        outputs: list[StructuredOutput],
    ) -> str:
        try:
            value = tool.schema.model_validate(arguments)
        except ValidationError as exc:
            # Feed the validation error back as a tool result so the model can
            # self-heal on the next turn instead of failing the whole task.
            return self._feedback(ok=False, error=f"结构校验失败: {exc}")
        outputs.append(StructuredOutput(tool_name=tool.name, value=value))
        count = sum(1 for output in outputs if output.tool_name == tool.name)
        return self._feedback(ok=True, message=f"已记录第 {count} 条 {tool.name}。")

    def _handle_function_tool(
        self,
        tool: FunctionTool,
        arguments: dict[str, Any],
    ) -> str:
        try:
            ret = tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - handler 为外部注入，需隔离异常并反馈给模型
            logger.warning("工具 %s 执行抛出异常", tool.name, exc_info=True)
            return self._feedback(ok=False, error=f"工具执行出错: {exc}")
        return self._serialize_tool_result(ret)

    def _query_status(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[AgentStatus, str, tuple[dict[str, Any], ...]]:
        query = [*messages, {"role": "user", "content": _STATUS_QUERY_PROMPT}]
        try:
            result = self.run_core(
                query,
                temperature=0.0,
                tools={
                    _STATUS_TOOL_NAME: StructuredOutputTool(
                        name=_STATUS_TOOL_NAME,
                        schema=TaskStatusReport,
                        description="用此工具汇报当前任务状态，status 只能是 success / failed / paused。",
                    )
                },
            )
        except LLMResponseError:
            logger.warning("任务状态查询失败，标记为 unknown", exc_info=True)
            return AgentStatus.UNKNOWN, "", tuple(query)
        reports = result.by_tool(_STATUS_TOOL_NAME)
        if not reports:
            return AgentStatus.UNKNOWN, "", result.messages
        report = reports[-1]
        return AgentStatus(report.status), report.detail, result.messages

    @staticmethod
    def _serialize_tool_result(value: Any) -> str:
        if value is None:
            return json.dumps({"ok": True}, ensure_ascii=False)
        if isinstance(value, str):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _feedback(*, ok: bool, message: str = "", error: str = "") -> str:
        payload: dict[str, Any] = {"ok": ok}
        if message:
            payload["message"] = message
        if error:
            payload["error"] = error
        return json.dumps(payload, ensure_ascii=False)


# ── 离线自测 ──────────────────────────────────────────
if __name__ == "__main__":
    # 用脚本化的伪客户端驱动工具调用循环，验证产出收集、结构自愈与状态自报，
    # 全程无需真实 API key 或网络。
    from api import LLMChatResult

    logging.basicConfig(level=logging.INFO)

    class _DemoStage(BaseModel):
        name: str
        summary: str

    def _call(call_id: str, name: str, arguments: dict[str, Any]) -> RawToolCall:
        return RawToolCall(id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))

    def _turn(text: str, calls: tuple[RawToolCall, ...]) -> LLMChatResult:
        return LLMChatResult(
            text=text,
            reasoning="",
            tool_calls=calls,
            message={"role": "assistant", "content": text},
            finish_reason="tool_calls" if calls else "stop",
        )

    class _ScriptedClient:
        """按预设脚本回放的伪客户端，仅实现 Agent 依赖的 chat。"""

        def __init__(self, turns: list[LLMChatResult]) -> None:
            self._turns = list(turns)

        def chat(self, messages: list[dict], **_: Any) -> LLMChatResult:
            return self._turns.pop(0)

    scripted = _ScriptedClient(
        turns=[
            # 轮 1：并行两条产出，第二条缺 summary，应触发校验失败并被反馈。
            _turn(
                "开始分析人物的人生阶段。",
                (
                    _call("c1", "output_life_stage", {"name": "少年", "summary": "蜀中读书任侠。"}),
                    _call("c2", "output_life_stage", {"name": "青年"}),
                ),
            ),
            # 轮 2：补回合法的青年阶段（自愈），并追加晚年阶段。
            _turn(
                "",
                (
                    _call("c3", "output_life_stage", {"name": "青年", "summary": "仗剑去国辞亲远游。"}),
                    _call("c4", "output_life_stage", {"name": "晚年", "summary": "飘零江湖终老当涂。"}),
                ),
            ),
            # 轮 3：不再调用工具，收尾。
            _turn("已输出三个主要人生阶段。", ()),
            # 轮 4：状态汇报，调用临时状态工具。
            _turn(
                "",
                (_call("c5", _STATUS_TOOL_NAME, {"status": "success", "detail": ""}),),
            ),
            # 轮 5：状态汇报收尾。
            _turn("", ()),
        ],
    )

    agent = Agent("你是一个历史人物分析专家。", client=scripted)
    agent.add_output_tool("output_life_stage", _DemoStage, "用此工具输出人物的一个人生阶段。")
    result = agent.run("请分析李白经历了哪些主要人生阶段，每个阶段一次工具调用。")

    stages = result.by_tool("output_life_stage")
    print(f"状态: {result.status.value} ({result.status_detail or '无补充'})")
    print(f"最终回复: {result.final_text}")
    print(f"中间正文: {list(result.intermediate_texts)}")
    print(f"实际轮数: {result.iterations}，被截断: {result.max_iterations_reached}")
    print(f"收集到 {len(stages)} 条人生阶段:")
    for index, stage in enumerate(stages, 1):
        data = stage.model_dump()
        print(f"  {index}. {data['name']} — {data['summary']}")

    assert len(stages) == 3, f"应收集 3 条合法产出，实际 {len(stages)}"
    assert result.status is AgentStatus.SUCCESS, result.status
    assert result.final_text == "已输出三个主要人生阶段。"
    assert result.iterations == 3, result.iterations
    print("\n离线自测通过。")
