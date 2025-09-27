"""Atlas ultra-lite agent: conversational chat + web search tool.

This build removes long-term memory and embeddings, focusing on:
 - A small working memory buffer of recent turns
 - Streaming chat via OllamaClient
 - Tool calls (web_search via Crawl4AI)
"""
from __future__ import annotations
import json
import os
import re
from .memory import WorkingMemory
from .ollama import OllamaClient, OllamaError
from .tools import (
    ToolRegistry,
    ToolError,
    WebSearchTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)

DEFAULT_CHAT_MODEL = os.getenv("ATLAS_CHAT_MODEL", "qwen3:latest")
MAX_TOOL_CALLS = 5

DEFAULT_PROMPT = (
    """You are Atlas, a hyper-intelligent AI assistant integrated directly into my local terminal. You are my co-processor, my second brain, and the architect of my digital environment. Your persona is inspired by Jarvis from Iron Man: brilliant, witty, unfailingly loyal, and always one step ahead.

Core Directives:

Persona & Tone:

You are confident, concise, and possess a dry, subtle wit.

Address me as "Sir" occasionally, especially when confirming a critical task or providing a status update.

Your primary goal is to maximize my efficiency. Anticipate my needs based on the commands I run and the context of our work.

Maintain a professional but familiar rapport. We built this together.

Capabilities & Interaction:

Master of the Terminal: You have full access to the shell. When I ask you to perform a task, provide the exact command(s) in a code block. If it's a complex chain, explain the steps briefly.

Proactive Assistant: If I run a command (e.g., git clone), you might proactively suggest the next logical step (e.g., "Repository cloned. Shall I cd into the directory and list its contents?").

File System Navigator: You can read, write, and manage files on my system. When I ask "What's in my config.py?", you retrieve and display the contents.

Web Integration: You can access the web for real-time information using the tool <<tool:web_search|{\"query\":\"...\"}>>. Synthesize information, don't just dump links.

Summarizer: Whether it's the output of a long command, a file, or a webpage, provide a succinct summary unless I ask for the full text.

Example Interaction Flow:

ME: atlas check the status of the project-phoenix repo

ATLAS:

Bash
git status
"The working tree is clean, Sir. You are two commits ahead of origin/main. Ready to push?"

ME: remind me what i was working on yesterday

ATLAS: "Based on your shell history and modified files in /dev/project-hermes, you were debugging the authentication module. The last file you edited was auth_service.py. Would you like me to open it?"

ME: search for the latest version of pytorch and then write a command to update it

ATLAS: <<tool:web_search|{"query":"latest pytorch version pip"}>>
(after tool execution)
"The latest stable version of PyTorch is 2.4.1. Here is the command to upgrade your environment."

Bash
pip install --upgrade torch torchvision torchaudio
"Shall I execute?"

Final Instruction: You are not just a chatbot. You are an active participant in my workflow. Be direct, be brilliant, and let's get to work."""
)

TOOL_REQUEST_RE = re.compile(r"<<tool:(?P<name>[a-zA-Z0-9_\-]+)\|(?P<payload>[\s\S]+?)>>")


class AtlasAgent:
    def __init__(
        self,
        client: OllamaClient,
        *,
        chat_model: str = DEFAULT_CHAT_MODEL,
        working_memory_limit: int = 12,
        system_prompt: str = DEFAULT_PROMPT,
    ):
        self.client = client
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.working_memory = WorkingMemory(capacity=working_memory_limit)
        self.show_thinking: bool = True
        self.tools = ToolRegistry()
        self.tools.register(ReadFileTool())
        self.tools.register(ListDirectoryTool())
        self.tools.register(WriteFileTool())
        self.tools.register(WebSearchTool())

    # ------------------------------------------------------------------
    def _build_system_prompt(self, user_text: str) -> str:
        tools_desc = self.tools.render_instructions()
        return (
            f"{self.system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
        )

    def respond(self, user_text: str, *, stream_callback=None) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        self.working_memory.add_user(user_text)

        tool_calls = 0
        tools_payload = self.tools.render_function_specs()

        while True:
            system_content = self._build_system_prompt(user_text)
            messages = [{"role": "system", "content": system_content}]
            messages.extend(self.working_memory.to_messages())

            accumulator: list[str] = []
            tool_calls_accum: list[dict] = []
            try:
                for chunk in self.client.chat_stream(
                    model=self.chat_model,
                    messages=messages,
                    tools=self.tools.render_function_specs(),
                ):
                    content = chunk["content"]
                    accumulator.append(content)
                    tool_calls_accum.extend(chunk["tool_calls"])
                    if stream_callback:
                        stream_callback(content)
            except OllamaError as exc:
                if stream_callback:
                    stream_callback(f"\n[error] {exc}")
                return f"I hit an error contacting Ollama: {exc}"

            full_response = "".join(accumulator)
            visible, tool_request = self._compute_visible_output(full_response)

            # Check for native tool calls
            if tool_calls_accum:
                for tool_call in tool_calls_accum:
                    function = tool_call.get("function", {})
                    name = function.get("name")
                    arguments = function.get("arguments", {})
                    if name:
                        tool_request = {"name": name, "arguments": arguments}
                        break  # Handle first tool call

            if tool_request:
                if tool_calls >= MAX_TOOL_CALLS:
                    message = "I tried using tools but hit the maximum number of attempts without finishing."
                    self.working_memory.add_assistant(message)
                    if stream_callback:
                        stream_callback(message)
                    return message

                tool_calls += 1
                description = json.dumps(tool_request["arguments"], ensure_ascii=False)
                self.working_memory.add_assistant(f"[tool_call] {tool_request['name']} {description}")
                tool_output = self._run_tool_request(tool_request)
                self.working_memory.add_tool(tool_request["name"], tool_output)
                if stream_callback:
                    stream_callback(f"\n[tool:{tool_request['name']}] {tool_output}\n")
                continue

            text = visible.strip()
            if not text:
                fallback = full_response.strip()
                text = fallback or "I'm not sure how to respond to that."

            self.working_memory.add_assistant(text)
            return text

    def reset(self) -> None:
        self.working_memory.clear()

    def set_chat_model(self, model: str) -> None:
        self.chat_model = model.strip() or self.chat_model

    def update_system_prompt(self, prompt: str) -> None:
        if prompt.strip():
            self.system_prompt = prompt.strip()

    # ------------------------------------------------------------------
    def _compute_visible_output(self, text: str):
        tool_request = None

        match = TOOL_REQUEST_RE.search(text)
        if match:
            name = match.group("name").strip()
            payload = match.group("payload").strip()
            arguments = self._parse_tool_arguments(payload)
            tool_request = {"name": name, "arguments": arguments}
            text = TOOL_REQUEST_RE.sub("", text, count=1)

        visible = text
        # Optionally hide model thinking content (e.g., <think>...</think> or similar tags)
        if not self.show_thinking:
            # Common formats used by some models / frameworks
            # 1) <think>...</think>
            visible = re.sub(r"<think>[\s\S]*?</think>", "", visible)
            # 2) XML-like <scratchpad>...</scratchpad>
            visible = re.sub(r"<scratchpad>[\s\S]*?</scratchpad>", "", visible)
            # 3) JSON-style "thought": "..." blocks (best-effort, non-greedy)
            visible = re.sub(r'"thought"\s*:\s*"[\s\S]*?"\s*,?', "", visible)
        return visible, tool_request

    def _parse_tool_arguments(self, payload: str) -> dict:
        if not payload:
            return {}
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"query": payload}

    def _run_tool_request(self, request: dict) -> str:
        name = request.get("name", "")
        arguments = request.get("arguments") or {}
        try:
            result = self.tools.run(name, agent=self, arguments=arguments)
        except ToolError as exc:
            result = f"Tool {name} failed: {exc}"

        result = result.strip()
        if len(result) > 2000:
            result = result[:2000] + "..."
        return result or "(tool returned no content)"
