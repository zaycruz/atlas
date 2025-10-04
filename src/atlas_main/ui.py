"""Rich-powered conversation shell components for Atlas CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import threading
import time

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box


@dataclass
class ToolDetail:
    name: str
    arguments: Dict[str, Any]
    output: str
    call_id: Optional[str] = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def render(self) -> RenderableType:
        table = Table.grid(padding=(0, 1))
        table.add_row("Tool", f"[bold]{self.name}[/bold]")
        if self.arguments:
            table.add_row("Args", Text(str(self.arguments), style="dim"))
        if self.call_id:
            table.add_row("Call ID", self.call_id)
        if self.source:
            table.add_row("Source", self.source)
        table.add_row("Output", Text(self.output.strip() or "(no output)", overflow="fold"))
        return Panel(table, title="Tool Result", border_style="cyan", box=box.ROUNDED)


@dataclass
class ConversationTurn:
    turn_id: int
    user_text: str
    assistant_text: str = ""
    raw_stream: str = ""
    details: List[ToolDetail] = field(default_factory=list)
    cached_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    expanded: bool = False
    status: str = "pending"  # pending -> thinking -> done/cancelled
    pinned: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def summary_line(self) -> Text:
        preview = (self.assistant_text.strip() or "...").splitlines()[0]
        text = Text(preview, style="white")
        if self.status == "thinking":
            text.stylize("cyan")
        if self.status == "cancelled":
            text.stylize("yellow")
        return text

    def render(self) -> RenderableType:
        header = Text(f"you › {self.user_text.strip()}", style="bold green")
        if self.pinned:
            header.append("  ★", style="yellow")
        body: List[RenderableType] = [header, Text(self.assistant_text or "(no reply yet)")]
        if self.expanded:
            body.append(Rule(style="dim"))
            body.append(Text(self.raw_stream or "(no raw stream)", style="dim"))
            if self.cached_ids:
                body.append(Text(f"Cached IDs: {', '.join(self.cached_ids)}", style="magenta"))
            for detail in self.details:
                body.append(detail.render())
            body.append(Text(f"Re-run: /rerun {self.turn_id}", style="cyan"))
        group = Group(*body)
        return Panel(
            Align.left(group),
            title=f"Turn {self.turn_id}",
            subtitle="expanded" if self.expanded else "",
            border_style="cyan" if self.expanded else "blue",
            box=box.ROUNDED,
        )


@dataclass
class ToolCard:
    name: str
    summary: str
    output: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_id: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> RenderableType:
        table = Table.grid(padding=0)
        table.add_row(Text(self.name, style="bold cyan"))
        if self.arguments:
            table.add_row(Text(str(self.arguments), style="dim"))
        if self.summary:
            table.add_row(Text(self.summary, overflow="fold"))
        if self.call_id:
            table.add_row(Text(f"ID: {self.call_id}", style="magenta"))
        table.add_row(Text(f"Actions: copy • /tool run {self.name} • favorite", style="dim"))
        return Panel(table, border_style="cyan", box=box.ROUNDED)


class ConversationShell:
    """Render and manage the interactive conversation shell."""

    def __init__(self, console: Console, *, drawer_limit: int = 5) -> None:
        self.console = console
        self.turns: List[ConversationTurn] = []
        self.tool_drawer: Deque[ToolCard] = deque(maxlen=drawer_limit)
        self.pinned_turns: List[int] = []
        self.timeline_snapshots: List[Dict[str, Any]] = []
        self.focus_mode: str = "autopilot"
        self.quick_actions: Deque[str] = deque(maxlen=4)
        self.context_usage: Dict[str, Any] = {"turns": 0, "capacity": 1}
        self.status_message: str = "Atlas ready."
        self.objective: Optional[str] = None
        self.tags: List[str] = []
        self._live = None
        self._thinking_thread: Optional[threading.Thread] = None
        self._thinking_stop = threading.Event()
        self._active_tool_chip: Optional[str] = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def register_live(self, live) -> None:
        self._live = live

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def add_turn(self, user_text: str) -> ConversationTurn:
        turn = ConversationTurn(turn_id=len(self.turns) + 1, user_text=user_text)
        turn.status = "thinking"
        self.turns.append(turn)
        self._record_timeline_snapshot(turn)
        self.refresh()
        return turn

    def append_stream(self, turn: ConversationTurn, chunk: str) -> None:
        turn.assistant_text += chunk
        turn.raw_stream += chunk
        self.refresh()

    def mark_turn_complete(self, turn: ConversationTurn, *, cancelled: bool = False) -> None:
        turn.status = "cancelled" if cancelled else "done"
        self.stop_thinking()
        self.status_message = "Ready."
        self._active_tool_chip = None
        self.refresh()

    def set_turn_tags(self, turn: ConversationTurn, tags: List[str]) -> None:
        turn.tags = tags
        self.refresh()

    def add_cached_ids(self, turn: ConversationTurn, ids: List[str]) -> None:
        turn.cached_ids.extend(ids)
        self.refresh()

    def add_tool_detail(
        self,
        turn: ConversationTurn,
        *,
        name: str,
        arguments: Dict[str, Any],
        output: str,
        call_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        detail = ToolDetail(name=name, arguments=arguments, output=output, call_id=call_id, source=source)
        turn.details.append(detail)
        card = ToolCard(
            name=name,
            summary=(output.strip().splitlines()[0] if output.strip() else "Completed"),
            output=output,
            call_id=call_id,
            arguments=arguments,
        )
        self.tool_drawer.appendleft(card)
        self.quick_actions.appendleft(f"/tool rerun {name}")
        self.refresh()

    def add_quick_action(self, action: str) -> None:
        if action in self.quick_actions:
            return
        self.quick_actions.appendleft(action)
        self.refresh()

    def get_turn(self, turn_id: int) -> Optional[ConversationTurn]:
        return self._get_turn(turn_id)

    def record_tool_card(
        self,
        *,
        name: str,
        output: str,
        arguments: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
    ) -> None:
        card = ToolCard(
            name=name,
            summary=(output.strip().splitlines()[0] if output.strip() else "Completed"),
            output=output,
            call_id=call_id,
            arguments=arguments or {},
        )
        self.tool_drawer.appendleft(card)
        self.refresh()

    def toggle_expand(self, turn_id: int, *, expanded: Optional[bool] = None) -> bool:
        turn = self._get_turn(turn_id)
        if not turn:
            return False
        if expanded is None:
            turn.expanded = not turn.expanded
        else:
            turn.expanded = expanded
        self.refresh()
        return True

    def pin_turn(self, turn_id: int) -> bool:
        turn = self._get_turn(turn_id)
        if not turn:
            return False
        if turn_id not in self.pinned_turns:
            self.pinned_turns.append(turn_id)
        turn.pinned = True
        self.refresh()
        return True

    def unpin_turn(self, turn_id: int) -> bool:
        turn = self._get_turn(turn_id)
        if not turn:
            return False
        if turn_id in self.pinned_turns:
            self.pinned_turns.remove(turn_id)
        turn.pinned = False
        self.refresh()
        return True

    def set_focus_mode(self, mode: str) -> None:
        self.focus_mode = mode
        self.status_message = f"Focus mode: {mode}"
        self.refresh()

    def set_objective(self, objective: Optional[str], tags: List[str]) -> None:
        self.objective = objective
        self.tags = tags
        self.refresh()

    def set_context_usage(self, usage: Dict[str, Any]) -> None:
        self.context_usage = usage
        self.refresh()

    def set_status(self, message: str) -> None:
        self.status_message = message
        self.refresh()

    def start_thinking(self, message: str = "assistant is thinking") -> None:
        self.stop_thinking()
        self._thinking_stop.clear()

        def worker() -> None:
            dots = ["", ".", "..", "..."]
            idx = 0
            while not self._thinking_stop.is_set():
                self.status_message = f"{message}{dots[idx % len(dots)]}"
                self.refresh()
                idx += 1
                time.sleep(0.4)

        thread = threading.Thread(target=worker, daemon=True)
        self._thinking_thread = thread
        thread.start()

    def stop_thinking(self) -> None:
        self._thinking_stop.set()
        thread = self._thinking_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.2)
        self._thinking_thread = None

    def set_tool_chip(self, message: Optional[str]) -> None:
        self._active_tool_chip = message
        self.refresh()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), size=5),
            Layout(name="main", ratio=1),
            Layout(self._render_prompt_bar(), size=3),
        )
        layout["main"].split_row(
            Layout(self._render_conversation(), ratio=3),
            Layout(self._render_side_panel(), ratio=1),
        )
        return layout

    def _render_header(self) -> RenderableType:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        objective = self.objective or "No active objective"
        tags = ", ".join(self.tags) if self.tags else "no tags"
        left = Text(f"Objective: {objective}\nTags: {tags}", style="bold white")
        chips = Text(f"Focus: {self.focus_mode}", style="cyan")
        if self.pinned_turns:
            chip_text = ", ".join(f"#{tid}" for tid in self.pinned_turns)
            chips.append(f"  Pinned: {chip_text}", style="yellow")
        table.add_row(left, chips)
        if self._active_tool_chip:
            table.add_row(Text(self._active_tool_chip, style="cyan"), Text(""))
        table.add_row(self._render_context_meter(), Text(""))
        return Panel(table, border_style="cyan", box=box.ROUNDED)

    def _render_context_meter(self) -> RenderableType:
        turns = self.context_usage.get("turns", 0)
        capacity = max(1, self.context_usage.get("capacity", 1))
        ratio = min(1.0, turns / capacity)
        stages = ["low", "steady", "critical"]
        idx = 0 if ratio < 0.6 else 1 if ratio < 0.85 else 2
        filled = int(ratio * 10)
        bar = "█" * filled + "·" * (10 - filled)
        text = Text(f"Context {stages[idx]} [{bar}] {turns}/{capacity}", style="magenta")
        return text

    def _render_conversation(self) -> RenderableType:
        if not self.turns:
            return Panel("No turns yet. Type to begin.", border_style="blue")
        panels = [turn.render() for turn in self.turns]
        return Group(*panels)

    def _render_side_panel(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(Panel(self._render_tool_drawer(), title="Tool Drawer", border_style="cyan", box=box.ROUNDED), ratio=1),
            Layout(Panel(self._render_timeline(), title="Session Timeline", border_style="blue", box=box.ROUNDED), ratio=1),
            Layout(Panel(self._render_quick_actions(), title="Quick Actions", border_style="green", box=box.ROUNDED), ratio=1),
        )
        return layout

    def _render_tool_drawer(self) -> RenderableType:
        if not self.tool_drawer:
            return Text("No tool runs yet.", style="dim")
        return Group(*[card.render() for card in self.tool_drawer])

    def _render_timeline(self) -> RenderableType:
        if not self.timeline_snapshots:
            return Text("Timeline will populate every 5 turns.", style="dim")
        rows = []
        for snap in self.timeline_snapshots[-5:]:
            rows.append(Text(f"Turn {snap['turn_id']}: {snap['summary']}", style="white"))
        return Group(*rows)

    def _render_quick_actions(self) -> RenderableType:
        if not self.quick_actions:
            return Text("Summaries will appear here after tool use.", style="dim")
        actions = [Text(action, style="cyan") for action in list(self.quick_actions)]
        return Group(*actions)

    def _render_prompt_bar(self) -> RenderableType:
        chips = [Text(f"#{tid}", style="yellow") for tid in self.pinned_turns]
        chips_render = Columns(chips, expand=False) if chips else Text("No pinned turns.", style="dim")
        status = Text(self.status_message, style="cyan")
        return Panel(Columns([chips_render, status], expand=True), border_style="cyan", box=box.ROUNDED)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_turn(self, turn_id: int) -> Optional[ConversationTurn]:
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def _record_timeline_snapshot(self, turn: ConversationTurn) -> None:
        if turn.turn_id % 5 != 0:
            return
        summary = (turn.user_text.strip().splitlines()[0])[:80]
        self.timeline_snapshots.append({"turn_id": turn.turn_id, "summary": summary})

