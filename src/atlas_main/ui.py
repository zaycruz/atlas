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
        
        # Memory tracking
        self.memory_events: Deque[Dict[str, Any]] = deque(maxlen=20)  # Recent memory operations
        self.memory_stats: Dict[str, Any] = {
            "episodic_count": 0,
            "semantic_count": 0,
            "reflections_count": 0,
            "last_harvest": None
        }
        
        # Conversation scrolling configuration
        self.max_visible_turns = 8         # Maximum turns to show at once
        self.scroll_offset = 0             # How many turns to skip from the end
        self.auto_scroll = True            # Whether to auto-scroll to latest
        
        # Test mode tracking
        self.test_mode = False             # Whether test mode is active

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
        
        # Auto-scroll to latest turn
        if self.auto_scroll:
            self.scroll_offset = 0  # Reset to show latest messages
        self.refresh()
        return turn

    def append_stream(self, turn: ConversationTurn, chunk: str) -> None:
        turn.assistant_text += chunk
        turn.raw_stream += chunk
        # Update status to show streaming
        if turn.status == "thinking":
            turn.status = "streaming"
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
    
    def add_memory_event(self, event_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track memory operations for the UI display."""
        event = {
            "type": event_type,  # "semantic_add", "reflection_add", "episodic_store", "harvest"
            "content": content[:100] + "..." if len(content) > 100 else content,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        self.memory_events.appendleft(event)
        
        # Update stats
        if event_type == "semantic_add":
            self.memory_stats["semantic_count"] += 1
        elif event_type == "reflection_add":
            self.memory_stats["reflections_count"] += 1
        elif event_type == "episodic_store":
            self.memory_stats["episodic_count"] += 1
        elif event_type == "harvest":
            self.memory_stats["last_harvest"] = datetime.utcnow()
        
        self.refresh()
    
    def update_memory_stats(self, stats: Dict[str, Any]) -> None:
        """Update memory statistics from the agent."""
        self.memory_stats.update(stats)
        # Update test mode if present in stats
        if "test_mode" in stats:
            self.test_mode = stats["test_mode"]
        self.refresh()

    def set_status(self, message: str) -> None:
        self.status_message = message
        self.refresh()

    def set_test_mode(self, enabled: bool) -> None:
        """Update test mode status for UI display."""
        self.test_mode = enabled
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
            Layout(self._render_header(), size=4),
            Layout(name="main", ratio=1),
        )
        layout["main"].split_row(
            Layout(self._render_chat_with_input(), ratio=3),
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
        """Render conversation as a flowing chat interface."""
        if not self.turns:
            return Panel(
                Text("Start chatting with Atlas...", style="dim", justify="center"),
                title="Chat",
                border_style="blue",
                box=box.ROUNDED
            )
        
        # Create flowing chat messages
        chat_messages = []
        
        for turn in self.turns:
            # User message
            user_text = Text(f"You: {turn.user_text.strip()}", style="bold green")
            chat_messages.append(user_text)
            
            # Assistant response
            if turn.assistant_text.strip():
                if turn.status == "streaming":
                    # Show streaming response with cursor
                    response_text = Text(f"Atlas: {turn.assistant_text.strip()}", style="cyan")
                    response_text.append("▊", style="cyan")
                else:
                    # Show completed response
                    response_text = Text(f"Atlas: {turn.assistant_text.strip()}", style="white")
                chat_messages.append(response_text)
            elif turn.status == "thinking":
                # Show thinking indicator
                thinking_text = Text("Atlas: ", style="cyan")
                thinking_text.append("thinking...", style="dim cyan")
                chat_messages.append(thinking_text)
            elif turn.status == "pending":
                # Show pending indicator with cursor
                pending_text = Text("Atlas: ", style="cyan")
                pending_text.append("▊", style="cyan")
                chat_messages.append(pending_text)
            
            # Add spacing between conversations
            chat_messages.append(Text(""))
        
        # Create scrollable view
        total_turns = len(self.turns)
        if total_turns > self.max_visible_turns:
            # Show scroll indicators
            start_idx = max(0, total_turns - self.max_visible_turns - self.scroll_offset)
            end_idx = total_turns - self.scroll_offset
            
            if start_idx > 0:
                scroll_up = Text(f"↑ {start_idx} earlier messages", style="dim", justify="center")
                chat_messages.insert(0, scroll_up)
                chat_messages.insert(1, Text(""))
            
            if self.scroll_offset > 0:
                scroll_down = Text(f"↓ {self.scroll_offset} newer messages", style="dim", justify="center") 
                chat_messages.append(Text(""))
                chat_messages.append(scroll_down)
        
        return Panel(
            Group(*chat_messages),
            title="Chat",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def _render_chat_with_input(self) -> RenderableType:
        """Render chat area with integrated input at the bottom."""
        chat_layout = Layout()
        chat_layout.split_column(
            Layout(self._render_conversation(), ratio=1),
            Layout(self._render_input_area(), size=3),
        )
        return chat_layout

    def _render_input_area(self) -> RenderableType:
        """Render the input area at bottom of chat."""
        input_content = []
        
        # Status line (minimal)
        if self.status_message != "Ready.":
            input_content.append(Text(self.status_message, style="dim cyan"))
        
        # Input prompt line
        input_content.append(Text("Type your message and press Enter...", style="dim"))
        
        return Panel(
            Group(*input_content) if input_content else Text("Type your message...", style="dim"),
            title="Input",
            border_style="green",
            box=box.ROUNDED
        )

    def _render_memory_panel(self) -> RenderableType:
        """Render the memory management panel showing memory operations and stats."""
        layout = Layout()
        layout.split_column(
            Layout(Panel(self._render_memory_stats(), title="Memory Stats", border_style="magenta", box=box.ROUNDED), ratio=1),
            Layout(Panel(self._render_memory_events(), title="Memory Events", border_style="magenta", box=box.ROUNDED), ratio=2),
        )
        return layout
    
    def _render_memory_stats(self) -> RenderableType:
        """Render enhanced memory statistics with visual progress bars for context only."""
        from rich.table import Table
        
        stats = self.memory_stats
        
        # Create a table for organized display
        table = Table.grid(padding=(0, 1))
        
        # Working Memory Section with visual bars
        working_stats = stats.get('working_memory', {})
        if working_stats:
            tokens = working_stats.get('tokens', 0)
            token_pct = working_stats.get('token_pct', 0)
            
            # Working memory header
            table.add_row(Text("CONTEXT USAGE", style="bold white"))
            
            # Token usage with visual bar - this is the real constraint
            if token_pct > 0:
                token_style = "red" if token_pct > 90 else "yellow" if token_pct > 75 else "blue"
                token_bar = "█" * int(token_pct / 10) + "░" * (10 - int(token_pct / 10))
                token_k = tokens / 1000
                table.add_row(Text(f"Context: {token_k:.1f}K [{token_bar}] {token_pct:.0f}%", style=token_style))
            else:
                # Fallback to turn count if token tracking isn't available
                turns = working_stats.get('turns', 0)
                capacity_pct = working_stats.get('capacity_pct', 0)
                capacity_style = "red" if capacity_pct > 90 else "yellow" if capacity_pct > 75 else "green"
                capacity_bar = "█" * int(capacity_pct / 10) + "░" * (10 - int(capacity_pct / 10))
                table.add_row(Text(f"Turns: {turns} [{capacity_bar}] {capacity_pct:.0f}%", style=capacity_style))
            
            table.add_row(Text(""))  # Spacing
        
        # Memory Layers Section - Simple integer counts
        table.add_row(Text("MEMORY LAYERS", style="bold white"))
        
        # Test mode indicator
        if self.test_mode:
            table.add_row(Text("⚠️  TEST MODE - No memory logging", style="bold yellow"))
            table.add_row(Text(""))  # Spacing
        
        episodic_count = stats.get('episodic_count', 0)
        semantic_count = stats.get('semantic_count', 0) 
        reflections_count = stats.get('reflections_count', 0)
        
        table.add_row(Text(f"Episodes: {episodic_count}", style="cyan"))
        table.add_row(Text(f"Facts: {semantic_count}", style="green"))
        table.add_row(Text(f"Insights: {reflections_count}", style="yellow"))
        
        # Quality Gate Statistics (if available)
        quality_stats = stats.get('quality_gates', {})
        if quality_stats:
            facts_accepted = quality_stats.get('facts_accepted', 0)
            reflections_accepted = quality_stats.get('reflections_accepted', 0)
            if facts_accepted > 0 or reflections_accepted > 0:
                table.add_row(Text(""))  # Spacing
                table.add_row(Text("QUALITY GATES", style="bold white"))
                table.add_row(Text(f"Facts accepted: {facts_accepted}", style="dim green"))
                table.add_row(Text(f"Insights accepted: {reflections_accepted}", style="dim yellow"))
        
        return table

    def _render_memory_events(self) -> RenderableType:
        """Render recent memory events."""
        if not self.memory_events:
            return Text("No memory events yet", style="dim")
        
        events = []
        for event in list(self.memory_events)[:8]:  # Show last 8 events
            timestamp = event['timestamp']
            age_seconds = (datetime.utcnow() - timestamp).total_seconds()
            
            # Format age
            if age_seconds < 60:
                age_str = f"{int(age_seconds)}s"
            elif age_seconds < 3600:
                age_str = f"{int(age_seconds/60)}m"
            else:
                age_str = f"{int(age_seconds/3600)}h"
            
            # Style by event type
            type_styles = {
                "semantic_add": "green",
                "reflection_add": "yellow", 
                "episodic_store": "cyan",
                "harvest": "magenta"
            }
            style = type_styles.get(event['type'], "white")
            
            # Format event
            event_name = event['type'].replace('_', ' ').title()
            content_preview = event['content'][:40] + "..." if len(event['content']) > 40 else event['content']
            
            event_text = Text(f"{age_str} {event_name}: {content_preview}", style=style, overflow="ellipsis")
            events.append(event_text)
        
        return Group(*events)

    def _render_side_panel(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(Panel(self._render_memory_stats(), title="Memory", border_style="magenta", box=box.ROUNDED), ratio=1),
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

