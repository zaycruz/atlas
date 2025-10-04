"""Simple chat UI for Atlas - natural conversation interface."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Deque
from collections import deque

from rich.align import Align
from rich.console import Console, RenderableType, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table


@dataclass
class ChatMessage:
    """Simple message in the chat history."""
    content: str
    is_user: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    streaming: bool = False  # True if still being streamed


class SimpleChatUI:
    """Clean, simple chat interface focused on conversation flow."""
    
    def __init__(self, console: Console, max_messages: int = 50):
        self.console = console
        self.max_messages = max_messages
        self.messages: Deque[ChatMessage] = deque(maxlen=max_messages)
        self.current_response: Optional[ChatMessage] = None
        self._live = None
        
        # Simple stats for the sidebar
        self.memory_stats: Dict[str, Any] = {}
        self.test_mode = False
        self.status = "Ready"
        
        # Thinking animation
        self._thinking_thread: Optional[threading.Thread] = None
        self._thinking_stop = threading.Event()
        self._thinking_message = ""

    def register_live(self, live) -> None:
        """Register the Live display for updates."""
        self._live = live

    def refresh(self) -> None:
        """Refresh the display."""
        if self._live is not None:
            self._live.update(self.render())

    def add_user_message(self, content: str) -> None:
        """Add a user message to the chat."""
        message = ChatMessage(content=content, is_user=True)
        self.messages.append(message)
        self.refresh()

    def start_assistant_response(self) -> ChatMessage:
        """Start a new assistant response that will be streamed."""
        self.current_response = ChatMessage(content="", is_user=False, streaming=True)
        self.messages.append(self.current_response)
        self.refresh()
        return self.current_response

    def append_to_response(self, chunk: str) -> None:
        """Append a chunk to the current streaming response."""
        if self.current_response:
            self.current_response.content += chunk
            self.refresh()

    def finish_response(self) -> None:
        """Mark the current response as complete."""
        if self.current_response:
            self.current_response.streaming = False
            self.current_response = None
            self.refresh()

    def start_thinking(self, message: str = "Atlas is thinking") -> None:
        """Start thinking animation."""
        self.stop_thinking()
        self._thinking_stop.clear()
        self._thinking_message = message

        def worker() -> None:
            dots = ["", ".", "..", "..."]
            idx = 0
            while not self._thinking_stop.is_set():
                self.status = f"{message}{dots[idx % len(dots)]}"
                self.refresh()
                idx += 1
                time.sleep(0.5)

        self._thinking_thread = threading.Thread(target=worker, daemon=True)
        self._thinking_thread.start()

    def stop_thinking(self) -> None:
        """Stop thinking animation."""
        self._thinking_stop.set()
        if self._thinking_thread and self._thinking_thread.is_alive():
            self._thinking_thread.join(timeout=0.2)
        self._thinking_thread = None
        self.status = "Ready"
        self.refresh()

    def update_memory_stats(self, stats: Dict[str, Any]) -> None:
        """Update memory statistics."""
        self.memory_stats = stats
        self.refresh()

    def set_test_mode(self, enabled: bool) -> None:
        """Set test mode status."""
        self.test_mode = enabled
        self.refresh()

    def render(self) -> RenderableType:
        """Render the chat interface."""
        layout = Layout()
        
        # Main layout: chat area + small sidebar
        layout.split_row(
            Layout(self._render_chat(), ratio=4),
            Layout(self._render_sidebar(), ratio=1),
        )
        
        return layout

    def _render_chat(self) -> RenderableType:
        """Render the main chat area."""
        if not self.messages:
            welcome = Text("Welcome to Atlas! Type a message to start chatting.", 
                          style="dim", justify="center")
            return Panel(
                Align.center(welcome, vertical="middle"),
                title="Atlas Chat",
                border_style="cyan",
                box=box.ROUNDED
            )

        # Render messages as a flowing conversation
        chat_content = []
        
        for message in self.messages:
            if message.is_user:
                # User message - right aligned, green
                text = Text(f"You: {message.content}", style="bold green")
                chat_content.append(text)
            else:
                # Assistant message - left aligned, white/cyan
                prefix = "Atlas: "
                content = message.content if message.content else "(thinking...)"
                
                if message.streaming:
                    # Add cursor for streaming messages
                    content += "▊"
                    style = "cyan"
                else:
                    style = "white"
                
                text = Text(f"{prefix}{content}", style=style)
                chat_content.append(text)
            
            # Add a small gap between messages
            chat_content.append(Text(""))

        # Create the chat panel
        chat_group = Group(*chat_content)
        
        return Panel(
            chat_group,
            title="Atlas Chat",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def _render_sidebar(self) -> RenderableType:
        """Render a minimal sidebar with essential info."""
        sidebar_layout = Layout()
        
        sidebar_layout.split_column(
            Layout(self._render_status(), size=3),
            Layout(self._render_memory_summary(), ratio=1),
        )
        
        return sidebar_layout

    def _render_status(self) -> RenderableType:
        """Render current status."""
        status_content = []
        
        # Current status
        status_content.append(Text(self.status, style="cyan"))
        
        # Test mode indicator
        if self.test_mode:
            status_content.append(Text("⚠️ TEST MODE", style="bold yellow"))
        
        return Panel(
            Group(*status_content),
            title="Status",
            border_style="blue",
            box=box.ROUNDED
        )

    def _render_memory_summary(self) -> RenderableType:
        """Render memory stats summary."""
        if not self.memory_stats:
            return Panel(
                Text("Memory not initialized", style="dim"),
                title="Memory",
                border_style="magenta",
                box=box.ROUNDED
            )

        content = []
        
        # Working memory context usage
        working_stats = self.memory_stats.get('working_memory', {})
        if working_stats:
            turns = working_stats.get('turns', 0)
            capacity_pct = working_stats.get('capacity_pct', 0)
            
            # Simple visual indicator
            if capacity_pct > 90:
                indicator = "🔴"
                style = "red"
            elif capacity_pct > 75:
                indicator = "🟡"
                style = "yellow"
            else:
                indicator = "🟢"
                style = "green"
                
            content.append(Text(f"{indicator} Context: {capacity_pct:.0f}%", style=style))
        
        # Memory layers
        episodic = self.memory_stats.get('episodic_count', 0)
        semantic = self.memory_stats.get('semantic_count', 0)
        reflections = self.memory_stats.get('reflections_count', 0)
        
        if episodic or semantic or reflections:
            content.append(Text(""))  # Spacing
            content.append(Text(f"Episodes: {episodic}", style="cyan"))
            content.append(Text(f"Facts: {semantic}", style="green"))
            content.append(Text(f"Insights: {reflections}", style="yellow"))

        return Panel(
            Group(*content) if content else Text("No data", style="dim"),
            title="Memory",
            border_style="magenta",
            box=box.ROUNDED
        )
