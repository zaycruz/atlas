"""Rule-first extraction of knowledge graph nodes from memory entries."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


FILE_PATTERN = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9]+)")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_/-]{3,})")
TICKET_PATTERN = re.compile(r"([A-Z]{2,10}-\d{1,6})")

DECISION_CUES = ("decided to", "we will", "let's", "should", "will plan", "resolve to")
TASK_CUES = ("todo", "next", "plan", "should", "need to", "let's", "remember to")


@dataclass
class ExtractedNode:
    key: str
    type: str
    label: str
    props: Dict[str, str]


@dataclass
class ExtractedEdge:
    source: str
    target: str
    type: str
    props: Dict[str, str]


@dataclass
class MemoryPayload:
    id: str
    kind: str  # semantic | reflection | episodic
    text: str
    tags: Iterable[str]
    ts: float
    source: Optional[str] = None


@dataclass
class ExtractionResult:
    nodes: List[ExtractedNode]
    edges: List[ExtractedEdge]


class MemoryGraphExtractor:
    """Turn memory items into graph nodes/edges using rules."""

    def __init__(self) -> None:
        pass

    def extract(self, payload: MemoryPayload) -> ExtractionResult:
        nodes: Dict[str, ExtractedNode] = {}
        edges: List[ExtractedEdge] = []

        def upsert_node(key: str, n_type: str, label: str, props: Optional[Dict[str, str]] = None) -> None:
            if key in nodes:
                return
            nodes[key] = ExtractedNode(key=key, type=n_type, label=label, props=props or {})

        mem_key = f"memory:{payload.id}"
        upsert_node(
            mem_key,
            "Memory",
            label=_truncate(payload.text, 80),
            props={
                "kind": payload.kind,
                "ts": str(payload.ts),
                "text": payload.text,
                **({"source": payload.source} if payload.source else {}),
            },
        )

        tags = _normalize_tags(payload.tags)
        topics = set(tags)
        topics.update(tag.strip().lower() for tag in HASHTAG_PATTERN.findall(payload.text))
        for topic in sorted(topics):
            if not topic:
                continue
            label = topic.replace("_", " ").title()
            key = f"topic:{topic}"
            upsert_node(key, "Topic", label, props={"slug": topic})
            edges.append(ExtractedEdge(source=mem_key, target=key, type="mentions", props={}))

        files = _extract_files(payload.text)
        for file_path in files:
            key = f"file:{file_path}"
            upsert_node(key, "File", file_path, props={"path": file_path})
            edges.append(ExtractedEdge(source=mem_key, target=key, type="references", props={}))
            project_slug = _project_from_path(file_path)
            if project_slug:
                project_key = f"project:{project_slug}"
                upsert_node(project_key, "Project", project_slug, props={"slug": project_slug})
                edges.append(ExtractedEdge(source=key, target=project_key, type="belongs_to", props={}))

        projects = _extract_projects(payload.text, tags)
        for project_slug in projects:
            key = f"project:{project_slug}"
            upsert_node(key, "Project", project_slug, props={"slug": project_slug})
            edges.append(ExtractedEdge(source=mem_key, target=key, type="mentions", props={}))

        ticket_ids = set(TICKET_PATTERN.findall(payload.text))
        for ticket in ticket_ids:
            key = f"project:{ticket.lower()}"
            upsert_node(key, "Project", ticket.upper(), props={"ticket": ticket.upper()})
            edges.append(ExtractedEdge(source=mem_key, target=key, type="mentions", props={}))

        lowered = payload.text.lower()
        if any(cue in lowered for cue in DECISION_CUES):
            decision_key = f"decision:{payload.id}"
            upsert_node(
                decision_key,
                "Decision",
                label=_truncate(payload.text, 80),
                props={"text": payload.text, "ts": str(payload.ts)},
            )
            edges.append(ExtractedEdge(source=mem_key, target=decision_key, type="decides", props={}))
            for topic in topics:
                edges.append(ExtractedEdge(source=decision_key, target=f"topic:{topic}", type="about", props={}))
        if any(cue in lowered for cue in TASK_CUES):
            task_key = f"task:{payload.id}"
            upsert_node(
                task_key,
                "Task",
                label=_truncate(payload.text, 80),
                props={"text": payload.text, "ts": str(payload.ts)},
            )
            edges.append(ExtractedEdge(source=mem_key, target=task_key, type="notes", props={}))
            for topic in topics:
                edges.append(ExtractedEdge(source=task_key, target=f"topic:{topic}", type="about", props={}))

        return ExtractionResult(nodes=list(nodes.values()), edges=edges)


def _normalize_tags(tags: Iterable[str]) -> Set[str]:
    normalized: Set[str] = set()
    for tag in tags:
        text = str(tag or "").strip().lower()
        if not text:
            continue
        normalized.add(text)
    return normalized


def _extract_files(text: str) -> Set[str]:
    matches = FILE_PATTERN.findall(text)
    sanitized: Set[str] = set()
    for match in matches:
        cleaned = match.strip()
        if cleaned:
            sanitized.add(_normalize_path(cleaned))
    return sanitized


def _normalize_path(path: str) -> str:
    collapsed = re.sub(r"/+", "/", path)
    collapsed = collapsed.strip(".")
    return collapsed


def _project_from_path(path: str) -> Optional[str]:
    parts = path.split("/")
    if not parts:
        return None
    root = parts[0].strip().lower()
    if root and len(root) > 1 and root not in {"tmp", "var", "etc"}:
        return root
    return None


def _extract_projects(text: str, tags: Iterable[str]) -> Set[str]:
    projects: Set[str] = set()
    for tag in tags:
        if tag.startswith("project:"):
            projects.add(tag.split(":", 1)[1])
        elif tag.startswith("repo:"):
            projects.add(tag.split(":", 1)[1])
    for match in FILE_PATTERN.findall(text):
        root = _project_from_path(match)
        if root:
            projects.add(root)
    return {proj.strip().lower() for proj in projects if proj}


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


__all__ = [
    "MemoryPayload",
    "ExtractionResult",
    "MemoryGraphExtractor",
]
