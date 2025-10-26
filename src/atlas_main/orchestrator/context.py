"""Codebase context analysis utilities for planner collaboration."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GitInfo:
    status: str
    recent_commits: List[Dict[str, str]]


class CodebaseContext:
    """Generate structured repository context for planner agents."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path

    async def analyze(self) -> Dict[str, Any]:
        structure_task = asyncio.create_task(self._get_structure())
        languages_task = asyncio.create_task(self._detect_languages())
        frameworks_task = asyncio.create_task(self._detect_frameworks())
        entry_points_task = asyncio.create_task(self._find_entry_points())
        test_framework_task = asyncio.create_task(self._detect_test_framework())
        dependencies_task = asyncio.create_task(self._get_dependencies())
        git_task = asyncio.create_task(self._get_git_info())

        structure = await structure_task
        languages = await languages_task
        frameworks = await frameworks_task
        entry_points = await entry_points_task
        test_framework = await test_framework_task
        dependencies = await dependencies_task
        git_info = await git_task

        return {
            "structure": structure,
            "languages": languages,
            "frameworks": frameworks,
            "entry_points": entry_points,
            "test_framework": test_framework,
            "dependencies": dependencies,
            "git_status": git_info.status,
            "recent_commits": git_info.recent_commits,
        }

    async def _get_structure(self, *, max_depth: int = 3) -> Dict[str, Any]:
        start = self.repo_path
        result: Dict[str, Any] = {}

        async def _walk(directory: Path, depth: int) -> Dict[str, Any]:
            if depth >= max_depth:
                return {}
            tree: Dict[str, Any] = {}
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except Exception:
                return tree
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    tree[entry.name] = await _walk(entry, depth + 1)
                elif entry.is_file():
                    tree.setdefault("__files__", []).append(entry.name)
            return tree

        result[start.name] = await _walk(start, depth=0)
        return result

    async def _detect_languages(self) -> List[str]:
        extensions = set()
        for path in self._iter_files():
            extensions.add(path.suffix.lower())
        language_map = {
            ".py": "Python",
            ".ts": "TypeScript",
            ".tsx": "TypeScript React",
            ".js": "JavaScript",
            ".jsx": "JavaScript React",
            ".rs": "Rust",
            ".go": "Go",
            ".java": "Java",
            ".cs": "C#",
            ".cpp": "C++",
            ".c": "C",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".rb": "Ruby",
            ".php": "PHP",
        }
        languages = {language_map[ext] for ext in extensions if ext in language_map}
        return sorted(languages)

    async def _detect_frameworks(self) -> List[str]:
        frameworks = set()
        if (self.repo_path / "package.json").exists():
            try:
                package_data = json.loads((self.repo_path / "package.json").read_text())
            except Exception:
                package_data = {}
            deps = self._gather_node_dependencies(package_data)
            frameworks.update(self._node_to_frameworks(deps))

        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib  # type: ignore
            except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
                tomllib = None
            if tomllib:
                try:
                    data = tomllib.loads(pyproject.read_text())
                    frameworks.update(self._python_to_frameworks(data))
                except Exception:
                    pass
        return sorted(frameworks)

    async def _find_entry_points(self) -> List[str]:
        entry_candidates = [
            "main.py",
            "app.py",
            "wsgi.py",
            "manage.py",
            "index.js",
            "index.ts",
            "index.tsx",
            "main.ts",
            "main.tsx",
        ]
        matches: List[str] = []
        for candidate in entry_candidates:
            for path in self._iter_files():
                if path.name == candidate:
                    try:
                        relative = str(path.relative_to(self.repo_path))
                    except Exception:
                        relative = path.name
                    matches.append(relative)
        return sorted(set(matches))

    async def _detect_test_framework(self) -> str:
        patterns = {
            "pytest": re.compile(r"pytest", re.IGNORECASE),
            "unittest": re.compile(r"import\s+unittest"),
            "jest": re.compile(r"@jest|jest\.config"),
            "vitest": re.compile(r"vitest"),
        }
        for path in self._iter_files():
            lower = path.name.lower()
            if lower.startswith("test_") or lower.endswith(("_test.py", ".spec.ts", ".test.ts", ".test.tsx")):
                if path.suffix == ".py":
                    return "pytest"
                return "jest"
        requirements = self.repo_path / "requirements.txt"
        if requirements.exists():
            try:
                content = requirements.read_text()
                if "pytest" in content:
                    return "pytest"
            except Exception:
                pass
        return "unknown"

    async def _get_git_info(self) -> GitInfo:
        status = await self._run_git(["status", "--short"])
        log_output = await self._run_git(["log", "-10", "--pretty=format:%h|%an|%ad|%s", "--date=short"])
        commits: List[Dict[str, str]] = []
        if log_output:
            for line in log_output.splitlines():
                parts = line.split("|", maxsplit=3)
                if len(parts) == 4:
                    commits.append(
                        {
                            "hash": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "summary": parts[3],
                        }
                    )
        return GitInfo(status=status.strip(), recent_commits=commits)

    async def _run_git(self, args: List[str]) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode != 0:
                return ""
            return stdout.decode("utf-8", errors="ignore")
        except FileNotFoundError:  # pragma: no cover - git unavailable
            return ""

    async def _get_dependencies(self) -> Dict[str, Any]:
        dependencies: Dict[str, Any] = {}
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                package_data = json.loads(package_json.read_text())
                dependencies["node"] = {
                    "dependencies": package_data.get("dependencies", {}),
                    "devDependencies": package_data.get("devDependencies", {}),
                }
            except Exception:
                dependencies["node"] = {}

        requirements = self.repo_path / "requirements.txt"
        if requirements.exists():
            try:
                dependencies["python"] = requirements.read_text().splitlines()
            except Exception:
                dependencies["python"] = []
        return dependencies

    def _iter_files(self) -> Iterable[Path]:
        for root, _, files in os.walk(self.repo_path):
            root_path = Path(root)
            if any(part.startswith(".") for part in root_path.relative_to(self.repo_path).parts):
                continue
            for name in files:
                if name.startswith("."):
                    continue
                yield root_path / name

    @staticmethod
    def _gather_node_dependencies(package_data: Dict[str, Any]) -> List[str]:
        deps = []
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            section_deps = package_data.get(section, {})
            if isinstance(section_deps, dict):
                deps.extend(section_deps.keys())
        return deps

    @staticmethod
    def _node_to_frameworks(dependencies: Iterable[str]) -> List[str]:
        frameworks = set()
        for dep in dependencies:
            name = dep.lower()
            if name in {"react", "react-dom"}:
                frameworks.add("React")
            elif name.startswith("@angular/"):
                frameworks.add("Angular")
            elif name.startswith("next"):
                frameworks.add("Next.js")
            elif name.startswith("vite"):
                frameworks.add("Vite")
            elif name.startswith("express"):
                frameworks.add("Express")
            elif name.startswith("nestjs"):
                frameworks.add("NestJS")
        return sorted(frameworks)

    @staticmethod
    def _python_to_frameworks(pyproject: Dict[str, Any]) -> List[str]:
        frameworks = set()
        deps = []
        poetry_deps = pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
        if isinstance(poetry_deps, dict):
            deps.extend(poetry_deps.keys())
        optional = pyproject.get("project", {}).get("optional-dependencies", {})
        for items in optional.values():
            if isinstance(items, list):
                deps.extend(items)
        for dep in deps:
            name = dep.lower()
            if "fastapi" in name:
                frameworks.add("FastAPI")
            elif "django" in name:
                frameworks.add("Django")
            elif "flask" in name:
                frameworks.add("Flask")
            elif "pytest" in name:
                frameworks.add("pytest")
        return sorted(frameworks)

