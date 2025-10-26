from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from atlas_main.orchestrator.branches import BranchStrategy


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "app.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "Initial commit")


def test_create_step_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    strategy = BranchStrategy(repo)
    branch = asyncio.run(strategy.create_step_branch("task", "step"))

    assert branch == "atlas/task/step"
    branches = _git(repo, "branch")
    assert "atlas/task/step" in branches


def test_sequential_merge_success(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    strategy = BranchStrategy(repo)

    branch1 = asyncio.run(strategy.create_step_branch("task", "frontend"))
    (repo / "frontend.txt").write_text("frontend\n", encoding="utf-8")
    _git(repo, "add", "frontend.txt")
    _git(repo, "commit", "-m", "Frontend changes")
    _git(repo, "checkout", "main")

    branch2 = asyncio.run(strategy.create_step_branch("task", "backend"))
    (repo / "backend.txt").write_text("backend\n", encoding="utf-8")
    _git(repo, "add", "backend.txt")
    _git(repo, "commit", "-m", "Backend changes")
    _git(repo, "checkout", "main")

    result = asyncio.run(strategy.merge_branches([branch1, branch2], target_branch="main", strategy="sequential"))

    assert result.success is True
    assert set(result.merged_branches) == {branch1, branch2}
    assert not result.conflicts
    assert (repo / "frontend.txt").exists()
    assert (repo / "backend.txt").exists()


def test_sequential_merge_conflict(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    strategy = BranchStrategy(repo)

    branch1 = asyncio.run(strategy.create_step_branch("task", "alpha"))
    (repo / "app.txt").write_text("alpha\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "Alpha change")
    _git(repo, "checkout", "main")

    branch2 = asyncio.run(strategy.create_step_branch("task", "beta"))
    (repo / "app.txt").write_text("beta\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "Beta change")
    _git(repo, "checkout", "main")

    result = asyncio.run(strategy.merge_branches([branch1, branch2], target_branch="main", strategy="smart"))

    assert result.success is False
    assert branch1 in result.merged_branches
    assert branch2 in result.failed_branches
    assert "app.txt" in result.conflicts


def test_cleanup_branches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    strategy = BranchStrategy(repo)

    branch = asyncio.run(strategy.create_step_branch("task", "cleanup"))
    _git(repo, "checkout", "main")

    asyncio.run(strategy.cleanup_branches([branch], force=True))

    branches = _git(repo, "branch")
    assert "atlas/task/cleanup" not in branches
