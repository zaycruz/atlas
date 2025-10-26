"""Git branch management utilities for parallel orchestration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class MergeResult:
    success: bool
    conflicts: List[str] = field(default_factory=list)
    merged_branches: List[str] = field(default_factory=list)
    failed_branches: List[str] = field(default_factory=list)
    message: str = ""


class BranchStrategy:
    """Lightweight git helper for creating, merging, and cleaning temporary branches."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = Path(repo_path).expanduser()

    async def create_step_branch(
        self,
        task_id: str,
        step_id: str,
        *,
        base_branch: str = "main",
    ) -> str:
        branch_name = f"atlas/{task_id}/{step_id}"
        await self._ensure_branch(base_branch)
        await self._checkout(base_branch)
        await self._git("checkout", "-B", branch_name, base_branch)
        return branch_name

    async def merge_branches(
        self,
        branches: Sequence[str],
        *,
        target_branch: str = "main",
        strategy: str = "smart",
    ) -> MergeResult:
        branches = list(dict.fromkeys(branches))
        if not branches:
            return MergeResult(success=True, message="No branches to merge.")
        await self._ensure_branch(target_branch)
        strategy = strategy.lower()
        if strategy == "smart" or strategy == "sequential":
            return await self._sequential_merge(branches, target_branch)
        if strategy == "octopus":
            return await self._octopus_merge(branches, target_branch)
        raise ValueError(f"Unknown merge strategy: {strategy}")

    async def cleanup_branches(self, branches: Iterable[str], *, force: bool = False) -> None:
        for branch in branches:
            flag = "-D" if force else "-d"
            await self._git("branch", flag, branch, check=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _sequential_merge(self, branches: Sequence[str], target: str) -> MergeResult:
        result = MergeResult(success=True, message="Sequential merge complete.")
        for branch in branches:
            await self._checkout(target)
            ok, conflicts, stderr = await self._merge_branch(branch)
            if ok:
                result.merged_branches.append(branch)
            else:
                result.success = False
                result.failed_branches.append(branch)
                result.conflicts.extend(conflicts)
                if stderr:
                    result.message = stderr.strip()
                break
        if result.conflicts:
            result.message = result.message or "Merge conflicts detected."
        return result

    async def _octopus_merge(self, branches: Sequence[str], target: str) -> MergeResult:
        await self._checkout(target)
        ok, conflicts, stderr = await self._merge_many(branches)
        if ok:
            return MergeResult(
                success=True,
                merged_branches=list(branches),
                message="Octopus merge complete.",
            )
        result = MergeResult(
            success=False,
            conflicts=conflicts,
            failed_branches=list(branches),
            message=stderr.strip() or "Octopus merge failed.",
        )
        return result

    async def _merge_branch(self, branch: str) -> Tuple[bool, List[str], str]:
        code, _, stderr = await self._run_git("merge", "--no-ff", branch)
        if code == 0:
            return True, [], ""
        conflicts = await self._collect_conflicts()
        await self._run_git("merge", "--abort", check=False)
        return False, conflicts, stderr

    async def _merge_many(self, branches: Sequence[str]) -> Tuple[bool, List[str], str]:
        code, _, stderr = await self._run_git("merge", "--no-ff", *branches)
        if code == 0:
            return True, [], ""
        conflicts = await self._collect_conflicts()
        await self._run_git("merge", "--abort", check=False)
        return False, conflicts, stderr

    async def _collect_conflicts(self) -> List[str]:
        code, stdout, _ = await self._run_git("diff", "--name-only", "--diff-filter=U")
        if code != 0 or not stdout:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    async def _checkout(self, branch: str) -> None:
        await self._git("checkout", branch)

    async def _ensure_branch(self, branch: str) -> None:
        code, _, _ = await self._run_git("rev-parse", "--verify", branch)
        if code != 0:
            raise RuntimeError(f"Branch '{branch}' does not exist in repository {self.repo_path}")

    async def _git(self, *args: str, check: bool = True) -> str:
        code, stdout, stderr = await self._run_git(*args)
        if check and code != 0:
            command = " ".join(args)
            raise RuntimeError(f"git {command} failed: {stderr.strip()}")
        return stdout

    async def _run_git(self, *args: str, check: bool = False) -> Tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {stderr.strip()}")
        return proc.returncode, stdout, stderr

