from __future__ import annotations

import asyncio
import json
import textwrap

from atlas_main.orchestrator.context import CodebaseContext


def test_codebase_context_analysis(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hello')\n")
    (repo / "src" / "index.ts").write_text("console.log('hi');\n")

    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text("def test_ok():\n    assert True\n")

    package_json = {
        "name": "example",
        "dependencies": {"react": "^18.0.0"},
        "devDependencies": {"vite": "^5.0.0"},
    }
    (repo / "package.json").write_text(json.dumps(package_json))

    pyproject_content = textwrap.dedent(
        """
        [tool.poetry]
        name = "example"
        version = "0.1.0"

        [tool.poetry.dependencies]
        python = "^3.11"
        fastapi = "^0.110"

        [project.optional-dependencies]
        tests = ["pytest>=7.0"]
        """
    ).strip()
    (repo / "pyproject.toml").write_text(pyproject_content)

    (repo / "requirements.txt").write_text("pytest==7.4.0\n")

    async def fake_run_git(self, args):
        if args[:2] == ["status", "--short"]:
            return " M src/main.py\n"
        if args[:2] == ["log", "-10"]:
            return "abc123|Alice|2024-01-01|Initial commit\ndef456|Bob|2024-01-02|Add feature"
        return ""

    monkeypatch.setattr(CodebaseContext, "_run_git", fake_run_git, raising=False)

    context = CodebaseContext(repo)
    result = asyncio.run(context.analyze())

    assert repo.name in result["structure"]
    assert "Python" in result["languages"]
    assert any(lang.startswith("TypeScript") for lang in result["languages"])
    assert "React" in result["frameworks"]
    assert "FastAPI" in result["frameworks"]
    assert any(path.endswith("main.py") for path in result["entry_points"])
    assert result["test_framework"] == "pytest"
    assert "node" in result["dependencies"]
    assert "python" in result["dependencies"]
    assert result["git_status"] == "M src/main.py"
    assert len(result["recent_commits"]) == 2
