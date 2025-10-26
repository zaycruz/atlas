from __future__ import annotations

import asyncio
import yaml

from atlas_main.agents.factory import AgentFactory


def test_factory_creates_reasoning_agent(tmp_path):
    config_path = tmp_path / "agents.yaml"
    config = {
        "agents": {
            "planner-deepseek": {
                "type": "reasoning",
                "provider": "ollama",
                "model": "deepseek-r1",
                "timeout": 123,
                "max_tokens": 4096,
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    factory = AgentFactory(config_path=config_path)

    adapter = asyncio.run(factory.create("planner-deepseek"))

    assert adapter.provider == "ollama"
    assert adapter.model == "deepseek-r1"
