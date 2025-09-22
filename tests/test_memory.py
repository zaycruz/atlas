import tempfile
import unittest
from pathlib import Path

from atlas_main.memory import EpisodicMemory, MemoryRecord, WorkingMemory, render_memory_snippets


def keyword_embedder(text: str):
    return [
        1.0 if "alpha" in text else 0.0,
        1.0 if "beta" in text else 0.0,
        1.0 if "gamma" in text else 0.0,
    ]


class WorkingMemoryTests(unittest.TestCase):
    def test_add_and_trim(self):
        wm = WorkingMemory(capacity=3)
        wm.add_user("hello")
        wm.add_assistant("hi")
        wm.add_user("another")
        wm.add_assistant("reply")
        self.assertEqual(len(wm.to_messages()), 3)
        self.assertEqual(wm.to_messages()[-1]["content"], "reply")


class EpisodicMemoryTests(unittest.TestCase):
    def test_remember_and_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            memory = EpisodicMemory(path, embedding_fn=keyword_embedder, max_records=10)
            memory.remember("alpha question", "alpha reply")
            memory.remember("beta question", "beta reply")

            recalled = memory.recall("tell me alpha", top_k=1)
            self.assertEqual(len(recalled), 1)
            self.assertIn("alpha", recalled[0].assistant)

    def test_max_records_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            memory = EpisodicMemory(path, embedding_fn=keyword_embedder, max_records=3)
            for idx in range(5):
                memory.remember(f"alpha {idx}", f"reply {idx}")

            recent = memory.get_recent(5)
            self.assertLessEqual(len(recent), 3)
            ids = [record.user for record in recent]
            self.assertNotIn("alpha 0", ids)
            self.assertIn("alpha 4", ids)


class RenderSnippetsTests(unittest.TestCase):
    def test_snippet_generation(self):
        records = [
            MemoryRecord(id="1", user="hi", assistant="hello there", timestamp=0.0),
            MemoryRecord(id="2", user="", assistant=" ", timestamp=0.0),
        ]
        snippets = render_memory_snippets(records)
        self.assertIn("hello there", snippets)
        self.assertTrue(snippets.startswith("- "))


if __name__ == "__main__":
    unittest.main()
