from atlas_main.memory import WorkingMemory


def test_working_memory_enforces_token_budget():
    wm = WorkingMemory(capacity=10, max_tokens=50)

    wm.add_user("a" * 120)
    assert len(wm.to_messages()) == 1
    assert wm.to_messages()[0]["role"] == "user"

    wm.add_assistant("b" * 120)
    after_second = wm.to_messages()
    assert len(after_second) == 1
    assert after_second[0]["role"] == "assistant"

    wm.add_user("c" * 40)
    after_third = wm.to_messages()
    assert len(after_third) == 2
    assert [msg["role"] for msg in after_third] == ["assistant", "user"]

    wm.add_assistant("d" * 120)
    final_messages = wm.to_messages()
    assert len(final_messages) == 2
    assert [msg["role"] for msg in final_messages] == ["user", "assistant"]
