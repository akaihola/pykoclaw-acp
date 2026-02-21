from __future__ import annotations

import asyncio
import hashlib

import pytest

from pykoclaw_acp.protocol import JsonRpcError


def _expected_response(prompt: str, *, turn: int, call: int) -> str:
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"Response[{prompt_hash}] to: {prompt} (turn={turn}, call={call})"


@pytest.mark.asyncio
async def test_initialize_before_anything(
    acp_client,
) -> None:
    result = await acp_client.initialize()
    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"] == {"loadSession": True}
    assert result["agentInfo"]["name"] == "pykoclaw"
    assert result["agentInfo"]["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_full_session_lifecycle(
    acp_client,
) -> None:
    await acp_client.initialize()
    session_id = await acp_client.new_session(cwd="/tmp/project")

    prompt_text = "Explain what this repository does"
    result = await acp_client.prompt(session_id, prompt_text)

    assert len(result.chunks) >= 2
    assert "".join(result.chunks) == _expected_response(prompt_text, turn=1, call=1)
    assert result.stop_response is not None
    assert result.stop_response["result"]["stopReason"] == "end_turn"
    assert result.error_updates == []


@pytest.mark.asyncio
async def test_multi_turn_conversation_is_deterministic_and_distinct(
    acp_client,
    mock_pool,
) -> None:
    session_id = await acp_client.new_session()
    prompts = ["First task", "Second task", "Third task"]

    results = [await acp_client.prompt(session_id, prompt) for prompt in prompts]
    combined = ["".join(result.chunks) for result in results]

    assert combined == [
        _expected_response(prompts[0], turn=1, call=1),
        _expected_response(prompts[1], turn=2, call=2),
        _expected_response(prompts[2], turn=3, call=3),
    ]
    assert len(set(combined)) == 3
    assert mock_pool.session_history[session_id] == prompts


@pytest.mark.asyncio
async def test_multiple_concurrent_sessions_are_isolated(
    acp_client,
    mock_pool,
) -> None:
    session_ids = [await acp_client.new_session() for _ in range(3)]
    prompts = ["Alpha question", "Beta question", "Gamma question"]

    results = await asyncio.gather(
        *(
            acp_client.prompt(session_id, prompt)
            for session_id, prompt in zip(session_ids, prompts)
        )
    )

    for index, result in enumerate(results):
        full_text = "".join(result.chunks)
        prompt_hash = hashlib.sha1(prompts[index].encode("utf-8")).hexdigest()[:8]
        assert full_text.startswith(f"Response[{prompt_hash}] to: {prompts[index]}")
        assert result.stop_response is not None
        assert result.stop_response["result"]["stopReason"] == "end_turn"

    for session_id, prompt in zip(session_ids, prompts):
        assert mock_pool.session_history[session_id] == [prompt]


@pytest.mark.asyncio
async def test_session_isolation_with_same_prompt_text(
    acp_client,
    mock_pool,
) -> None:
    session_a = await acp_client.new_session()
    session_b = await acp_client.new_session()
    prompt_text = "Use the same prompt"

    result_a = await acp_client.prompt(session_a, prompt_text)
    result_b = await acp_client.prompt(session_b, prompt_text)

    assert "".join(result_a.chunks) != ""
    assert "".join(result_b.chunks) != ""
    assert mock_pool.session_history[session_a] == [prompt_text]
    assert mock_pool.session_history[session_b] == [prompt_text]


@pytest.mark.asyncio
async def test_error_prompt_to_invalid_session(acp_client) -> None:
    messages = await acp_client.request(
        "session/prompt",
        {
            "sessionId": "bogus-session",
            "prompt": [{"type": "text", "text": "Hello"}],
        },
    )

    assert len(messages) == 1
    assert messages[0]["error"]["code"] == JsonRpcError.INVALID_SESSION


@pytest.mark.asyncio
async def test_error_empty_prompt(acp_client) -> None:
    session_id = await acp_client.new_session()
    messages = await acp_client.request(
        "session/prompt",
        {
            "sessionId": session_id,
            "prompt": [],
        },
    )

    assert len(messages) == 1
    assert messages[0]["error"]["code"] == JsonRpcError.INVALID_PARAMS


@pytest.mark.asyncio
async def test_streaming_fidelity_chunk_concat_matches_full_response(
    acp_client,
) -> None:
    session_id = await acp_client.new_session()
    prompt_text = "Chunk this response carefully"
    result = await acp_client.prompt(session_id, prompt_text)

    assert len(result.chunks) >= 2
    assert "".join(result.chunks) == _expected_response(prompt_text, turn=1, call=1)


@pytest.mark.asyncio
async def test_server_resilience_after_mock_pool_failure(
    acp_client,
    mock_pool,
) -> None:
    session_id = await acp_client.new_session()
    failing_prompt = "trigger controlled failure"
    mock_pool.fail_prompts.add(failing_prompt)

    failed = await acp_client.prompt(session_id, failing_prompt)
    assert failed.error_updates
    assert failed.stop_response is not None
    assert failed.stop_response["result"]["stopReason"] == "end_turn"

    follow_up_prompt = "service still available"
    recovered = await acp_client.prompt(session_id, follow_up_prompt)
    assert recovered.error_updates == []
    assert "".join(recovered.chunks) == _expected_response(
        follow_up_prompt, turn=2, call=2
    )
    assert recovered.stop_response is not None
    assert recovered.stop_response["result"]["stopReason"] == "end_turn"
