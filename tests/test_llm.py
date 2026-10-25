"""Tests for the Ollama client, with the HTTP calls replaced by fakes."""

import json

import httpx
import pytest

from app.llm import LLMError, OllamaClient


def fake_post(reply_content=None, status=200, raise_error=None):
    calls = []

    def _post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json})
        if raise_error:
            raise raise_error
        request = httpx.Request("POST", url)
        body = {"message": {"content": reply_content}} if reply_content is not None else {}
        return httpx.Response(status, json=body, request=request)

    return _post, calls


def test_sends_model_messages_and_low_temperature(monkeypatch):
    post, calls = fake_post("Hello!")
    monkeypatch.setattr(httpx, "post", post)
    client = OllamaClient("http://ollama:11434/", "qwen2.5:1.5b")

    assert client.chat([{"role": "user", "content": "hi"}]) == "Hello!"
    sent = calls[0]
    assert sent["url"] == "http://ollama:11434/api/chat"
    assert sent["json"]["model"] == "qwen2.5:1.5b"
    assert sent["json"]["stream"] is False
    assert sent["json"]["options"]["temperature"] == 0.0
    assert "format" not in sent["json"]


def test_chat_json_sends_schema_and_parses_reply(monkeypatch):
    post, calls = fake_post(json.dumps({"action": "answer"}))
    monkeypatch.setattr(httpx, "post", post)
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}

    result = OllamaClient("http://x", "m").chat_json([], json_schema=schema)
    assert result == {"action": "answer"}
    assert calls[0]["json"]["format"] == schema


def test_connection_error_becomes_llm_error(monkeypatch):
    post, _ = fake_post(raise_error=httpx.ConnectError("refused"))
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat([])


def test_server_error_becomes_llm_error(monkeypatch):
    post, _ = fake_post("oops", status=500)
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat([])


def test_invalid_json_becomes_llm_error(monkeypatch):
    post, _ = fake_post("this is not json")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat_json([], json_schema={})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda seconds: None)


def test_sends_fixed_seed(monkeypatch):
    post, calls = fake_post("ok")
    monkeypatch.setattr(httpx, "post", post)
    OllamaClient("http://x", "m").chat([])
    assert calls[0]["json"]["options"]["seed"] == 42


def test_retries_once_after_connection_error(monkeypatch):
    attempts = []

    def flaky_post(url, json=None, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"message": {"content": "back"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", flaky_post)
    assert OllamaClient("http://x", "m").chat([]) == "back"
    assert len(attempts) == 2


def test_does_not_retry_a_bad_request(monkeypatch):
    post, calls = fake_post("bad", status=400)
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat([])
    assert len(calls) == 1


def test_does_not_retry_a_timeout(monkeypatch):
    post, calls = fake_post(raise_error=httpx.ReadTimeout("slow"))
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMError):
        OllamaClient("http://x", "m").chat([])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "models, expected",
    [([{"name": "qwen2.5:3b"}], "ok"), ([{"name": "llama3:8b"}], "model_missing")],
)
def test_status(monkeypatch, models, expected):
    def fake_get(url, timeout=None):
        return httpx.Response(200, json={"models": models}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaClient("http://x", "qwen2.5:3b").status() == expected


def test_status_unreachable(monkeypatch):
    def fake_get(url, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert OllamaClient("http://x", "qwen2.5:3b").status() == "unreachable"
