"""The web service as its callers see it: the JSON endpoints, the limits, and MCP over real HTTP."""
import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from mcp.client import Client
from starlette.testclient import TestClient

from optimizer import kinds, web

SELECTION = {"items": [{"name": "a", "cost": 3, "value": 4}, {"name": "b", "cost": 4, "value": 5}], "budget": 5}


@pytest.fixture(scope="module")
def http():
    with TestClient(web.app) as client:  # runs the app's startup, which MCP over HTTP needs
        yield client


def as_client(name: str) -> dict:
    """Each test calls from its own address, so per-client limits do not leak between tests."""
    return {"x-forwarded-for": f"203.0.113.{abs(hash(name)) % 250}"}


def test_solve_gives_the_same_answer_as_the_tool(http):
    r = http.post("/solve", json={"kind": "selection", "data": SELECTION}, headers=as_client("same"))
    assert r.status_code == 200 and r.json() == kinds.solve("selection", SELECTION)
    assert r.json()["status"] == "OPTIMAL" and r.json()["total_value"] == 5


def test_mistakes_are_answered_in_plain_words(http):
    h = as_client("mistakes")
    r = http.post("/solve", content=b"not json", headers=h)
    assert r.status_code == 400 and "GET /formats" in r.json()["problems"][0]
    r = http.post("/solve", json={"kind": "selection", "data": {"items": []}}, headers=h)
    assert r.status_code == 422 and r.json()["status"] == "INVALID_INPUT"


def test_formats_list_every_kind(http):
    r = http.get("/formats", headers=as_client("formats"))
    assert r.json()["kinds"] == list(kinds.KIND_NAMES) and "shifts:" in r.json()["catalogue"]
    assert http.get("/formats/cutting", headers=as_client("formats")).text.startswith("cutting: ")
    assert http.get("/formats/knapsack", headers=as_client("formats")).status_code == 404


def test_health_is_never_limited(http, monkeypatch):
    monkeypatch.setattr(web, "MAX_CONCURRENT", 0)
    assert http.get("/health").json()["status"] == "ok"
    assert http.get("/formats", headers=as_client("health")).status_code == 503


def test_too_many_requests_from_one_client(http, monkeypatch):
    monkeypatch.setattr(web, "REQUESTS_PER_MINUTE", 3)
    h = as_client("rate")
    codes = [http.get("/formats", headers=h).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert http.get("/formats", headers=as_client("another client")).status_code == 200


def test_oversized_requests_are_refused(http, monkeypatch):
    monkeypatch.setattr(web, "MAX_BODY_BYTES", 100)
    r = http.post("/solve", json={"kind": "selection", "data": SELECTION}, headers=as_client("big"))
    assert r.status_code == 413


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mcp_over_http_end_to_end():
    """A real HTTP server and the SDK's own client, as an assistant with a connector URL would call it."""
    port = free_port()
    # a fresh app: the MCP session manager starts once per app, and the other tests started web.app
    srv = uvicorn.Server(uvicorn.Config(web.create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)

    async def go():
        async with Client(f"http://127.0.0.1:{port}/mcp") as c:
            names = [t.name for t in (await c.list_tools()).tools]
            res = await c.call_tool("solve", {"kind": "selection", "data": SELECTION})
            return names, json.loads("".join(getattr(x, "text", "") for x in res.content))
    try:
        names, answer = asyncio.run(go())
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
    assert names == ["solve", "get_format"]
    assert answer["status"] == "OPTIMAL" and answer["total_value"] == 5
