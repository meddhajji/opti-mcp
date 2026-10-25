"""The optimizer as a web service: the same engine as the plugin and the library, reachable by URL.

  /mcp             MCP over HTTP: the same two tools as the local plugin, for AI assistants that
                   take a connector URL (for example a custom connector in Claude)
  POST /solve      {"kind": "...", "data": {...}}: the same answer as the solve tool, as JSON
  GET  /formats    the catalogue of kinds; /formats/{kind}: the full format of one kind
  GET  /health     liveness check for the host

Nothing a caller sends is stored, and only the kind, status, size and time of each solve are
logged. A free public instance has to stay usable for everyone, hence three limits: request
size, solves running at once, and requests per minute from one client.

Run locally:  python -m optimizer.web     (http://127.0.0.1:8000; HOST and PORT override it)
In Docker:    see the Dockerfile at the repo root.
"""
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anyio
import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from . import kinds
from .server import server

MAX_BODY_BYTES = int(os.environ.get("OPTIMIZER_MAX_BODY_BYTES", 256_000))
MAX_CONCURRENT = int(os.environ.get("OPTIMIZER_MAX_CONCURRENT", 4))
REQUESTS_PER_MINUTE = int(os.environ.get("OPTIMIZER_REQUESTS_PER_MINUTE", 30))

log = logging.getLogger("optimizer.web")

def package_version() -> str:
    """pyproject.toml's version when running from the source (a checkout, or the Docker image), else
    the installed package's: an editable install keeps the version it was installed with."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        found = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
        if found:
            return found.group(1)
    try:
        return version("optimizer-mcp")
    except PackageNotFoundError:
        return "dev"


VERSION = package_version()


def problem(status: int, message: str) -> JSONResponse:
    return JSONResponse({"status": "REFUSED", "meaning": message}, status_code=status)


@server.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": VERSION})


@server.custom_route("/formats", methods=["GET"])
async def formats(request: Request) -> JSONResponse:
    return JSONResponse({"kinds": list(kinds.KIND_NAMES), "catalogue": kinds.catalogue()})


@server.custom_route("/formats/{kind}", methods=["GET"])
async def one_format(request: Request) -> PlainTextResponse:
    kind = request.path_params["kind"]
    return PlainTextResponse(kinds.get_format(kind), status_code=200 if kind in kinds.KINDS else 404)


@server.custom_route("/solve", methods=["POST"])
async def solve(request: Request) -> JSONResponse:
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return problem(413, f"The request is larger than {MAX_BODY_BYTES:,} bytes.")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict) or "kind" not in payload or "data" not in payload:
        return JSONResponse(kinds.invalid(['send JSON of the form {"kind": "...", "data": {...}}; '
                                           "GET /formats lists the kinds"]), status_code=400)
    started = time.monotonic()
    answer = await anyio.to_thread.run_sync(kinds.solve, payload["kind"], payload["data"])
    log.info("solve kind=%s status=%s bytes=%d seconds=%.2f", payload["kind"], answer.get("status"), len(body),
             time.monotonic() - started)
    return JSONResponse(answer, status_code=422 if answer.get("status") == "INVALID_INPUT" else 200)


class Limits:
    """Requests per minute from one client, and solves running at once, for every endpoint but /health.

    In memory, so per instance: enough for a free single-instance deployment, not for a fleet."""

    def __init__(self, app):
        self.app, self.busy, self.recent = app, 0, defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        now, client = time.monotonic(), client_address(scope)
        calls = self.recent[client]
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= REQUESTS_PER_MINUTE:
            return await problem(429, f"At most {REQUESTS_PER_MINUTE} requests per minute: wait a minute and "
                                      "try again.")(scope, receive, send)
        calls.append(now)
        if len(self.recent) > 10_000:  # forget clients idle for a minute, so memory stays bounded
            for key in [k for k, v in self.recent.items() if not v or now - v[-1] > 60]:
                del self.recent[key]
        if self.busy >= MAX_CONCURRENT:
            return await problem(503, "The service is busy with other problems: try again in a few seconds.")(
                scope, receive, send)
        self.busy += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.busy -= 1


def client_address(scope) -> str:
    """The caller's address; behind the host's proxy, the first X-Forwarded-For entry."""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            return value.decode("latin-1").split(",")[0].strip()
    return (scope.get("client") or ("unknown",))[0]


def create_app():
    # Host checking protects servers on localhost from DNS rebinding; a public service names its
    # own domain in OPTIMIZER_ALLOWED_HOSTS to turn it on (comma-separated, e.g. "example.org").
    allowed = [h.strip() for h in os.environ.get("OPTIMIZER_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(allowed), allowed_hosts=allowed,
                                         allowed_origins=[f"https://{h}" for h in allowed])
    app = server.streamable_http_app(stateless_http=True, json_response=True, max_request_body_size=MAX_BODY_BYTES,
                                     transport_security=security, host="0.0.0.0")
    app.add_middleware(Limits)
    return app


app = create_app()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
