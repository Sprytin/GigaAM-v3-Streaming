"""A small WebSocket server around the recognizer (aiohttp).

    pip install ".[server]"   # in a clone of GigaAM-v3-Streaming
    python examples/ws_server.py --port 8765

Protocol, one stream per connection:

* client -> server: binary frames of 16 kHz mono s16le PCM, any size;
  the text frame ``{"type": "end"}`` ends the stream (closing works too).
* server -> client: JSON text frames
  ``{"type": "partial", "text", "stable_text", "start", "end"}`` and
  ``{"type": "final", "text", "start", "end", "words": [{"text", "start", "end"}]}``;
  after ``end`` the server sends the remaining finals, then ``{"type": "done"}``.

All streams share one model. Inference runs on one worker thread, so GPU work
from different connections is serialized and the event loop stays free.
Streams are not batched: each costs about a third of real time on an RTX 4070
SUPER, so ``--max-streams`` (default 2) refuses further connections with 503
instead of letting every stream fall behind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from aiohttp import WSMsgType, web

from gigaam_v3_streaming import Final, StreamingRecognizer, load_model

log = logging.getLogger("gigaam_v3_streaming.server")


def to_json(event) -> str:
    kind = "final" if isinstance(event, Final) else "partial"
    return json.dumps({"type": kind, **asdict(event)}, ensure_ascii=False)


async def stream(request: web.Request) -> web.StreamResponse:
    app = request.app
    if app["streams"] >= app["max_streams"]:
        return web.Response(status=503, text="all recognition streams are busy")
    app["streams"] += 1
    try:
        return await _serve(request)
    finally:
        app["streams"] -= 1


async def _serve(request: web.Request) -> web.WebSocketResponse:
    app = request.app
    socket = web.WebSocketResponse(max_msg_size=1 << 20)
    await socket.prepare(request)
    recognizer = StreamingRecognizer(app["model"])
    loop = asyncio.get_running_loop()

    async def run(call, *args):
        for event in await loop.run_in_executor(app["worker"], call, *args):
            await socket.send_str(to_json(event))

    try:
        async for message in socket:
            if message.type == WSMsgType.BINARY:
                if len(message.data) % 2:
                    await socket.close(code=1003, message=b"PCM16 frames must have even length")
                    break
                await run(recognizer.feed, message.data)
            elif message.type == WSMsgType.TEXT:
                if json.loads(message.data).get("type") == "end":
                    break
        if not socket.closed:
            await run(recognizer.finish)
            await socket.send_str(json.dumps({"type": "done"}))
            await socket.close()
    finally:
        # On the worker, after any feed still running there: the recognizer is
        # never touched from two threads.
        await asyncio.shield(loop.run_in_executor(app["worker"], recognizer.cancel))
    return socket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-streams", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app["model"] = load_model(args.model, args.device)
    app["worker"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
    app["streams"], app["max_streams"] = 0, args.max_streams
    app.router.add_get("/", stream)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
