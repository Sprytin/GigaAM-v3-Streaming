"""Send a WAV file to ``ws_server.py`` at real-time speed and print events.

python examples/ws_client.py speech.wav --url ws://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import asyncio
import json

import aiohttp
import soundfile as sf

RATE = 16_000
CHUNK = RATE // 10


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("wav", help="16 kHz mono WAV")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/")
    parser.add_argument("--fast", action="store_true", help="do not pace at 1x")
    args = parser.parse_args()

    audio, rate = sf.read(args.wav, dtype="int16")
    if rate != RATE or audio.ndim != 1:
        raise SystemExit("expected a 16 kHz mono WAV")
    pcm = audio.astype("<i2")

    async with aiohttp.ClientSession() as session, session.ws_connect(args.url) as socket:

        async def send() -> None:
            for at in range(0, pcm.size, CHUNK):
                await socket.send_bytes(pcm[at : at + CHUNK].tobytes())
                if not args.fast:
                    await asyncio.sleep(CHUNK / RATE)
            await socket.send_str(json.dumps({"type": "end"}))

        sender = asyncio.create_task(send())
        async for message in socket:
            event = json.loads(message.data)
            if event["type"] == "done":
                break
            if event["type"] == "final":
                print(f"\r{event['start']:7.2f}s  {event['text']}{' ' * 20}")
            else:
                print(f"\r      ...  {event['text'][-80:]}", end="", flush=True)
        await sender


if __name__ == "__main__":
    asyncio.run(main())
