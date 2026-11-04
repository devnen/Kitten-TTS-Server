# File: wyoming_server.py
# Wyoming protocol (https://github.com/OHF-Voice/wyoming) TTS server so that
# Home Assistant can use KittenTTS. Runs on its own TCP port next to the FastAPI app.

import logging
import socket
from typing import Optional

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

import engine
import utils
from config import get_gen_default_speed

logger = logging.getLogger(__name__)

_ATTRIBUTION = Attribution(name="KittenML", url="https://github.com/KittenML/KittenTTS")
_LANGUAGE = "en_US"
_SAMPLES_PER_CHUNK = 1024
_TEXT_CHUNK_SIZE = 120  # Same default as the /tts endpoint.
_ENGINE_SAMPLE_RATE = 24000  # Hardcoded in engine.synthesize.


def build_info() -> Info:
    """Built per Describe because the voice list changes when the model is hot-swapped."""
    model = engine.get_model_info()
    voices = [
        TtsVoice(
            name=voice,
            description=voice,
            attribution=_ATTRIBUTION,
            installed=True,
            version=model.get("version"),
            languages=[_LANGUAGE],
        )
        for voice in engine.get_available_voices()
    ]
    return Info(
        tts=[
            TtsProgram(
                name="kitten-tts",
                description=model.get("display_name") or "KittenTTS",
                attribution=_ATTRIBUTION,
                installed=engine.MODEL_LOADED,
                version=model.get("version"),
                voices=voices,
                supports_synthesize_streaming=False,
            )
        ]
    )


def _to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(np.asarray(audio, dtype=np.float32).squeeze(), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


class KittenEventHandler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(build_info().event())
            return True

        if not Synthesize.is_type(event.type):
            return True

        try:
            await self._synthesize(Synthesize.from_event(event))
        except Exception as err:
            logger.error("Wyoming synthesize failed: %s", err, exc_info=True)
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            return False

        return True

    async def _synthesize(self, request: Synthesize) -> None:
        if not engine.MODEL_LOADED:
            raise RuntimeError("TTS model is not loaded")

        voice: str = (
            request.voice.name
            if request.voice and request.voice.name
            else engine.get_model_info()["default_voice"]
        )

        chunks = utils.chunk_text_by_sentences(
            " ".join(request.text.strip().splitlines()), _TEXT_CHUNK_SIZE
        )
        speed = get_gen_default_speed()
        started = False

        for text in chunks:
            # Called on the event-loop thread, like the HTTP endpoints, because the
            # engine's espeak phonemizer is not safe to use from several threads.
            audio, rate = engine.synthesize(text, voice, speed)

            if audio is None or rate is None:
                raise RuntimeError(f"Synthesis failed for voice '{voice}'")

            if not started:
                await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
                started = True

            pcm = _to_pcm16(audio)
            step = _SAMPLES_PER_CHUNK * 2
            for offset in range(0, len(pcm), step):
                await self.write_event(
                    AudioChunk(
                        audio=pcm[offset : offset + step],
                        rate=rate,
                        width=2,
                        channels=1,
                    ).event()
                )

        if not started:
            # Nothing speakable in the text; still send a well-formed empty stream.
            await self.write_event(
                AudioStart(rate=_ENGINE_SAMPLE_RATE, width=2, channels=1).event()
            )

        await self.write_event(AudioStop().event())


async def start_wyoming_server(
    host: str, port: int, zeroconf_name: Optional[str] = None
) -> "WyomingService":
    """Starts the Wyoming listener; advertises it over mDNS when zeroconf_name is set."""
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    await server.start(KittenEventHandler)
    logger.info("Wyoming TTS server listening on tcp://%s:%s", host, port)

    service = WyomingService(server)
    if zeroconf_name:
        # Discovery is a convenience; the listener keeps working without it.
        try:
            await service.register_zeroconf(zeroconf_name, port, host)
        except Exception as err:
            logger.warning("Zeroconf advertisement failed: %s", err)
    return service


class WyomingService:
    def __init__(self, server: AsyncServer) -> None:
        self._server = server
        self._zeroconf = None
        self._zeroconf_info = None

    async def register_zeroconf(self, name: str, port: int, host: str) -> None:
        from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

        info = AsyncServiceInfo(
            "_wyoming._tcp.local.",
            f"{name}._wyoming._tcp.local.",
            addresses=[socket.inet_aton(_advertised_ip(host))],
            port=port,
        )
        zc = AsyncZeroconf()
        try:
            await zc.async_register_service(info)
        except Exception:
            await zc.async_close()
            raise
        self._zeroconf, self._zeroconf_info = zc, info
        logger.info("Zeroconf: advertising '%s' on %s:%s", name, _advertised_ip(host), port)

    async def stop(self) -> None:
        if self._zeroconf is not None and self._zeroconf_info is not None:
            await self._zeroconf.async_unregister_service(self._zeroconf_info)
            await self._zeroconf.async_close()
            self._zeroconf = self._zeroconf_info = None
        await self._server.stop()


def _advertised_ip(bind_host: str) -> str:
    """A wildcard bind address is not reachable, so advertise the LAN-facing IP instead."""
    if bind_host not in ("0.0.0.0", "", "::"):
        return bind_host

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("224.0.0.251", 1))  # mDNS multicast address; sends no packets.
        return sock.getsockname()[0]
