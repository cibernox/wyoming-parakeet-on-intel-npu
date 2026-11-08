import asyncio
import unittest
from unittest import mock

from wyoming.info import Info

from wyoming_parakeet_npu.handler import ParakeetEventHandler


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DisconnectClosesWriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_closes_writer_when_client_disconnects(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_eof()  # client closed its side: run() returns
        writer = _FakeWriter()

        handler = ParakeetEventHandler(
            Info(asr=[]), mock.Mock(), asyncio.Lock(), reader, writer
        )
        await handler.run()

        self.assertTrue(writer.closed, "writer was not closed after client disconnect")


if __name__ == "__main__":
    unittest.main()
