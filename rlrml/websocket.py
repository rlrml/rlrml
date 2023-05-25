import asyncio
import websockets
from threading import Thread


class Server:

    @classmethod
    def serve_in_dedicated_thread(cls, host, port, **kwargs):
        loop = asyncio.new_event_loop()
        server = cls(loop=loop, **kwargs)

        def _background():
            asyncio.set_event_loop(loop)

            async def start_server():
                await websockets.serve(server.handler, host, port)

            asyncio.run_coroutine_threadsafe(start_server(), loop)
            loop.run_forever()

        thread = Thread(target=_background, daemon=True)
        thread.start()

        return server

    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.connected = set()

    async def process_and_broadcast_message(self, message, prepare_for_broadcast=lambda x: x):
        processed_message = prepare_for_broadcast(message)
        websockets.broadcast(self.connected, processed_message)

    def process_message(self, message):
        return asyncio.run_coroutine_threadsafe(
            self.process_and_broadcast_message(message),
            self.loop
        )

    async def handler(self, websocket, path):
        self.connected.add(websocket)
        try:
            async for message in websocket:
                self.process_message(message)
        finally:
            # Unregister.
            self.connected.remove(websocket)
