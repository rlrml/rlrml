import asyncio
import enum
import json
import logging
import numpy as np
import websockets

from threading import Thread


logger = logging.getLogger(__name__)


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

    def __init__(self, loop=None, client_message_handler=lambda x: None):
        self.loop = loop or asyncio.get_event_loop()
        self.connected = set()
        self.client_message_handler = client_message_handler

    async def process_and_broadcast_message(self, message, prepare_for_broadcast=lambda x: x):
        processed_message = prepare_for_broadcast(message)
        websockets.broadcast(self.connected, processed_message)

    def process_message(self, message, prepare_for_broadcast=lambda x: x):
        return asyncio.run_coroutine_threadsafe(
            self.process_and_broadcast_message(
                message, prepare_for_broadcast=prepare_for_broadcast
            ),
            self.loop
        )

    async def handler(self, websocket, path):
        self.connected.add(websocket)
        logger.info(f"New websocket connection, count: {len(self.connected)}")
        try:
            async for message in websocket:
                self.client_message_handler(message)
        finally:
            self.connected.remove(websocket)
            logger.info(f"Removed connection, count: {len(self.connected)}")


class LossType(enum.StrEnum):

    START_TRAINING = enum.auto()
    STOP_TRAINING = enum.auto()


class FrontendManager:

    def __init__(self, host, port, trainer, label_scaler):
        self._trainer = trainer
        self._training_thread = None
        self._training_should_continue = True
        self._label_scaler = label_scaler
        self._server = Server.serve_in_dedicated_thread(
            host, port, client_message_handler=self._handle_client_message
        )
        self._message_type_to_handler = {
            LossType.START_TRAINING: self._start_training,
            LossType.STOP_TRAINING: self._stop_training,
        }

    def _handle_client_message(self, message):
        message = json.loads(message)
        if 'type' not in message:
            logger.warn(f"Message {message} did not contain type key")
            return
        message_type = message['type']
        if message['type'] not in self._message_type_to_handler:
            logger.warn(f"Unrecognized message type {message_type}")

        return self._message_type_to_handler[message['type']](message)

    def _start_training(self, message):
        if self._training_thread is not None:
            logger.warn("Attempt to start training even though training thread already exists")
            return
        self._training_should_continue = True
        self._training_thread = Thread(
            target=self._train, kwargs=message.get("args", {}), daemon=True
        )
        self._training_thread.start()

    def _stop_training(self, _message):
        self._training_should_continue = False

    def _train(self, epochs=None):
        self._trainer.train(epochs=epochs, on_epoch_finish=self._on_epoch_finish)
        training_thread = self._training_thread
        self._training_thread = None

    def _prepare_training_info_for_broadcast(self, kwargs):
        del kwargs["trainer"]
        kwargs['loss'] = np.sqrt(self._label_scaler.unscale_no_translate(kwargs['loss']))
        kwargs['y_loss'] = np.sqrt(self._label_scaler.unscale_no_translate(
            kwargs['y_loss'].cpu()
        )).tolist()
        kwargs['y_pred'] = self._label_scaler.unscale(kwargs['y_pred'].cpu()).tolist()
        kwargs['y'] = self._label_scaler.unscale(kwargs['y'].cpu()).tolist()
        return kwargs

    def _on_epoch_finish(self, **kwargs):
        self._server.process_and_broadcast_message(
            kwargs, self._prepare_training_info_for_broadcast
        )
        return self._training_should_continue
