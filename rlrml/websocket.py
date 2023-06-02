import asyncio
import enum
import json
import logging
import numpy as np
import torch
import websockets

from threading import Thread
from . import metadata


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

    async def process_and_broadcast_message(
            self, message, prepare_for_broadcast=lambda x: x
    ):
        try:
            processed_message = prepare_for_broadcast(message)
        except Exception as e:
            logger.warning(f"Exception {e} encountered preparing {message}")
        websockets.broadcast(self.connected, processed_message)

    def send_message_to_clients(self, message, prepare_for_broadcast=lambda x: x):
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
                try:
                    self.client_message_handler(message)
                except Exception as e:
                    logger.warn(f"Exception {e} when handling client message {message}")
        finally:
            self.connected.remove(websocket)
            logger.info(f"Removed connection, count: {len(self.connected)}")


class MessageType(enum.StrEnum):

    START_LOSS_ANALYSIS = enum.auto()
    START_TRAINING = enum.auto()
    STOP_TRAINING = enum.auto()
    SAVE_MODEL = enum.auto()
    PLAYER_MMR_OVERRIDE = enum.auto()
    BUST_LABEL_CACHE = enum.auto()


class FrontendManager:

    def __init__(
            self, host, port, trainer, label_scaler,
            player_cache, model, builder, args, parser
    ):
        self._args = args
        self._builder = builder
        self._parser = parser
        self._trainer = trainer
        self._training_thread = None
        self._training_should_continue = True
        self._player_cache = player_cache
        self._model = model
        self._loss_epoch_counter = 0
        self._label_scaler = label_scaler
        self._server = Server.serve_in_dedicated_thread(
            host, port, client_message_handler=self._handle_client_message
        )
        self._message_type_to_handler = {
            MessageType.START_LOSS_ANALYSIS: self._start_loss_analysis,
            MessageType.START_TRAINING: self._start_training,
            MessageType.STOP_TRAINING: self._stop_training,
            MessageType.SAVE_MODEL: self._save_model,
            MessageType.PLAYER_MMR_OVERRIDE: self._set_player_mmr_override,
            MessageType.BUST_LABEL_CACHE: self._bust_label_cache,
        }

    def _handle_client_message(self, message):
        message = json.loads(message)
        if 'type' not in message:
            logger.warn(f"Message {message} did not contain type key")
            return
        logger.info(f"Client Request: {message}")
        message_type = message['type']
        if message['type'] not in self._message_type_to_handler:
            logger.warn(f"Unrecognized message type {message_type}")

        return self._message_type_to_handler[message['type']](**message.get('data', {}))

    def _make_client_message(self, message_type, data):
        return json.dumps({"type": message_type, "data": data})

    def _ensure_training_thread_ready(self):
        if self._training_thread is not None:
            logger.warn(
                "Attempt to start training even though "
                "training thread already exists"
            )
            return False
        return True

    def _start_loss_analysis(self):
        if not self._ensure_training_thread_ready():
            return

        self._training_should_continue = True
        self._training_thread = Thread(
            target=self._calculate_loss, daemon=True
        )
        self._training_thread.start()

    def _start_training(self, **kwargs):
        if not self._ensure_training_thread_ready():
            return
        self._training_should_continue = True
        self._training_thread = Thread(
            target=self._train, kwargs=kwargs, daemon=True
        )
        self._server.send_message_to_clients(
            self._make_client_message(
                "training_start",
                {"player_count": self._args.playlist.player_count}
            )
        )
        self._training_thread.start()

    def _stop_training(self):
        self._training_should_continue = False

    def _save_model(self, model_filepath=None):
        default = self._args.model_path
        model_filepath = model_filepath or default
        torch.save(self._model.state_dict(), model_filepath)

    def _set_player_mmr_override(self, tracker_suffix, mmr=None, clear=False):
        player = metadata.PlatformPlayer.from_tracker_suffix(tracker_suffix)
        if clear:
            self._player_cache.remove_manual_override(player)
        self._player_cache.insert_manual_override(player, mmr or None)

    def _bust_label_cache(self):
        self.builder.torch_dataset.bust_label_cache()

    def _calculate_loss(self):
        self._model.eval()
        self._trainer.process_loss(self._process_loss_batch)

    def _process_loss_batch(self, training_data, y_pred, loss_tensor):
        data = {
            "y_loss": loss_tensor.detach(),
            "y_pred": y_pred.detach(),
            "mask": training_data.mask.detach(),
            "y": training_data.y.detach(),
            "meta": training_data.meta,
            "epoch": self._loss_epoch_counter,
            "uuids": training_data.uuids,
        }
        self._loss_epoch_counter += 1
        self._server.send_message_to_clients(
            data, self._prepare_loss_batch_for_broadcast
        )
        return self._training_should_continue

    def _prepare_loss_batch_for_broadcast(self, data):
        self._transform_data(data)
        return self._make_client_message("loss_batch", data)

    def _train(self, epochs=None):
        self._model.train()
        self._trainer.train(epochs=epochs, on_epoch_finish=self._on_epoch_finish)
        self._training_thread = None

    def _transform_data(self, data):
        data['y_loss'] = np.sqrt(self._label_scaler.unscale_no_translate(
            data['y_loss'].cpu()
        )).tolist()
        data['mask'] = data['mask'].cpu().tolist()
        data['y_pred'] = self._label_scaler.unscale(data['y_pred'].cpu()).tolist()
        data['y'] = self._label_scaler.unscale(data['y'].cpu()).tolist()
        data['tracker_suffixes'] = [
            [player.tracker_suffix for player in meta.player_order]
            for meta in data['meta']
        ]
        del data['meta']
        return data

    def _prepare_training_info_for_broadcast(self, data):
        if "trainer" in data:
            del data["trainer"]
        data['loss'] = np.sqrt(self._label_scaler.unscale_no_translate(data['loss']))
        self._transform_data(data)
        return self._make_client_message("training_epoch", data)

    def _on_epoch_finish(self, **kwargs):
        self._server.send_message_to_clients(
            kwargs, self._prepare_training_info_for_broadcast
        )
        return self._training_should_continue
