import json
import lmdb


DEFAULT_VALUE = '{}'.encode('utf-8')


class ReplayAttributesDB:

    def __init__(self, filepath, db_name="replay_attributes", **kwargs):
        self._filepath = filepath
        kwargs.setdefault("max_dbs", 16)
        kwargs.setdefault("map_size", 2 * 1024 ** 3)
        self._env = lmdb.open(filepath, **kwargs)
        self._db = self._env.open_db(db_name.encode('utf-8'))

    def put_replay_attributes(self, uuid, attributes):
        encoded_uuid = self._encode_key(uuid)
        with self._env.begin(db=self._db, write=True) as txn:
            current_values = self._decode_value(txn.get(encoded_uuid) or DEFAULT_VALUE)
            current_values.update(attributes)
            txn.put(encoded_uuid, self._encode_value(current_values))

    def put_replay_attribute(self, uuid, attribute, value):
        self.put_replay_attributes(uuid, [(attribute, value)])

    def get_replay_attribute(self, uuid, attribute):
        return self.get_replay_attributes(uuid).get(attribute)

    def get_replay_attributes(self, uuid):
        with self._env.begin(db=self._db) as txn:
            return self._decode_value(
                txn.get(self._encode_key(uuid)) or DEFAULT_VALUE
            )

    def _encode_key(self, uuid):
        return uuid.encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return key_bytes.decode('utf-8')

    def _encode_value(self, value):
        return json.dumps(value).encode('utf-8')

    def _decode_value(self, value_bytes: bytes):
        return json.loads(value_bytes.decode('utf-8'))
