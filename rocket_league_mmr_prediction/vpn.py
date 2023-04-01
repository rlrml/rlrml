import backoff
from sdbus_block import networkmanager as nm


def _any_vpn(connection_settings):
    try:
        return 'vpn' in connection_settings.get_settings()['connection']['type']
    except (KeyError, nm.exceptions.NmSettingsPermissionDeniedError):
        return False


class VPNCycler:
    """Automatically cycle between network manager vpn connections using dbus."""

    def __init__(self, connection_selector=_any_vpn):
        """Initialize the VPN cycle by using the connection_selector to select vpn connections."""
        self._settings = nm.NetworkManagerSettings()
        self._nm = nm.NetworkManager()
        self._connection_selector = connection_selector
        self._reinitialize_connections()

    @property
    def _active_connection(self):
        return self._connections[self._active_connection_index]

    def _reinitialize_connections(self):
        connections = [
            (path, nm.NetworkConnectionSettings(path))
            for path in self._settings.list_connections()
        ]

        self._connections = [
            connection for connection in connections
            if self._connection_selector(connection[1])
        ]

        self._active_connection_index = -1

    def _currently_active_known_connections(self):
        connection_paths = [c[0] for c in self._connections]
        for path in self._nm.active_connections:
            active_connection = nm.ActiveConnection(path)
            if active_connection.connection in connection_paths:
                yield path

    def _deactivate_unselected_connections(self):
        for path in self._currently_active_known_connections():
            self._deactivate_connection(path)

    def activate_next_connection(self):
        """Activate the next vpn connection."""
        if self._active_connection is None:
            self._active_connection_index = 0
        else:
            self._active_connection_index = \
                (self._active_connection_index + 1) % len(self._connections)

        self._deactivate_unselected_connections()
        self._activate_connection(self._active_connection[0])

    def _deactivate_connection(self, path):
        return self._nm.deactivate_connection(path)

    def _activate_connection(self, path):
        return self._nm.activate_connection(path, "/", "/")

    def cycle_vpn_backoff(self, *args, **kwargs):
        """Cycle vpn when we hit backoff."""
        cycle_condition = kwargs.get('cycle_vpn_condition', lambda x: True)
        existing_on_backoff = kwargs.get('on_backoff', lambda x: None)

        def maybe_cycle_vpn(details):
            if cycle_condition(details):
                self.activate_next_connection()
            existing_on_backoff(details)

        kwargs['on_backoff'] = maybe_cycle_vpn
        return backoff.on_exception(*args, **kwargs)
