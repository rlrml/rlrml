import backoff
from flask import Flask

from .. import tracker_network
from .. import vpn


app = Flask(__name__)
vpn_cycler = vpn.VPNCycler()
scraper = tracker_network.CloudScraperTrackerNetwork()
base_uri="https://api.tracker.gg"


def _constant_retry(constant):
    def get_value(exception):
        value_to_return = constant
        return value_to_return
    return get_value


@vpn_cycler.cycle_vpn_backoff(
    backoff.runtime,
    tracker_network.Non200Exception,
    giveup=lambda e: e.status_code not in (429, 403),
    value=_constant_retry(4)
)
def _do_get(path):
    return scraper._get(f"{base_uri}/{path}")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy(path):
    return _do_get(path)
