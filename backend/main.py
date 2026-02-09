from flask import Flask
import os
from typing import Optional

app = Flask(__name__)


@app.route('/')
def get_academy_status() -> str:
    """Return the current status of MAPS Academy Brain."""
    return "The MAPS Academy Brain is Online!"


def get_port() -> int:
    """Get the port from environment variable or default to 10000."""
    return int(os.environ.get("PORT", 10000))


if __name__ == "__main__":
    port = get_port()
    app.run(host='0.0.0.0', port=port, debug=False)
