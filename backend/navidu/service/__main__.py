import os

import uvicorn

from .app import create_app
from .structured_log import configure_logging


def main():
    configure_logging()
    uvicorn.run(create_app(), host=os.getenv("NAVIDU_HOST", "127.0.0.1"),
                port=int(os.getenv("NAVIDU_PORT", "8000")), access_log=False, log_config=None)


if __name__ == "__main__":
    main()
