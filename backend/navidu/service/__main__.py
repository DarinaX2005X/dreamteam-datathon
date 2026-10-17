import logging
import os

import uvicorn

from .app import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(), host=os.getenv("NAVIDU_HOST", "127.0.0.1"),
                port=int(os.getenv("NAVIDU_PORT", "8000")), access_log=False)


if __name__ == "__main__":
    main()
