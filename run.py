import logging
import multiprocessing as mp
import sys
from typing import Optional

from runtime.config import Config, EnvSettings
from runtime.task_dispatcher import TaskDispatcher
from utils import setup_logging


logger = logging.getLogger("main")


def main():
    envs = EnvSettings()

    setup_logging(debug=envs.debug)

    logger.info(">>> System Boot Sequence Initiated <<<")

    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing Context: spawn (Enforced)")
    except RuntimeError as e:
        logger.warning("Multiprocessing context already set: %s", e)

    cfg = Config.load(envs)
    dispatcher: Optional[TaskDispatcher] = None

    try:
        logger.info("Initializing Task Dispatcher...")
        dispatcher = TaskDispatcher(cfg)

        logger.info("Services Starting... (Press Ctrl+C to stop)")
        dispatcher.run()  # This blocks until shutdown

    except KeyboardInterrupt:
        logger.info("Shutdown Signal Received (SIGINT).")
    except Exception as e:
        logger.critical("Fatal System Error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if dispatcher:
            logger.info("Executing Graceful Shutdown...")
            try:
                dispatcher.cleanup()
            except Exception as e:
                logger.error("Cleanup failed: %s", e, exc_info=True)

        logger.info(">>> System Shutdown Complete <<<")


if __name__ == "__main__":
    main()
