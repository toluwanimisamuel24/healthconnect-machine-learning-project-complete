"""
logging_config.py
------------------
Week 6 addition.

Week 5 had every entry-point script (pipeline.py, train_baseline.py) call
`logging.basicConfig(...)` independently at import time. This looked fine in
isolation but has a real bug: Python's logging.basicConfig() only takes
effect on its FIRST call per process -- every later call is silently a
no-op unless force=True is passed. Because pipeline.py is imported (and
therefore its basicConfig() executed) before train_baseline.py's own
basicConfig() call ever runs, train_baseline.py's logging configuration was
always being discarded. It happened to look correct anyway because both
calls used identical settings -- but the moment Week 6 needed to add a file
handler (this module), that silent discarding would have meant the file
handler only sometimes got attached, depending on which script imported
which other script first.

Centralising it here, called explicitly once per process by each script's
`if __name__ == "__main__":` entry point, removes that fragility.
"""

import logging

from . import config

_configured = False


def setup_logging(log_to_file: bool = True) -> None:
    """
    Configure root logging once per process. Safe to call multiple times --
    subsequent calls are no-ops, so importing several src modules that each
    call this at their entry point never double-attaches handlers.
    """
    global _configured
    if _configured:
        return

    handlers = [logging.StreamHandler()]

    if log_to_file:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.PIPELINE_LOG_PATH))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,  # ensure this configuration wins regardless of import order
    )
    _configured = True
