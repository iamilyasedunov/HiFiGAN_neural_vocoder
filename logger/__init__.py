from .logger import *
from .visualization import *
from datetime import datetime
import json
import dataclasses


def write_json(content, fname):
    fname = Path(fname)
    content["device"] = content["device"].__str__()
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_writer(config):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger_ = logging.getLogger(config.name)
    logger_.setLevel(log_levels[config.verbosity])
    save_dir = Path("saved/")
    exper_name = config.exper_name
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")
    _save_dir = save_dir / "models" / exper_name / run_id
    _log_dir = save_dir / "log" / exper_name / run_id

    # make directory for saving checkpoints and log.
    exist_ok = run_id == ""
    _save_dir.mkdir(parents=True, exist_ok=exist_ok)
    _log_dir.mkdir(parents=True, exist_ok=exist_ok)
    config.save_dir = _save_dir.__str__()
    config.log_dir = _log_dir.__str__()
    print(f"log_dir: {_log_dir}")
    print(dataclasses.asdict(config))
    write_json(dataclasses.asdict(config), save_dir / "config.json")

    writer = WanDBWriter(dataclasses.asdict(config), logger_)
    return writer
