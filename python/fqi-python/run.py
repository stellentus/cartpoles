from pathlib import Path
from typing import Dict
import click
import json
from fqi import FQI
import torch as th
from sklearn.model_selection import ParameterGrid


device = th.device("cuda" if th.cuda.is_available() else "cpu")


def run_with_config(config: Dict):
    fqi = FQI(device)
    fqi.initialize()


@click.command()
@click.option("--config", "-c", required=True, help="Path to config file.")
@click.option("--sweep", default=0, help="Sweep number.")
@click.option("--run", default=0, help="Run number.")
def main(config_path, sweep: int, run: int):
    print(f"Config file: {config_path}")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = json.load(f)
    
    agent_config = config["agent-settings"]
    experiment_settings = config["experiment-settings"]

    sweeps = ParameterGrid(agent_config.get("sweep", {}))
    



    if sweep == -1:
        for s in range(config["sweeps"]):
            run_with_config(config)
    else:
        run_with_config(config)


if __name__ == "__main__":
    main()

