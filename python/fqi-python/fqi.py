from os import stat
from typing import Dict

import logging
from pathlib import Path

import cherry as ch
import torch as th
import torch.nn as nn
import torch.optim as optim
import pandas as pd

logger = logging.getLogger(__name__)


class FQI:
    def __init__(self, device) -> None:
        self.device = device

        self.exp_attr = None

        self.bf = None
        self.bf_valid = None

        self.learning_net = None
        self.target_net = None

        self.last_action = None
        self.last_state = None

    def initialize(self, run: int, exp_attr: Dict):
        self.exp_attr = exp_attr
        self._init_bfs()
        self._load_data_log(run)
        self._init_model()

    def fit(self):
        logger.info("training...")
        pass

    def _init_bfs(self):
        self.bf = ch.ExperienceReplay(device=self.device)
        self.bf_valid = ch.ExperienceReplay(device=self.device)

    def _load_data_log(self, run: int):
        path_to_base = Path(__file__).resolve().parents[3]
        trace_dir = path_to_base / self.exp_attr["datalog"]
        # load training data
        trace_file = trace_dir / f"traces-{run}.csv"
        logger.info(f"Training set path: {trace_file}")
        self._load_data_log_file(trace_file, self.bf)

        # load validation data
        trace_file = trace_dir / f"traces-{(run + 1) % self.attr['fqi-numDataset']}.csv"
        logger.info(f"Validation set path: {trace_file}")
        self._load_data_log_file(trace_file, self.bf_valid)

    @staticmethod
    def _load_data_log_file(trace_file: Path, bf: ch.ExperienceReplay):
        bf.empty()

        df = pd.read_csv(trace_file)
        for idx, row in df.iterrows():
            bf.append(
                row["previous state"],
                row["action"],
                row["reward"],
                row["new state"],
                1 - row["terminal"],
            )

    @staticmethod
    def _build_model(in_dim, hidden, out_dim):
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _init_model(self):
        self.learning_net = self._build_model(
            self.exp_attr["state-len"],
            self.exp_attr["fqi-hidden"],
            self.exp_attr["num-actions"],
        )

        self.target_net = self._build_model(
            self.exp_attr["state-len"],
            self.exp_attr["fqi-hidden"],
            self.exp_attr["num-actions"],
        )

