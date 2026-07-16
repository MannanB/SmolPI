# main.py
import argparse
from dataclasses import asdict

import wandb

from algorithms import create_algorithm
from core.config import load_config
from datasets import create_dataloader
from environments import create_environment
from models import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model = create_model(config.model).to(config.device)
    environment = create_environment(config.environment)
    dataloader = create_dataloader(config.dataset)

    wandb_run = None
    if config.use_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
        )

    algorithm = create_algorithm(config.algorithm, model, config.device, wandb_run)

    algorithm.train(dataloader)


if __name__ == "__main__":
    main()
