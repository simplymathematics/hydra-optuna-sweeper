# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="torch-conf", config_name="config")
def torch_sphere(cfg: DictConfig) -> float:
    x: torch.Tensor = torch.tensor(cfg.x)
    y: torch.Tensor = torch.tensor(cfg.y)
    return (x**2 + y**2).item()


if __name__ == "__main__":
    torch_sphere()
