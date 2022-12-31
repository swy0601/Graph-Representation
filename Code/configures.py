import os
import torch
from tap import Tap
from typing import List


class DataParser(Tap):
    dataset_dir: str = '../Dataset/Packaged Pkl/input.pkl'
    dataset_name = 'code_hh'
    num_node_features: int = 840
    num_classes: int = 3
    avg_num_nodes: int = 150
    data_split_ratio: List = [0.6, 0.2, 0.2]  # the ratio of training, validation and testing set for random split
    batch_size: int = 15
    seed: int = 1
    k_fold: int = 5


class TrainParser(Tap):
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_epochs: int = 100
    save_epoch: int = 10
    early_stopping: int = 100


class ModelParser(Tap):
    device: str = "cuda"
    checkpoint: str = 'checkpoint'
    hidden_channels: int = 128
    mlp_hidden: int = 64
    model_name = 'dmon'


data_args = DataParser().parse_args(known_only=True)
print("Dataset using:", data_args.dataset_dir)
train_args = TrainParser().parse_args(known_only=True)
model_args = ModelParser().parse_args(known_only=True)
