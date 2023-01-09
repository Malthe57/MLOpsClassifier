import argparse
import sys

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

test_img = torch.load("data/processed/test_images_norm.pt")
test_label = torch.load("data/processed/test_labels.pt")
test = {"images": test_img, "labels": test_label}
from model import MyAwesomeModel


# create trainLoader
class tloader(Dataset):
    def __init__(self, train):
        self.img_labels = train["labels"]
        self.img = train["images"]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.img_labels[idx]
        return image, label


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    wandb.watch(model, log_freq=100)
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    test_set = test
    # Create test loader
    testloader = DataLoader(tloader(test_set), batch_size=64, shuffle=True)
    running_acc = []
    for images, labels in testloader:
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        running_acc.append(accuracy.item())
    print(f"Accuracy: {np.mean(running_acc) * 100}%")


cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
