import argparse
import sys

import click
import numpy as np
import torch
import torch.nn as nn
from data import mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader, Dataset


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
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()

    trainloader = DataLoader(tloader(train_set), batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_list = []
    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            loss_list.append(loss.item())
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")

    torch.save(model.state_dict(), "trained_model.pt")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    _, test_set = mnist()
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


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
