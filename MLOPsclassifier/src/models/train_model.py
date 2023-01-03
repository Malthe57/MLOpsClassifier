import argparse
import sys
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import click
import numpy as np


train_img = torch.load("data/processed/train_images_norm.pt")
train_label = torch.load("data/processed/train_labels.pt")
trainz = {'images':train_img,'labels':train_label}
from model import MyAwesomeModel


# create trainLoader
class tloader(Dataset):
    def __init__(self, train):
        self.img_labels = train['labels']
        self.img = train['images']

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
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set = trainz

    trainloader = DataLoader(tloader(train_set), batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_list = []
    epochs = 10
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

    torch.save(model.state_dict(), 'models/trained_model.pt')


cli.add_command(train)

if __name__ == "__main__":
    cli()





