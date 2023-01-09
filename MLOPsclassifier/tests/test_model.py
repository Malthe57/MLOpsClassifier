from tests import _PROJECT_ROOT
import os
import torch
from src.models.model import MyAwesomeModel
model = MyAwesomeModel()

def test_model_output():
    assert model(torch.ones(1,28,28)).shape==(1,10)