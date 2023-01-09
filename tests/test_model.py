from tests import _PROJECT_ROOT
import os
import torch
from src.models.model import MyAwesomeModel
import pytest
model = MyAwesomeModel()


def test_model_output():
    assert model(torch.ones(1,1,28,28)).shape==(1,10),"Output of model is not correct dimensions"



#def test_error_on_wrong_shape():
  # with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
   #   model(torch.randn(1,2,3))