from tests import _PROJECT_ROOT
import os
import torch
from src.models.model import MyAwesomeModel
import pytest
model = MyAwesomeModel()

@pytest.mark.parametrize("test_input,expected", [(torch.ones(1,1,28,28), (1,10)), (torch.zeros(64,1,28,28), (64,10))])
def test_model_output(test_input,expected):
    assert model(test_input).shape==expected,"Output of model is not correct dimensions"



#def test_error_on_wrong_shape():
  # with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
   #   model(torch.randn(1,2,3))