from tests import _PATH_DATA
import torch
import os
import pytest

N_train = 40000
N_test = 5000
#test length of training data and length of test data
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_length():
    training_data = torch.load(os.path.join(_PATH_DATA, "processed/train_images_norm.pt"))
    test_data = torch.load(os.path.join(_PATH_DATA, "processed/test_images_norm.pt"))
    assert len(training_data) == N_train and len(test_data) == N_test, "Dataset did not have the correct number of samples"
#test shape of data
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_shape():
    training_data = torch.load(os.path.join(_PATH_DATA, "processed/train_images_norm.pt"))
    assert [i.shape == [28,28] for i in training_data],"Training data does not have the correct shape"
#test that all different labels are present in both test and training data
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_all_labels():
    training_labels = torch.load(os.path.join(_PATH_DATA, "processed/train_labels.pt"))
    test_labels = torch.load(os.path.join(_PATH_DATA, "processed/test_labels.pt"))
    assert torch.all(torch.unique(training_labels) == torch.tensor([0,1,2,3,4,5,6,7,8,9])) and torch.all(torch.unique(test_labels) == torch.tensor([0,1,2,3,4,5,6,7,8,9])),"Not all labels in dataset"


