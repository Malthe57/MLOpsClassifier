from tests import _PATH_DATA
import torch
import os

training_data = torch.load(os.path.join(_PATH_DATA, "processed/train_images_norm.pt"))
test_data = torch.load(os.path.join(_PATH_DATA, "processed/test_images_norm.pt"))
training_labels = torch.load(os.path.join(_PATH_DATA, "processed/train_labels.pt"))
test_labels = torch.load(os.path.join(_PATH_DATA, "processed/test_labels.pt"))
N_train = 25000
N_test = 5000
#test length of training data and length of test data
def test_data_length():
    assert len(training_data) == N_train and len(test_data) == N_test
#test shape of data
def test_shape():
    assert [i.shape == [28,28] for i in training_data]
#test that all different labels are present in both test and training data
def test_all_labels():
    assert torch.all(torch.unique(training_labels) == torch.tensor([0,1,2,3,4,5,6,7,8,9])) and torch.all(torch.unique(test_labels) == torch.tensor([0,1,2,3,4,5,6,7,8,9]))