# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import torch
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train = np.load(input_filepath +"/train_{}.npz".format(0))
    images_vec = train['images']
    labels_vec = train['labels']
    for i in range(1,5):
        train = np.load(input_filepath + "/train_{}.npz".format(i))
        images = train['images']
        labels = train['labels']
        images_vec = np.concatenate((images_vec, images))
        labels_vec = np.concatenate((labels_vec, labels))

    images_tens = torch.tensor(images_vec)
    labels_tens = torch.tensor(labels_vec)
    images_norm = (images_tens-images_tens.mean())/images_tens.std()
    torch.save(images_norm, os.path.join(output_filepath, 'train_images_norm.pt'))
    torch.save(labels_tens, os.path.join(output_filepath , 'train_labels.pt'))

    #process test set
    test = np.load(input_filepath + "/test.npz")
    test_images = test['images']
    test_labels = test['labels']
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels)
    test_images_norm = (test_images-test_images.mean())/test_images.std()
    torch.save(test_images_norm, os.path.join(output_filepath, 'test_images_norm.pt'))
    torch.save(test_labels, os.path.join(output_filepath ,'test_labels.pt'))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
