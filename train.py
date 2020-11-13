import numpy as np
import tensorflow as tf

from data_utils import MITLoader, DataGenerator

if __name__ == "__main__":
    mit_loader = MITLoader()
    training_set = DataGenerator(mit_loader, 'train', 64)
    validation_set = DataGenerator(mit_loader, 'valid', 64)

    