import random

from numpy.random import seed
import tensorflow as tf

# Random seeds for better replication, remove for eval
seed(1337)
random.seed(1337)
tf.random.set_seed(1337)
tf.keras.utils.set_random_seed(1337)
