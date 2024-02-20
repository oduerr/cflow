import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input
import numpy as np

def print_hello():
    print("Hello from clflow.utils")

def scale_df(dat_tf):
    dat_min = tf.reduce_min(dat_tf, axis=0)
    dat_max = tf.reduce_max(dat_tf, axis=0)
    dat_scaled = (dat_tf - dat_min) / (dat_max - dat_min)
    return dat_scaled

