import tensorflow as tf
from tensorflow.keras.layers import *
import os
import numpy as np
import scipy
import sobol_seq
import json
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
import json
from model.pricer import Pricer


data_dir = './'

"""## Training routine"""
if __name__ == "__main__":
    ## create the pricer object
    obj = Pricer(model='VG', cp='call', exercise='American')
    save_name = '_'.join([obj.model,obj.exercise[:2],obj.cp])
    plot_paras = [{'T':1}]

    ## prepare all pre-calculation
    obj.data_preparer(train_size=500000,test_size=10000)

    ## build the network
    obj.net_builder(3, 3, 500)

    train_loss_list, val_loss_list = [], []

    ## train the network
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_loss, val_loss = obj.train(opt = opt, n_epochs=15, plot_paras = plot_paras)
    train_loss_list += train_loss
    val_loss_list += val_loss

    opt = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 15*(obj.train_size//200), 0.1))
    train_loss, val_loss = obj.train(opt = opt, n_epochs=15, plot_paras = plot_paras)
    train_loss_list += train_loss
    val_loss_list += val_loss

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_loss, val_loss = obj.train(opt = opt, n_epochs=15, plot_paras = plot_paras)
    train_loss_list += train_loss
    val_loss_list += val_loss

    ## save the model
    obj.set_name(save_name)
    obj.save_model(train_loss_list, val_loss_list)

    ## evaluate the model over the test set
    pred = obj.predict()
    test_price = obj.test_price_fun()
    print(obj.compare(pred,test_price))