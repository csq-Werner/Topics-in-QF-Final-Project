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

obj = Pricer(model='NIG', cp='call', exercise='American')
save_name = '_'.join([obj.model,obj.exercise[:2],obj.cp])
obj.data_preparer(train_size=50000,test_size=100)
obj.net_builder(3, 3, 500)

obj.set_name(save_name)
obj.load_model()

# Evaluate the model over the test set
pred = obj.predict()
test_price = obj.test_price_fun()
print("MAE Loss:", obj.compare(pred,test_price))

# Plot curves

x_sorted = np.sort(obj.test_tensor['x'],axis=0)
S_sorted = np.exp(x_sorted)
obj.plot(S_sorted, T=1)

# Plot loss
with open(os.path.join(save_name, "loss_list.json"), "r") as f:
    loss_data = json.load(f)

train_loss = loss_data["train_loss"]
val_loss = loss_data["val_loss"]
plt.subplot(2, 2, 3)
# plt.figure()
epochs = list(range(len(train_loss)))
plt.plot(epochs, train_loss, color="orange", label="train loss")
plt.plot(epochs, val_loss, color="deepskyblue", label="val loss")
plt.legend()
plt.show()