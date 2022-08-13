from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import json
train_loss, val_loss = [], []
with open("data.txt", "r") as f:
    for line in f.readlines():
        t, v = line.strip().split()
        train_loss.append(float(t))
        val_loss.append(float(v))

with open("VG_Am_call/loss_data.json", "w") as f:
    json.dump({
        "train_loss": train_loss,
        "val_loss": val_loss
    }, f)
epochs = list(range(len(train_loss)))
plt.plot(epochs, train_loss, "g-", label="train loss")
plt.plot(epochs, val_loss, "r-", label="val loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("VG_Am_call/loss.jpg")