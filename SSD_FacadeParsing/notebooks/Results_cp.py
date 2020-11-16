#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from math import sqrt as sqrt
import numpy as np
import itertools
from torch.autograd import Function
import torch.nn.init as init
from ssd_project.model import ssd
from ssd_project.functions.detection import *
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#best_model = torch.load("../saved_models/BEST_model_ssd300.pth.tar")
#best_model = torch.load("../saved_models/BEST_model_ssd300_bgpr.pth.tar")
best_model = torch.load("../saved_models/first_best_model_ssd300.pth.tar")
model = ssd.build_ssd(num_classes = 4)
model.load_state_dict(best_model["model_state_dict"])
device = "cuda"
model = model.to(device)

epochs_trained = best_model["epoch"]
best_avg_loss = best_model["loss"]
t_loss_bvals = best_model["training_losses_batch_values"]
t_loss_bavgs = best_model["training_losses_batch_avgs"]
v_loss_bvals = best_model["validation_losses_batch_values"]
v_loss_bavgs = best_model["validation_losses_batch_avgs"]

best_avg_loss

imgs = glob.glob("../data/ssd_ilija_data/original_images/img_resized_*")
#imgs = glob.glob("/data/ssd_ilija_data/original_images/img_resized_*")
imgs.sort()

#for 
print(len(imgs))
img, bboxes, labels, scores = predict_objects(model, imgs[55], min_score=0.2, max_overlap = 0.01, top_k=200)

annotated_img = draw_detected_objects(imgs[55], bboxes, labels, scores)

FT.to_pil_image(annotated_img)

imgplot = plt.imshow(annotated_img)
plt.show()

epochs_arr = list(range(epochs_trained+1))
plt.plot(epochs_arr, t_loss_bavgs, label = "Training Average Loss Per Epoch")
plt.plot(epochs_arr, v_loss_bavgs, label = "Validation Average Loss Per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

epochs_arr = list(range(epochs_trained+1))
plt.plot(epochs_arr, t_loss_bvals, label = "Training Random Value Loss Per Epoch")
plt.plot(epochs_arr, v_loss_bvals, label = "Validation Random Value Loss Per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
