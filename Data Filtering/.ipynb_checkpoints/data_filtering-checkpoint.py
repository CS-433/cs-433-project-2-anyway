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
# color detection of red and green color in image
# necessary libraries
import matplotlib.patches as patches
from imutils import paths
import argparse
import imutils
import os.path
from cv2 import cv2
from imutils import paths
import re
from helpers import *


# filter with building detection model (breif select)
# please fill the follwoing directory correctly
def filter_with_building_detection(
        working_directory="./SSD_FacadeParsing/",
        path_to_save_good="./data/prediction_good/",
        path_to_save_bad="./data/prediction_bad/",
        file_good="./data/prediction_good/ratios_good.csv",# the file used for storing
        file_bad="./data/prediction_bad/ratios_bad.csv", # the file used for storing
        data_directory="./data/original_images/*"):
    if('SSD_FacadeParsing' in os.getcwd()):
        None
    else:
        print("you seem to use wrong directory, then I will change it to the right one")
        os.chdir(working_directory)  # change the directory to your working space
    imgs = glob.glob(data_directory)
    # lauching the model firstly
    if not torch.cuda.is_available():
        best_model = torch.load("./saved_models/Best_model_ssd300.pth.tar",map_location=torch.device('cpu'))
        model = ssd.build_ssd(num_classes = 4)
        model.load_state_dict(best_model["model_state_dict"])
        device = "cpu"
        model = model.to(device)
        print("Oh! you are trying to use CPU, are you sure??")
    else:
        best_model = torch.load("./saved_models/Best_model_ssd300.pth.tar")
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

    imgs.sort()
    if os.path.exists(file_good):
        locFile_good = open(file_good, "a")
    else:
        locFile_good = open(file_good, "w+")
        locFile_good.write("Names, Ratios\n")
    if os.path.exists(file_bad):
        locFile_bad = open(file_bad, "a")
    else:
        locFile_bad = open(file_bad, "w+")
        locFile_bad.write("Names, Ratios\n")
    for i, img in enumerate(imgs):
        pred_img, bboxes, labels, scores = predict_objects(
            model, img, min_score=0.2, max_overlap=0.01, top_k=200)
        annotated_img = FT.to_pil_image(
            draw_detected_objects(
                img, bboxes, labels, scores))
        num = return_num_img(img)
        name = "annotated_img_" + num + ".png"
        path_img_good = path_to_save_good + name
        path_img_bad = path_to_save_bad + name
        # Windows Ratio calculation
        building_area = (bboxes[0][2] - bboxes[0][0]) * \
            (bboxes[0][3] - bboxes[0][1])
        position_building = [
            bboxes[0][0],
            bboxes[0][1],
            bboxes[0][2],
            bboxes[0][3]]
        windows_area = 0
        flag = True  # assume it in the building
        for j in range(1, len(bboxes)):
            windows_area = windows_area + \
                (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])
            position_window = [
                bboxes[j][0],
                bboxes[j][1],
                bboxes[j][2],
                bboxes[j][3]]
            flag = flag & compare(position_window, position_building)
        if (flag == False):
            print('The window outside the building')
        ratio = windows_area / building_area
        if ((ratio > 0.025) & (ratio < 0.85) & flag):
            # if ((ratio>0.025)&(ratio<0.85)):
            pred_img.save(path_img_good, "PNG")
            locFile_good.write(
                name + "," + str(float(ratio.detach().numpy())) + '\n')
        else:
            pred_img.save(path_img_bad, "PNG")
            locFile_bad.write(
                name + "," + str(float(ratio.detach().numpy())) + '\n')
    print("The good guys and bad guys are seperated, please check {}, {} to find you result".format(file_good,file_bad))
    locFile_bad.close()
    locFile_good.close()


def filter_with_green():
    save = []
    names = []
    reasons = []
    working_directory="./SSD_FacadeParsing/"
    if('SSD_FacadeParsing' in os.getcwd()):
        None
    else:
        print("you seem to use wrong directory, then I will change it to the right one")
        os.chdir(working_directory)  # change the directory to your working space
    imgPaths = sorted(list(paths.list_images("./data/original_images")))
    path_to_save_bad = "./data/detect_green_removed/"
    path_to_save_good = "./data/detect_green_retained/"
    for imgPath in imgPaths:
        number = get_numbers_from_filename(imgPath)
        image = cv2.imread(imgPath)  # load image
        area_green = green_area(image)
        total_area = image.shape[0] * image.shape[1]
        ratio_green = area_green / total_area
        # print(ratio_green)
        saved = 0
        # Store the images with green_detected
        if ratio_green > 0.25:
            # num = return_num_img(imgPath)
            name = "img_" + number + ".jpg"
            path_img_bad = path_to_save_bad + name
            saved = 0
            reason = 'too many greens'
            cv2.imwrite(path_img_bad, image)
        else:
            # num = return_num_img(imgPath)
            name = "img_" + number + ".jpg"
            path_img_good = path_to_save_good + name
            saved = 1
            reason = 'None'
            cv2.imwrite(path_img_good, image)
        names.append(name)
        save.append(saved)
        reasons.append(reason)
    # Create an .csv file to store all generated data, and recorde the index
    # of saved or deleted image
    df_green = pd.DataFrame(names, columns=['name'])
    df_save = pd.DataFrame(save)
    # df_names = pd.DataFrame(names)
    df_reason = pd.DataFrame(reasons)
    df_green['save'] = df_save
    # df_class['names'] = df_names
    df_green['reason'] = df_reason
    return df_green



# filter cars with original SSD
# For now, download a new model from nvidia. It is a pre-trained model.
# The model is object detection model, that could be used to detect car, person, and something like that
# In this project, we used it to detect car. And we think we can use it to detect person as well, if there are always people on street.
# Modify is very easy
def SSD_filter_car():
    precision = 'fp32'
    if not torch.cuda.is_available():
        device = "cpu"
        print("Oh! you are trying to use CPU, are you sure??")
    else:
        device = "cuda"
    ssd_model = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_ssd',
        model_math=precision,map_location=torch.device(device))
    utils = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_ssd_processing_utils',map_location=torch.device(device))
    ssd_model.to(device)
    ssd_model.eval()
    # The following shows how it works
    imgPath = sorted(list(paths.list_images("./data/detect_green_retained")))
    path_to_save_good = "./data/detect_car_retained/"
    path_to_save_bad = "./data/detect_car_removed/"
    save = []
    names = []
    while (imgPath):
        # Grab data in a group of three
        if len(imgPath) < 3:
            uris = []
            for i in range(len(imgPath)):
                uris.append(imgPath.pop(0))
        else:
            uris = imgPath[0:3]
            imgPath = imgPath[3:]

        # Detect car
        try:
            images = [cv2.imread(k) for k in uris]
            inputs = [utils.prepare_input(uri) for uri in uris]
            tensor = utils.prepare_tensor(inputs, precision == 'fp16')
            detections_batch = ssd_model(tensor)
            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [
                utils.pick_best(
                    results,
                    0.40) for results in results_per_input]
            classes_to_labels = utils.get_coco_object_dictionary()
        except BaseException:
            continue

        for image_idx in range(len(best_results_per_input)):
            # Show original, denormalized image...
            # image = inputs[image_idx]
            image = images[image_idx]
            # ...with detections
            bboxes, classes, confidences = results_per_input[image_idx]
            area = 0
            confidence = 0
            if (3 in classes):  # 3 means car in SSD_Model
                area = 0
                for idx in np.where(classes == 3)[0]:
                    idx = int(idx)
                    left, bot, right, top = bboxes[idx]
                    area = area + (right - left) * (top - bot)
                    confidence = max(confidence, confidences[idx])
            #             print(area)
            #             print(confidence)
            if (area > 0.12) & (confidence > 0.7):
                # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                number = get_numbers_from_filename(uris[image_idx])
                name = "img_" + number + ".jpg"
                saved = 0
                path_img_bad = path_to_save_bad + name
                cv2.imwrite(path_img_bad, image)
            else:
                # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                number = get_numbers_from_filename(uris[image_idx])
                name = "img_" + number + ".jpg"
                saved = 1
                path_img_good = path_to_save_good + name
                cv2.imwrite(path_img_good, image)
            names.append(name)
            save.append(saved)
    # A new .csv would be generated here, save it.
    df_car = pd.DataFrame(names, columns=['name'])
    df_save = pd.DataFrame(save)
    df_car['save'] = df_save
    df_car.to_csv('df_car.csv')
    return df_car


# the following code helps plot the annotated image
def plot_annotated():
    imgs = sorted(glob.glob("./data/detect_car_retained/*"))
    path_to_save = "./data/result/"
    for i, img in enumerate(imgs):
        pred_img, bboxes, labels, scores = predict_objects(
            model, img, min_score=0.2, max_overlap=0.01, top_k=200)
        annotated_img = FT.to_pil_image(
            draw_detected_objects(
                img, bboxes, labels, scores))
        number = get_numbers_from_filename(img)
        path_img = path_to_save + "annotated_img_" + number + ".png"
        annotated_img.save(path_img, "PNG")
