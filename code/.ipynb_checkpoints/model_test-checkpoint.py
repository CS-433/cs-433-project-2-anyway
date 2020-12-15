import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from imutils import paths
from places365.detector_best import build_detect
import os
from helpers import *


# The following are global values
class_save = []
idxes = []
names = []
probility = []
reasons = []
save = []

def f_detect(img_name):  # new for direct view
    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(
        model_file,
        map_location=lambda storage,
        loc: storage)
    state_dict = {
        str.replace(
            k,
            'module.',
            ''): v for k,
        v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # Load the building list
    df = pd.read_csv("../building_class_list.csv")
    buildings = list(df['class'])

    # Open the img
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    prob_all, label_all = h_x.sort(0, True)

    # find the best label which is building (from the first two highest prob)
    prob_all = list(prob_all)
    label_all = list(label_all)
    best_prob = prob_all[0]
    best_label = label_all[0]
    count = 0
    while (best_label not in buildings and count < 3):
        count += 1
        prob_all.remove(best_prob)
        label_all.remove(best_label)
        if not prob_all:
            break
        else:
            best_prob = prob_all[0]
            best_label = label_all[0]

    return best_prob, best_label

# The following code shows how to use place365 to filter data
# Load the test image
def place365_filter():
    # Load the test image   
    imgPath = sorted(list(paths.list_images("./test_imgs")))
    # Path to save the dataset
    path_to_save = "./test_imgs/save"

    # Load the classes
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # Load the building list
    df = pd.read_csv("../building_class_list.csv")
    buildings = list(df['class'])

    class_save = []
    idxes = []
    names = []
    probility = []
    reasons = []
    save = []

    while(imgPath):
        image = imgPath.pop()
        # Building detection
        try:
            prob,label = f_detect(image)
        except:
            continue
        print(image)
        name = image.replace("./test_imgs","")# this one should be modified, if different name
        names.append(name)
        class_save.append(classes[label])
        idxes.append(label)
        detect = 0

        img = Image.open(image)
        if label in buildings: # Check if one of the top two classes detected is building
            if prob > 0.07: # Filter out the image if the probability is less than 0.1
                print('Save: ', name)
                path_buildings = path_to_save + name
                img.save(path_buildings)
                detect = 1
                reason = "None"
            else: # write to csv - reason - prob less than threshold
                prob = 0.0
                reason = "prob less than threshold"
                print(reason)
        else: # write to csv - reason - no building detected
            prob = 0.0
            reason = "no building detected"
            print(reason)
        probility.append(prob)
        reasons.append(reason)
        save.append(detect)
    df_class = pd.DataFrame(class_save,columns =['class'])
    df_idxes = pd.DataFrame(idxes)
    df_names = pd.DataFrame(names)
    df_save = pd.DataFrame(save)
    df_probility = pd.DataFrame(probility)
    df_reasons = pd.DataFrame(reasons)
    df_class['names'] = df_names
    #df_class['img_index']
    df_class['idxes'] = df_idxes
    df_class['detected_building'] = df_save
    df_class['probility'] = df_probility
    df_class['reason_unsaved'] = df_reasons
    df_class.to_csv('df_class.csv')
    return df_class

def show_analysis():
    """
    show a little analysis of these data
    """
    df = pd.read_csv("data/stats_facade_detection.csv")
    df.save_after_greenSSD.value_counts()
    df.save_after_original_ssd.value_counts()
    
    temp = df[df['detected_building'] == True]
    mean_1 = temp['probility'].mean()
    
    temp = temp[temp['save_after_greenSSD']==True]
    mean_2 = temp['probility'].mean()
    
    temp = temp[temp['save_after_original_ssd']==True]
    mean_3 = temp['probility'].mean()