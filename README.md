# Facades and Openings Detection Based on Different Deep Learning Models
### Authors: Xiaorang Guo, Qunyou Liu and Yiwen Ma

## Introduction
Our project is mainly about detecting the facades and openings of the raw images grabbing from Google Street View. The work is mainly divided into three parts.
The first part is to grab the data from Google Street-View, and the main functions and some small samples are in the folder "downloading data"
The second part is to use three methods to filter the raw images downloaded from google and make the first prediction with an SSD_facades model. Code for this part is in the folder "Data Filtering"
The last part is to use the pix2pix model to detect the objects and make some improvements for this model. Code for this part is in folder "pix2pix"

## Contents
### Downloading the images(folder "downloading data")
This folder is used for grabbing data from Google Street-View API. In the make_dataset.py, we allow it to download 6 pictures per location， 3 for headings and 2 for pitches。
This part is commented in the function img_to_db. Note: When using this part, please comment on the part for downloading to collect only one picture per location.
After trying with a small dataset, we picked the best heading and pitch, then use this parameter to download to the whole dataset.

- Examples of how to use 
Examples for using this is in the Jupyter Notebooks downloading picture.

NOTE: Can't run it directly!
When using this downloading function, we need two things: the key to google street API and the shapefile which contains the location of each building.
We regret that we cannot provide the shape file for this project because it is privately provided by the lab.
In the Unlable folder, we provided a few pictures which are downloaded from google. In this project, we downloaded around 6000 pictures.

### Filter the images(folder "Data Filtering")

In this part, we designed three methods to filter the raw images: building detection, green area detection and car detection.

#### place365 model
First of all, the code in the modeling folder is used. This model is a CNNs based pre-trained model, named the place365 model, which can detect the the probability of different scenario of the image.
The pre-trained model contains a text file, describing 400+ scenarios it can detect. The class contains both indoors and outdoors scenarios. In this project, we selected several outside scenarios in a .csv file as building list because we focused on the facade detection. Then we applied the model to detect the place of images. Based on the results it output, if the place is contained in the building list, the image is regarded as desired facade, so we saved the image in the folder 'prediction_good'. Otherwise, it would be stored in 'prediction_bad'.   
The example is in file demo and the source code has already been functionalized in model_test.py. Note: the directory needs to be changed when the function is run.

#### green area detection
The second filter we designed based on computer version, which uses the cv2 library as a toolbox to detect the pixel color of the images.
There are still many images in 'prediction_good' that contain a large part of plants like trees and grass. Apparently, these kinds of images are not our desired testing data, which should be filtered out.
In our code, we used the cv2 library to detect the ratio of green area which usually represents trees, plants or grass. A image which contains a green ratio larger than 30% of the whole image would be filtered out ('detect_green_removed'), otherwise, we keep it('detect_green_retained'). 

#### car detection
The filter is based on a Nvidia object detection model. It uses the SSD model from PyTorch to detect objects in an image. As we know, the SSD model can detect more than 80 classes, such as a car, human, and so on. A set of labels and its corresponding probabilities would be reported by the model. We observe that there are some images in the folder 'detect_green_removed' still have some noise such as a big car which covers the facade. Thus, we applied the SSD model to our dataset and filter the image based on two critiria: if the car area ratio is greater than 0.12  and the confidence of detection reported by the SSD model is larger than 70%
The output images would be saved separately as before in two folders('detect_car_removed') and ('detect_car_retained')
Attention! This model can only be run with the help of Cuda. Please feel free to change the target object. Many other objects might exist in the images as well, like the person, etc., in other scenarios. You can change it according to the class number, which can be seen in the variable class.

### pix2pix model(folder "pix2pix")

#### working with pix2pix model
In this part, we used the pix2pix model to make the building detection on the images we got after the data filtering process. When applying the pre-trained model, the result of prediction is not good because the building type of the ground truth dataset are  different the images from google street view. Also, almost all the training images contains only facade over the whole image area. They do not contain any background noise as the images we grabbed from the real world. 
There are 12 classes in the original model, and in this project, we only focused on four of them (background, facade, opening, cover). 
We did several steps to make some improvements to this model:
- First, we merged the 12 colors into 4 and tested both the original dataset and our new dataset. What we get at this point is the predicted data with only 4 colors, but still, the background noise would be misclassified. 
- Secondly, we manually revised our new dataset's result to let these non-facades part to be backgrounds and we revised 70 pictures. We put these 70 images along with the 400 original images into the training datasets. We retrained the model and applied it to get the new results.

#### code for model pix2pix
- the pix2pix folder contains a readme file that is written by the original author, and the specific information about this model is in that readme file.

- The jupyter notebook "facades_baseline" contains all steps to train and to test. Tips, it is better to train the model on Colab. 

- we revised the util.py and visualizer.py under the path '/pix2pix/pytorch-CycleGAN-and-pix2pix/util/' in order to merge the 12 colors into 4 colors and calculate the ratio between the facades and backgrounds and openings and facades. After merging the color, If you don't need to calculate the ratio(maybe you are adjusting the parameters), please switched back to the old version, because it is faster.

- In the dataset folder, there are three folders(train, test, val), we actually have 470 pictures in train dataset, 104 pictures in val and around 2100 pictures in test. For convenience, we only provide a few samples.

## Thanks
Upon the completion of our project, we would like to express our gratitude to the host lab IMAC and our mentors Mr. Alireza Khodaverdian and Mr. Saeed for their guide and help.
