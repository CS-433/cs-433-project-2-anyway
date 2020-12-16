## Title
Facades and Opening Detection Based on Different Deep Learning Models
## Authors
Authors: Xiaorang Guo, Qunyou Liu and Yiwen Ma

## Introduction
Our project is mainly about detecting the facades and openings of the buildings. And the work is mainly divided into three parts.
The first part is grabbing the data from Google Street-View, and the main functions and some small samples are in the folder "downloading data"
The second part is using three methods to filter the raw images downloaded from google and make the first prediction with an SSD_facades model. Code for this part is in the folder " "
The last part is using the pix2pix model to detect the objects and make some improvements for this model. Code for this part is in folder "pix2pix"

## Details for the three part

## Downloading the images(folder "downloading data")
This folder is used for grabbing data from Google Street-View API. In the make_dataset.py, we allow it to download 6 pictures per location， 3 for headings and 2 for pitches。
This part is commented on in the function img_to_db. And when using this part, please comment on the part for downloading only one picture per location.
After trying with a small dataset, we picked the best heading and pitch, then use this parameter to download to the whole dataset.

- Examples of how to use 
Examples for using this is in the Jupyter Notebooks downloading picture.

- Can't run it directly!!!!!
when using this downloading function, we need two things. The first is the key to google street API. And the second is the shapefile which contains the location of each building.
In this project, our shapefile is provided by the lab, and it is a secret document that is owned by the Swiss government, so we can't upload it on GitHub. Sorry about that.
But in the Unlable folder, we provided a few pictures which are downloaded from google. And in this project, we downloaded around 6000 pictures.

## Filter the images(folder "Data Filtering")

In this part, we used three kinds of methods to detect buildings, green areas, and cars respectively.

#### place365 model
First of all, the code in the modeling folder is used. This model is a CNNs based pre-trained model, named the place365 model, which can detect where the is image taken, according to the contents in images.
The pre-trained model contains a text file, describing 365 places it can detect. The place contains indoors and outdoors place together. Therefore, we select several desired outside locations in a .csv file. Then we used this kind of model to detect the place of images, if the place is contained by the .cvs file, the image is regarded as desired outside building, store it. While if the building is not on the list, we store it in another file.   The example is in file demo and the source code has already been functionalized in model_test.py.
One more thing about model_test.py, the directory needs to be changed when the function is run.

#### green detection model
The second model we used is a pure computer version based model, which use the cv2 library as a toolbox to detect images.
As it is seen in images, there are many images contain a large part of plants like trees and grass. Apparently, these kinds of images are not suitable for further process, and they should be filtered.
In our code, we used the cv2 library to detect the ratio of green, which would mean plants like grass and plant most of the time. A threshold is set in the code. Every time we process one image, code would output the ratio of it. If it is larger than the threshold delete it, Otherwise keep it.
The kept ones and filtered ones will be stored in different directories.

#### SSD model
The model is an Nvidia object detection model. It uses the SSD model to detect objects in images. As we know, the SSD model can detect 80 classes, such as a car, human, and vase. Due to the location of our model, only a small part of these classes will exist in our image, of which car is the most frequently appear one. Therefore, we set the car as our detect object. In this model, a set of labels and probabilities are output, and we select the label car and its probability as judge threshold. Here, we set another threshold and filter the images according to it.
Attention! This model can only be run with the help of Cuda. Please feel free to change the target object. Many other objects might exist in the images as well, like the person, etc., in other scenarios. You can change it according to the class number, which can be seen in the variable class.

## pix2pix model(folder "pix2pix")

## working with pix2pix model
This part is using the pix2pix model to predict the images we get after the data filtering process. When applying the original trained model, the result of prediction is not good because the original datasets are from the city and there are not too many backgrounds. Also, there are 12 classes in the original model, and in this project, we only need four(background, facade, opening, cover). 
So we did several steps to make some improvements to this model. 
- First, we merge the 12 colors into 4 and tested both the original dataset and our new dataset. What we get at this point is the predicted data with only 4 colors, but still, the trees and roads(some interfere) are still recognized as facades. 
- So secondly, we manually revised our new dataset's result to let these non-facades part to be backgrounds and we revised 70 pictures.
- The last step, we put these 70 images along with the 400 original images into the training datasets. And retrain it for the second time. Then get our new results.
- Also, according to the new results' performance, we have revised the training datasets to get better results.

## code for model pix2pix
- the pix2pix folder contains a readme file that is written by the original author, and the specific information about this model is in that readme file.

- The jupyter notebook "facades_baseline" contains all steps to train and to test. Tips, it is better to train the model on Colab. 

- we revised the util.py and visualizer.py in folder "util" in order to merge the 12 colors into 4 colors and calculate the ratio between the facades and backgrounds and openings and facades. After merging the color, If you don't need to calculate the ratio(maybe you are adjusting the parameters), please switched back to the old version, because it is faster.

- In the dataset folder, there are three folders(train, test, val), we actually have 470 pictures in train dataset, 104 pictures in val and around 2100 pictures in test. For convenience, we only provide a few samples.

## Thanks
Upon the completion of our project, we would like to express our gratitude to the host lab IMAC and our mentors Mr. Alireza Khodaverdian and Mr. Saeed for their guide and help.