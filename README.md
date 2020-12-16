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

- Examples on how to use 
Examples for using this is in the Jupyter Notebooks downloading picture.

- Can't run it directly!!!!!
when using this downloading function, we need two things. The first is the key to google street API. And the second is the shapefile which contains the location of each building.
In this project, our shapefile is provided by the lab, and it is a secret document that is owned by the Swiss government, so we can't upload it on GitHub. Sorry about that.
But in the Unlable folder, we provided a few pictures which are downloaded from google. And in this project, we downloaded around 6000 pictures.

## Filter the images(folder "Data Filtering")

## pix2pix model
This part is using the pix2pix model to predict the images we get after the data filtering process. When applying the original trained model, the result of prediction is not good because the original datasets are from the city and there are not too many backgrounds.
Also, there are 12 classes in the original model, and in this project, we only need four(background, facade, opening, cover). So want we did first merge the 12 colors into 4 and tested both the original dataset and our new dataset.
Then what we get at this point is the predicted data with only 4 colors, but still, the trees and roads(some interfere) are still recognized as facades. So then we manually revised our new dataset's result to let these non-facades part to be backgrounds and we revised 70 pictures.
After that, we put these 70 images along with the 400 original images into the training datasets. And retrain it for the second time. Then get our new results.

- the pix2pix folder contains a readme file that is written by the original author, and the specific information about this model is in that readme file.

- we revised the util.py and visualizer.py in folder "util" 
##Thanks
We really appreciate for the host lab IMAC and our mentor Mr.Alireza Khodaverdian and Mr.Saeed. They gave us a lot of help