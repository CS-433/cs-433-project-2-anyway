## Description
This folder is used for grabbing data from Google Street-View API. In the make_dataset.py, we allow it to download 6 pictures per location， 3 for headings and 2 for pitches。
This part is commented on in the function img_to_db. And when using this part, please comment on the part for downloading only one picture per location.
After small trying with a small dataset, we picked the best heading and pitch, then use this parameter to download to the whole dataset.

## Examples on how to use 
Examples for using this is in the Jupyter Notebooks.

## Can't run it directly!!!!!
when using this downloading function, we need two things. The first is the key to google street API. And the second is the shapefile which contains the location of each building.
In this project, our shapefile is provided by the lab, and it is a secret document that is owned by the Swiss government, so we can't upload it on GitHub. Sorry about that.
But in the Unlable folder, we provided a few pictures which are downloaded from google. And in this project, we downloaded around 6000 pictures.


