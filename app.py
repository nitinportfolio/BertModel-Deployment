from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from scripts import s3
from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os
import time
import torch
from transformers import pipeline
from PIL import Image
import base64
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoImageProcessor 

model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

app = FastAPI()

device = torch.device("cude") if torch.cuda.is_available() else torch.device("cpu")


############ Downloading ML Models ################

# Jus to make sure all files are downloaded from s3 server
force_download = False # True

model_name = "tinybert-sentiment-analysis/"
local_path = "ml-models/" + model_name
# it will check if model is already downloaded or not. if not then it will download from s3
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
sentiment_model = pipeline("text-classification", model=local_path, device=device)

model_name = "tinybert-disaster-tweet/"
local_path = "ml-models/" + model_name
# it will check if model is already downloaded or not. if not then it will download from s3
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)

twitter_model = pipeline("text-classification", model=local_path, device=device)


model_name = "vit-human-pose-classification/"
local_path = "ml-models/" + model_name
# it will check if model is already downloaded or not. if not then it will download from s3
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)

pose_model = pipeline("image-classification", model=local_path, device=device, image_processor=image_processor)



############ Download ML Models Ends ################



@app.get("/")
def hello():
    return ("Hello Nitin")

@app.post("/api/v1/sentiment_analysis")
# data validation task
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]
    output = NLPDataOutput(model_name="tinybert-sentiment-analysis",
                           text = data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    return output

@app.post("/api/v1/disaster_classifier")
# data validation task
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = twitter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]
    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text = data.text,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)
    return output

@app.post("/api/v1/pose_classifier")
# data validation task
def pose_classifier(data: ImageDataInput):
    start = time.time()
    output = pose_model(data.url)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x[0]["label"] for x in output]
    scores = [x[0]["score"] for x in output]
    output = ImageDataOutput(model_name="vit-human-pose-classification",
                           url = data.url,
                           labels=labels,
                           scores=scores,
                           prediction_time=prediction_time)

    return output


if __name__=="__main__":
    uvicorn.run(app="app:app", port = 8502, reload=True, host="0.0.0.0")
    #uvicorn.run(app="app:app", port = 8000, reload=True)
    # we can give custom host and port 

