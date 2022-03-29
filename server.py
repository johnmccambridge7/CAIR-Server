import io
import os
import time
import boto3
import json
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import albumentations
from torchvision.utils import save_image
from gradCAM import *
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import requests
from io import BytesIO
from skimage.io import imread
from google.cloud import firestore
from google.cloud.firestore_v1 import ArrayRemove, ArrayUnion

import torch
from models import EfficientNet

import shutil
import uvicorn
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('./service.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://cair-d2e7d-default-rtdb.firebaseio.com'
})

firestoreDB = firestore.Client()

Checkpoint_DIR = '.'
transform = transforms.ToPILImage()

class Location(BaseModel):
    lat: str
    lng: str
    zm: str

def load_checkpoint(checkpoint, cuda):
    from imageModels.brnet import BRNet, BRNetv2
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, checkpoint)
                          ), "{} does not exists.".format(checkpoint)
    method = checkpoint.split('_')[0]
    model = method.split('-')[0]
    is_multi = False
    if "MCFCN" in model or "BRNet" in model:
        is_multi = True
    src_ch, tar_ch, base_kernel = [int(x) for x in method.split('-')[1].split("*")]
    net = eval(model)(src_ch, tar_ch, base_kernel)
    net.load_state_dict(
        torch.load(os.path.join(Checkpoint_DIR, checkpoint), map_location=torch.device('cpu')))

    print("Loaded checkpoint: {}".format(checkpoint))
    return net.eval(), is_multi

housingModel, is_multi = load_checkpoint('BRNet-3*1*24-NZ32km2_iter_5000.pth', False)

KERNEL = '9c_b4ns_768_768_ext_15ep'
OUTPUT = 9
DATA = './data'
FOLDER = 512
META = False
SIZE = 512
DEBUG = True
BATCH = 10
TESTING_NUMBER = 10

device = 'cpu'

app = FastAPI()

model = EfficientNet(
        'efficientnet_b4',
        OUTPUT,
        nMeta=0,
        metaDim=[int(nd) for nd in [512,128]],
        pretrained=True
    )

model = model.to(device)
model_file = 'models/9c_b4ns_768_640_ext_15ep_best_fold3.pth'

try:  # single GPU model_file
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=True)
except:  # multi GPU model_file
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)

model.eval()

visualizer = GradCam(model, device)

def transform_image(bytes):
    tfs = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])
        
    image = Image.open(io.BytesIO(bytes))
    return tfs(image).unsqueeze(0)

# upload and store the image and metadata -> process on ensemble -> store results -> push to client


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    return s3_client.upload_file(file_name, bucket, object_name)

@app.post('/upload')
async def upload(image: UploadFile = File(...)):
    directory = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(f'{directory}/uploads'):
        os.makedirs(f'{directory}/uploads')
        print('Making directory!', directory)

    filename = f'{directory}/uploads/{time.time()}-{image.filename}'
    content = await image.read()
    transformed = transform_image(content)
    save_image(transformed[0], filename)

    s3 = boto3.client('s3')

    with open(filename, "rb") as f:
        s3.upload_fileobj(f, "cair-bucket", filename)

    return { 'success': filename }

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    # upload and store the image from device
    directory = os.path.dirname(os.path.realpath(__file__))
    filename = f'{directory}/uploads/{time.time()}-{image.filename}'
    f = open(f'{filename}', 'wb')
    content = await image.read()
    # perform inference on an ensemble on models
    transformed = transform_image(content)

    save_image(transformed[0], filename)

    # convertedImage = Image.fromarray()
    # convertedImage.save(filename)

    logits = model(transformed).softmax(1)
    print(logits)

    converted = convert(np.array(Image.open(filename)))

    composedInput = Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        image_net_preprocessing
    ])(converted).unsqueeze(0)

    gradients = tensor2img(visualizer(composedInput, None, postprocessing=image_net_postprocessing)[0])
    plt.imsave(f'{filename}', gradients)

    gradientImage = cv2.imread(f'{filename}')
    res, pngImage = cv2.imencode(".png", gradientImage)

    return StreamingResponse(io.BytesIO(pngImage.tobytes()), media_type="image/png") # {"filename": image.filename}

@app.post('/location')
async def location(data: Location):
    lat, lng, zm = data.lat, data.lng, data.zm
    # print(lat, lng, zm)
    url = f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lng},{lat},{zm},0/1200x1200?access_token=pk.eyJ1Ijoiam9obm1jY2FtYnJpZGdlIiwiYSI6ImNrejh5MXh4djFwNjEydm16ZHVxMWRhMnAifQ.XuCE1B8RucG4DWsa1iIczQ'
    response = requests.get(url)
    mapImage = Image.open(BytesIO(response.content))
    x = np.array(mapImage)

    width, height = 1200, 1200
    new_width, new_height = 50, 50

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    x = (x / 255).transpose((2, 0, 1))

    img = torch.from_numpy(x).float()
    gen_y = housingModel(img.unsqueeze(0))

    locationSquare = transform(np.squeeze(gen_y[0]))
    locationSquare = locationSquare.crop((left, top, right, bottom))
    locationSquare = np.array(locationSquare)
    locationScore = np.round(np.sum(locationSquare) / (new_height * new_width), 2)

    db.reference().set({ "location": f"{locationScore}" })
    
    save_image(gen_y[0], "output.png")
    mapImage.save("real.png")

    return "John"

# uvicorn server:app --reload --workers 1 --host 0.0.0.0 --port 8008