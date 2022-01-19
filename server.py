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

import torch
from models import EfficientNet

import shutil
import uvicorn
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile

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

def sign_s3(file_name, file_type):
  S3_BUCKET = os.environ.get('S3_BUCKET')
  s3 = boto3.client('s3')

  presigned_post = s3.generate_presigned_post(
    Bucket = S3_BUCKET,
    Key = file_name,
    Fields = {"acl": "public-read", "Content-Type": file_type},
    Conditions = [
      {"acl": "public-read"},
      {"Content-Type": file_type}
    ],
    ExpiresIn = 3600
  )

  return json.dumps({
    'data': presigned_post,
    'url': 'https://%s.s3.amazonaws.com/%s' % (S3_BUCKET, file_name)
  })

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