import torch
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.transforms import Compose, Normalize

from torch.nn import ReLU
from torch.autograd import Variable
from torch.nn import AvgPool2d, Conv2d, Linear, ReLU, MaxPool2d, BatchNorm2d
import torch.nn.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
image_net_std = torch.Tensor([0.229, 0.224, 0.225])

import matplotlib.pyplot as plt

class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

image_net_preprocessing = Compose([
    Normalize(
        mean=image_net_mean,
        std=image_net_std
    )
])

image_net_postprocessing = Compose([
    NormalizeInverse(
        mean=image_net_mean,
        std=image_net_std)
])

def tensor2img(tensor, ax=plt):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img

def tensor2cam(image, cam):
    image_with_heatmap = image2cam(image.squeeze().permute(1,2,0).cpu().numpy(),
              cam.detach().cpu().numpy())

    return torch.from_numpy(image_with_heatmap).permute(2,0,1)

def image2cam(image, cam):
    h, w, c = image.shape
    cam -= np.min(cam)
    cam /= np.max(cam)  # Normalize between 0-1
    cam = cv2.resize(cam, (w,h))

    cam = np.uint8(cam * 255.0)
    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    img_with_cam = img_with_cam + (image * 255)
    img_with_cam /= np.max(img_with_cam)

    return img_with_cam


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    credits to https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def imshow(tensor):
    tensor = tensor.squeeze()
    if len(tensor.shape) > 2: tensor = tensor.permute(1, 2, 0)
    img = tensor.cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()

def module2traced(module, inputs):
    handles, modules = [], []

    def trace(module, inputs, outputs):
        modules.append(module)

    def traverse(module):
        for m in module.children():
            traverse(m)
        is_leaf = len(list(module.children())) == 0
        if is_leaf: handles.append(module.register_forward_hook(trace))

    traverse(module)

    _ = module(inputs)

    [h.remove() for h in handles]

    return modules

class Base:
    def __init__(self, module, device):
        self.module, self.device = module, device
        self.handles = []

    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, inputs, layer, *args, **kwargs):
        return inputs, {}

class GradCam(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handles = []
        self.gradients = None
        self.conv_outputs = None

    def store_outputs_and_grad(self, layer):
        def store_grads(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def store_outputs(module, input, outputs):
            if module == layer:
                self.conv_outputs = outputs

        self.handles.append(layer.register_forward_hook(store_outputs))
        self.handles.append(layer.register_backward_hook(store_grads))

    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_out[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))


    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, input_image, layer, guide=False, target_class=None, postprocessing=lambda x: x, regression=False):
        self.clean()
        self.module.zero_grad()

        if layer is None:
            modules = module2traced(self.module, input_image)
            for i, module in enumerate(modules):
                if isinstance(module, Conv2d):
                    layer = module

        self.store_outputs_and_grad(layer)

        if guide: self.guide(self.module)

        input_var = Variable(input_image, requires_grad=True).to(self.device)
        predictions = self.module(input_var)

        if target_class is None: values, target_class = torch.max(predictions, dim=1)
        if regression: predictions.backward(gradient=target_class, retain_graph=True)
        else:
            target = torch.zeros(predictions.size()).to(self.device)
            target[0][target_class] = 1
            predictions.backward(gradient=target, retain_graph=True)

        with torch.no_grad():
            avg_channel_grad = F.adaptive_avg_pool2d(self.gradients.data, 1)
            self.cam = F.relu(torch.sum(self.conv_outputs[0] * avg_channel_grad[0], dim=0))

            image_with_heatmap = tensor2cam(postprocessing(input_image.squeeze().cpu()), self.cam)

        self.clean()

        return image_with_heatmap.unsqueeze(0), { 'prediction': target_class}

def convert(img):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    # plt.imshow(grayScale)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # plt.imshow(blackhat)

    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    # plt.imshow(thresh2)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA)
    # plt.imshow(dst)
    return dst