# Import necessary libraries
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from category_list import category_list as classnames
from helper import *

from torch.cuda.amp import autocast
import numpy as np


class SketchCLIP(object):
    model_backbone="ViT-B/16"
    n_ctx=5
    ctx_init = None
    input_shape=(224,224)
    precision="fp32"
    seed=1
    prompt_depth = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = SketchCLIP()



clip_model = load_clip_to_cpu(cfg)
if cfg.precision == "fp32" or cfg.precision == "amp":
    clip_model.float()

# Function to load the model with learned prompts
def load_model(cfg, classnames, clip_model, model_name, del_ = True):
    model = CustomCLIP_MAPLE_Adaptive(cfg, classnames, clip_model).to(device)
    model = nn.DataParallel(model)
    state_dict = torch.load(model_name, map_location=device)
    if "module.prompt_learner.token_prefix" in state_dict and del_:
        del state_dict["module.prompt_learner.token_prefix"]
    if "module.prompt_learner.token_suffix" in state_dict and del_:
        del state_dict["module.prompt_learner.token_suffix"]
    model.load_state_dict(state_dict, strict=False)
    return model


model_name = "SketchCLIP.pth"
model = load_model(cfg, classnames, clip_model, model_name)
model.eval()

# Define the prediction function
def save_image(image, path):
    image.save(path)

def predict(image):
    gradio_sample = image['composite']
    

    
    gradio_sample = np.array(gradio_sample)
    gradio_sample = np.repeat(gradio_sample[:, :, np.newaxis], 3, axis=2)
    gradio_sample[gradio_sample>10] = 255
    image = Image.fromarray(gradio_sample.astype('uint8'))

   
    # save_image(image, 'test.png') -> to debug

    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prec = cfg.precision
        if prec == "amp":
            with autocast():
                output = model(image_tensor, None, None, training=False, logits_only=True)
        else:
            output = model(image_tensor, None, None, training=False, logits_only=True)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    # Get the top 5 predictions
    values, indices = probabilities[0].topk(5)
    results = {classnames[indices[i]]: float(values[i]) for i in range(len(indices))}
    return results


iface = gr.Interface(
    fn=predict,
    inputs = gr.ImageEditor(type="pil", image_mode="L", crop_size=(512, 512)),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classification with Learned Prompts",
    description="Upload an image, and the model will classify it based on the provided categories using learned prompts."
)

# Launch the app
iface.launch()