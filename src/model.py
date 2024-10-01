from diffusers import StableDiffusionInpaintPipeline
from imantics import Mask, Polygons
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch
from PIL import Image

class BannerGenerator:
    def __init__(self, config):
        self.config = config
        self.SEED = 2023
        self.NEGATIVE_PROMPT =self.config['default_params']['NEGATIVE_PROMPT']
        self.DEVICE = torch.device(0)
        self.HEIGHT,self. WIDTH = self.config['default_params']['HEIGHT'], self.config['default_params']['WIDTH']
        self.NUM_IMAGES_PER_PROMPT = self.config['default_params']['NUM_IMAGES_PER_PROMPT']
        self.NUM_INFERENCE_STEPS = self.config['default_params']['NUM_INFERENCE_STEPS']
        self.GUIDANCE_SCALE = self.config['default_params']['GUIDANCE_SCALE']
        self.MARGIN_BORDER = self.config['default_params']['MARGIN_BORDER']
        self.TRANSFORM = A.ReplayCompose([
            A.ShiftScaleRotate(shift_limit=0.3,
                            scale_limit=[-0.6, -0.5],
                            rotate_limit=10,
                            p=1.0,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=[255, 255, 255],
                            mask_value=[255, 255, 255])
        ])
        
        self.image_options = {
                "Upto 60 percent off": os.path.join(os.getcwd(), config['loader_params']['input_directory'],'text_60_percent_off.png'),
                "Buy2Get1 Free": os.path.join(os.getcwd(), config['loader_params']['input_directory'],'text_buy_2_get_1.png') 
            }
        
    def model_loader(self, prompt):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting").to(self.DEVICE)
        
        # Set generator seed for reproducibility
        generator = torch.Generator(device=self.DEVICE)
        generator = generator.manual_seed(self.SEED)
        # Load the augmented image 
        image = self.image1 
        image = np.array(image)
        # Create the mask based on the image's alpha channel
        if image.shape[2] == 4:  # If there are 4 channels (RGBA)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[image[:, :, 3] != 0] = 255  # Use the alpha channel for the mask
        else:  # If there are only 3 channels (RGB)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask.fill(255)  # Mark the entire image for inpainting, adjust this as needed
            
        # Prepare the input image for inpainting
        input_image = self.image1.resize((self.HEIGHT,self. WIDTH)) 
        mask_image = Image.fromarray(mask).resize((self.HEIGHT,self. WIDTH))
        transformed = self.TRANSFORM(image=np.array(input_image), mask=np.array(mask_image))

        # Invert mask
        mask_invert = cv2.bitwise_not(transformed["mask"])

        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        mask_dilate = cv2.dilate(transformed["mask"], kernel, iterations=2)

        # Erode and invert mask
        mask_erode = cv2.erode(transformed["mask"], kernel, iterations=25)
        mask_invert_erode = cv2.bitwise_not(mask_erode)

        # Create minimum bounding box around the mask
        contours, hierarchy = cv2.findContours(mask_invert_erode, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

        # Prepare the mask for inpainting
        mask_invert_erode = np.zeros(
            (mask_invert_erode.shape[0], mask_invert_erode.shape[1]), dtype=np.uint8)

        if contours:
            xmin, ymin, width, height = cv2.boundingRect(contours[0])
            cv2.rectangle(mask_invert_erode, (xmin - self.MARGIN_BORDER, ymin - self.MARGIN_BORDER),
                        (xmin + width + self.MARGIN_BORDER, ymin + height + self.MARGIN_BORDER),
                        color=(255, 255, 255),
                        thickness=-1)

        # Convert mask to PIL image
        mask_dilate = Image.fromarray(mask_dilate)
        # Perform inpainting using Stable Diffusion

        out_images = pipe(prompt= prompt,
                        negative_prompt=self.NEGATIVE_PROMPT,
                        image=Image.fromarray(transformed["image"][:,:,:3]),
                        mask_image=mask_dilate,
                        height=self.HEIGHT,
                        width=self.WIDTH,
                        num_images_per_prompt=self.NUM_IMAGES_PER_PROMPT,
                        num_inference_steps=self.NUM_INFERENCE_STEPS,
                        guidance_scale=self.GUIDANCE_SCALE,
                        generator=generator).images

        return out_images
               
    def generator(self,  image1, image2, prompt):
        self.image1 = image1
        self.image2 = image2
        self.prompt = prompt
        SEED = self.config['default_params']['SEED']
        random.seed(SEED)
        
        # Generic list of keywords to guide generation away from
        NEGATIVE_PROMPT = ("")
        if len(self.prompt)==0:
            PROMPT = "Create a background with high quality single christmas original photo. It should be full HD, clean minimum design. No text should be put."
        else :
            PROMPT = self.prompt

        padding = 300
        base_image = Image.open(os.path.join(os.getcwd(), self.config['loader_params']['input_directory'],'bg.png')).convert("RGBA").resize((self.config['default_params']['BANNER_W'], self.config['default_params']['BANNER_H']))
    
        object_image = self.image1.convert("RGBA").resize((self.image1.size[0] // 2, self.image1.size[1] // 2))
        text_image_path = self.image_options[self.image2]  # Get the path from the options
        text_image = Image.open(text_image_path).resize((self.config['default_params']['BANNER_W'], self.config['default_params']['BANNER_H']))

        # Get the dimensions of the base image
        base_width, base_height = base_image.size

        # Get the dimensions of the object to place
        object_width, object_height = object_image.size

        # Calculate position: Center-right
        x_position = (base_width - object_width - padding)  # Align to the right edge
        y_position = (base_height - object_height) // 2  # Vertically centered
        base_image.paste(object_image, (x_position, y_position), object_image)
        
        
        text_image = np.array(text_image)
        out_img = self.model_loader(PROMPT)[-1].resize((self.config['default_params']['BANNER_W'], self.config['default_params']['BANNER_H'])) 
        out_img = np.array(out_img)[:, :, :3]
        base_image = np.array(base_image)[:, :, :3]
        height, width = out_img.shape[:2]
        for k in range(3):  # Iterate over RGB channels
            for i in range(height):  # Loop over rows (height)
                for j in range(width):  # Loop over columns (width)
                    if base_image[i][j][k] == 0 and text_image[i][j][k] != 0:
                        pass
                    elif base_image[i][j][k] != 0 :
                        out_img[i][j][k] = base_image[i][j][k]
                    elif text_image[i][j][k] == 0:
                        # print(out_img[i][j][k])
                        if out_img[i][j][k] <= 200:
                            out_img[i][j][k] = 255
                        else:
                            out_img[i][j][k] = 128
                            
        # Convert back to PIL Image
        out_img_pil = Image.fromarray(out_img)
        return out_img_pil
