import gradio as gr
from PIL import Image
import json
import random
import time
import urllib
from zipfile import ZipFile

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from imantics import Mask, Polygons
from PIL import Image

# Predefined image paths (replace these with your actual image paths)
image_options = {
    "Upto 60 percent off": "/home/drovco/Bhumika/Artiflex/assets/text_60_percent_off.png",
    "Buy2Get1 Free": "/home/drovco/Bhumika/Artiflex/assets/text_buy_2_get_1.png"
}

# Function to load selected image
def load_image(selected_image):
    return Image.open(image_options[selected_image])





default_prompts = [
    "Create a background with a high-quality single Christmas original photo.",
    "Generate an abstract artwork with vibrant colors.",
    "Design a serene landscape with mountains and a river."
]

# Function to handle the input
def process_images(image1, image2, prompt):
    
    # Set seed for reproducibility
    SEED = 2023
    random.seed(SEED)
    DEVICE = torch.device(0)
 

    # Change this to vary the input size to the model
    HEIGHT, WIDTH = 512, 512

    # Change this to generate multiple images each run at the cost of increased GPU memory usage.
    NUM_IMAGES_PER_PROMPT = 1

    # Change this to vary the level of detail in the output image.
    NUM_INFERENCE_STEPS = 50

    # Change this to vary how much the generation should follow the input prompt.
    GUIDANCE_SCALE = 7.5

    # Margin around the object mask to remove before inpainting.
    MARGIN_BORDER = 20

    # Generic list of keywords to guide generation away from
    NEGATIVE_PROMPT = ("")

    # PROMPT = "Create a background with high quality single christmas original photo. It should be full HD, clean minimum design. No text should be put."


    # Input prompt
    if len(prompt)==0:
        PROMPT = "Create a background with high quality single christmas original photo. It should be full HD, clean minimum design. No text should be put."
    else :
        PROMPT=prompt

    padding = 300
    # Define the transformation for image augmentation
    TRANSFORM = A.ReplayCompose([
        A.ShiftScaleRotate(shift_limit=0.3,
                        scale_limit=[-0.6, -0.5],
                        rotate_limit=10,
                        p=1.0,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                        mask_value=[255, 255, 255])
    ])

    # Load the base image and the object to be placed
    base_image = Image.open("/home/drovco/Bhumika/Artiflex/assets/bg.png").convert("RGBA").resize((1360, 800))
    #base_image =  image3.convert("RGBA").resize((1360, 800))x
    
    #object_image = image1.convert("RGBA").resize((200, 450))
    object_image = image1.convert("RGBA").resize((image1.size[0] // 2, image1.size[1] // 2))
    text_image_path = image_options[image2]  # Get the path from the options
    text_image = Image.open(text_image_path).resize((1360, 800)) 


    # Get the dimensions of the base image
    base_width, base_height = base_image.size

    # Get the dimensions of the object to place
    object_width, object_height = object_image.size

    # Calculate position: Center-right
    x_position = (base_width - object_width - padding)  # Align to the right edge
    y_position = (base_height - object_height) // 2  # Vertically centered

    # Paste the object onto the base image
    # Now using object_image as the mask
    base_image.paste(object_image, (x_position, y_position), object_image)

    # Save the result
    # base_image.save("/home/drovco/Bhumika/stable_diffusion_hf/output.png")

    # Optional: Show the final image
    base_image.show()

    # Load the inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting").to(DEVICE)

    # Set generator seed for reproducibility
    generator = torch.Generator(device=DEVICE)
    generator = generator.manual_seed(SEED)

    # Load the augmented image 
    image = image1 
    image = np.array(image)

    # Create the mask based on the image's alpha channel
    if image.shape[2] == 4:  # If there are 4 channels (RGBA)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[image[:, :, 3] != 0] = 255  # Use the alpha channel for the mask
    else:  # If there are only 3 channels (RGB)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask.fill(255)  # Mark the entire image for inpainting, adjust this as needed
    

    # Prepare the input image for inpainting
    input_image = image1.resize((HEIGHT, WIDTH)) 
    mask_image = Image.fromarray(mask).resize((HEIGHT, WIDTH))

    # Transform the images
    transformed = TRANSFORM(image=np.array(input_image), mask=np.array(mask_image))

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
        cv2.rectangle(mask_invert_erode, (xmin - MARGIN_BORDER, ymin - MARGIN_BORDER),
                    (xmin + width + MARGIN_BORDER, ymin + height + MARGIN_BORDER),
                    color=(255, 255, 255),
                    thickness=-1)

    # Convert mask to PIL image
    mask_dilate = Image.fromarray(mask_dilate)
    # Perform inpainting using Stable Diffusion

    out_images = pipe(prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=Image.fromarray(transformed["image"][:,:,:3]),
                    mask_image=mask_dilate,
                    height=HEIGHT,
                    width=WIDTH,
                    num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator).images

    # Display the output image
    plt.imshow(out_images[-1])
    plt.axis('off')
    plt.title('Inpainted Image')
    plt.show()


    # Optionally save the output image
    # out_images[-1].save("/home/drovco/Bhumika/stable_diffusion_hf/inpainted_output.png") 
   # text_pil_img=image2.resize((1360,800))
   # text_image = np.array(text_pil_img)
    text_image = np.array(text_image)
    out_img = out_images[-1].resize((1360,800)) 
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

    # You can modify this to return a generated image, combined image, etc.
    return out_img_pil

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="teal",
    text_size="lg",
    font=[gr.themes.GoogleFont('Montserrat'), gr.themes.GoogleFont('ui-sans-serif'), gr.themes.GoogleFont('system-ui'), 'sans-serif'],
)

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("/home/drovco/Bhumika/Artiflex/assets/logo_hackathon.png", elem_id="logo", label="Logo", show_label=False, height=50, width=50)
        with gr.Column(scale=9):
            gr.Markdown("# Welcome to Artiflex - your AI assistant for creativity")
    # gr.HTML(html_code) 
    with gr.Row():
        image1_input = gr.Image(label="Upload Object image", type="pil",  height=300, width=300)
        text_image_dropdown = gr.Dropdown(
            label="Select Text Image",
            choices=list(image_options.keys()),
            value=list(image_options.keys())[0]  # Default selection
        )
        
        
        
        
      #  image2_input = gr.Image(label="Upload text image", type="pil",  height=300, width=300)
       # image3_input = gr.Image(label="Upload background image", type="pil")
    
    prompt_text = gr.Textbox(label="Optional Prompt ", placeholder="Enter a prompt (optional)")
     # Create a dropdown for selecting prompt options
    prompt_input = gr.Dropdown(
        label="Default Prompt",
        choices=default_prompts,
        value=default_prompts[0]  # Set the default selected value
    )
    
    generate_button = gr.Button("Generate and Display")
    output_image = gr.Image(label="Displayed Image")
    
    # When the button is clicked, call process_images with inputs and display the result
    generate_button.click(
        fn=process_images,
        inputs=[image1_input, text_image_dropdown, prompt_text if prompt_text else prompt_input],
        outputs=output_image
    )

# Launch the Gradio app with a public link
demo.launch(share=True)


