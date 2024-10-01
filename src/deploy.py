
from PIL import Image
import os
import yaml
from PIL import Image
from model import BannerGenerator
# Load the YAML file

import gradio as gr
from gradio.themes import Soft
from PIL import Image
import os
import yaml
from model import BannerGenerator

# Load YAML config
with open(os.path.join(os.getcwd(), 'configs', 'model.yaml')) as file:
    config = yaml.safe_load(file)

# Gradio theme setup
theme = Soft(
    primary_hue="purple",
    secondary_hue="teal",
    text_size="lg",
    font=[gr.themes.GoogleFont('Montserrat'), gr.themes.GoogleFont('ui-sans-serif'), gr.themes.GoogleFont('system-ui'), 'sans-serif'],
)

# Function to load images
def load_image(selected_image):
    return Image.open(image_options[selected_image])

# Dropdown options
image_options = {
    "Upto 60 percent off": os.path.join(os.getcwd(), config['loader_params']['input_directory'],'text_60_percent_off.png'),
    "Buy2Get1 Free": os.path.join(os.getcwd(), config['loader_params']['input_directory'],'text_buy_2_get_1.png')
}

# Gradio interface
with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(os.path.join(os.getcwd(), config['loader_params']['input_directory'],'logo.png'), elem_id="logo", label="Logo", show_label=False, height=50, width=50)
        with gr.Column(scale=9):
            gr.Markdown("# Welcome to Artiflex - your AI assistant for promotional content creation")
    
    # Image upload and dropdown
    with gr.Row():
        image1_input = gr.Image(label="Upload product image", type="pil",  height=300, width=300)
        text_image_dropdown = gr.Dropdown(
            label="Select offer",
            choices=list(image_options.keys()),
            value=list(image_options.keys())[0]  # Default selection
        )
        
    # Prompt text and dropdown
    prompt_text = gr.Textbox(label="Optional Prompt ", placeholder="Enter a prompt (optional)")
    prompt_input = gr.Dropdown(
        label="Default Prompt",
        choices=config['loader_params']['default_prompts'],
        value=config['loader_params']['default_prompts'][0]
    )
    
    generate_button = gr.Button("Generate and Display Banner")
    output_image = gr.Image(label="Displayed Image")
    
    # When button is clicked
    obj = BannerGenerator(config)
    generate_button.click(
        fn=obj.generator,  # Pass the function without invoking it
        inputs=[image1_input, text_image_dropdown, prompt_text],
        outputs=output_image
    )

# Launch Gradio app
demo.launch(share=True)

