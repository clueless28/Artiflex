# Artiflex: AI-Powered Promotional Banner Generator

Welcome to **Artiflex**, an AI assistant designed to help you create promotional banners and content effortlessly. Using AI-driven image processing and text inpainting, this tool allows users to upload product images, select from predefined offers, and optionally enter prompts to generate promotional content.

### Current Solution
In the current workflow (MVP at the time of submission), the system takes in three inputs:  
1. **Prompt (text)**  
2. **Product image**     
3. **Offer selection (preset list)**  

The application extracts a mask from the product image and applies a preset offer font image according to the selection. These elements are placed within a preset layout. This layout is then provided as input to a **stable diffusion model**, which designs the banner according to user inputs such as theme, gradient, and overall look.

#### *Not in scope:*
1. Dynamic layout
2. Refined prompt
3. Conditioned generative AI
4. Multiple products per banner

![Orange and Beige Ganesh Chaturthi Sale Banner-2](https://github.com/user-attachments/assets/3c7a9caa-4694-4e3d-8bda-9047f60a209c)



## Features

- **AI-Powered Banner Generation**: Utilize Stable Diffusion's Inpainting model to create high-quality promotional banners.
- **Custom Image Upload**: Users can upload their product images.
- **Predefined Offers**: Choose from predefined text-based promotions (e.g., "Buy2Get1 Free", "Up to 60% Off").
- **Prompt Input**: Optionally enter custom prompts to generate personalized banners.
- **Responsive UI**: Built using Gradio with a sleek and intuitive user interface.

## Table of Contents
- [Architecture](#Architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Customization](#customization)
- [Future Work](#future-work)

## Installation

### 1. Clone the Repository
```
git clone https://github.com/yourusername/artiflex.git
cd artiflex

```

### 2. Set Up a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Usage
To run the application, execute the following command in your terminal:
```
python src/deploy.py 
nohup python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 src/llm.py > output.log 2>&1 &
``

## Sample outputs:
<img width="738" alt="Screenshot 2024-09-30 at 12 22 23 PM" src="https://github.com/user-attachments/assets/7f043fc4-2c86-43da-8ac3-5246d5350591">






