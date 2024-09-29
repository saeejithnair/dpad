import replicate
import requests
import os
import uuid
from io import BytesIO
import base64
from PIL import Image
import gradio as gr

# Replace with your actual API keys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
PHOTOROOM_API_KEY = os.getenv("PHOTOROOM_API_KEY")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def generate_tile(prompt, prompt_strength, reference_image=None):
    input_data = {
        "prompt": prompt,
        "go_fast": True,
        "guidance": 3.5,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "output_quality": 80,
        "prompt_strength": prompt_strength,
        "num_inference_steps": 28
    }
    
    if reference_image is not None:
        # Convert PIL Image to base64
        buffered = BytesIO()
        reference_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        input_data["image"] = f"data:image/png;base64,{img_str}"

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input=input_data
    )
    
    # The output is a list with one URL
    image_url = output[0]
    
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    return img

def remove_background(input_image):
    url = "https://sdk.photoroom.com/v1/segment"
    
    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    input_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {"image_file": img_byte_arr}
    headers = {"x-api-key": PHOTOROOM_API_KEY}
    
    response = requests.post(url, files=files, headers=headers)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def process_image(prompt, prompt_strength, reference_image):
    # Generate the tile
    generated_image = generate_tile(prompt, prompt_strength, reference_image)
    
    # Remove the background
    bg_removed_image = remove_background(generated_image)
    
    return generated_image, bg_removed_image

# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=0, maximum=1, value=0.8, step=0.01, label="Prompt Strength"),
        gr.Image(label="Reference Image (Optional)", type="pil")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Image(label="Background Removed Image")
    ],
    title="Tile Generator with Background Removal",
    description="Generate a tile based on a prompt and optionally a reference image, then remove its background. Adjust prompt strength to control the influence of the prompt vs. the reference image."
)

# Launch the app
iface.launch()