import gradio as gr
import numpy as np
from PIL import Image
import os

def update_editor(image_path):
    if image_path is None:
        return None, None
    img = Image.open(image_path).convert('RGBA')
    # Resize image to fit in the editor (assuming 512x512 is a good size for editing)
    editor_img = img.resize((512, 512), Image.LANCZOS)
    return np.array(editor_img), img.size

def generate_mask(editor_output, original_size):
    if editor_output is None or not isinstance(editor_output, dict):
        print("Invalid editor output")
        return None
    
    if 'layers' not in editor_output or not editor_output['layers']:
        print("No layers found in editor output")
        return None
    
    # The mask is in the first (and usually only) layer
    mask = editor_output['layers'][0]
    
    # Convert mask to binary (0 or 255)
    # Use alpha channel as mask
    binary_mask = np.where(mask[:,:,3] > 0, 255, 0).astype(np.uint8)
    
    # Resize the mask to the original image size
    mask_image = Image.fromarray(binary_mask, mode='L').resize(original_size, Image.LANCZOS)
    
    return mask_image

def process_and_save(image_path, editor_output, original_size):
    if image_path is None or editor_output is None:
        return "No image or mask to process.", None, None
    
    # Open the original image and crop it to square
    original_image = Image.open(image_path).convert('RGBA')
    original_image = crop_to_square(original_image)
    
    # Generate the mask
    mask_image = generate_mask(editor_output, original_image.size)
    
    if mask_image is None:
        return "Failed to generate mask.", None, None
    
    # Create the processed image
    processed_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    processed_image.paste(original_image, (0, 0), mask_image)
    
    # Ensure the output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Generate output paths
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    mask_path = os.path.join("outputs", f"{name}-mask.png")
    processed_path = os.path.join("outputs", f"{name}-processed.png")
    
    # Save the mask and processed image
    mask_image.save(mask_path)
    processed_image.save(processed_path)
    
    return f"Mask saved as {mask_path}\nProcessed image saved as {processed_path}", mask_image, processed_image

def crop_to_square(image):
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))

with gr.Blocks() as app:
    gr.Markdown("# Image Mask Generator")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="filepath")
            generate_btn = gr.Button("Generate Mask and Process Image")
        
        with gr.Column(scale=2):
            editor = gr.ImageEditor(
                label="Draw Mask",
                brush=gr.Brush(colors=["#ffffff"], default_size=70),  # Set size to 100 (maximum)
                type="numpy"
            )
    
    with gr.Row():
        mask_output = gr.Image(label="Generated Mask", type="pil")
        processed_output = gr.Image(label="Processed Image", type="pil")
    
    save_output = gr.Textbox(label="Save Status")
    
    original_size = gr.State()
    
    input_image.change(
        update_editor,
        inputs=input_image,
        outputs=[editor, original_size]
    )
    
    generate_btn.click(
        process_and_save,
        inputs=[input_image, editor, original_size],
        outputs=[save_output, mask_output, processed_output]
    )

app.launch()