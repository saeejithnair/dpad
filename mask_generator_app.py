import gradio as gr
import numpy as np
from PIL import Image
import os

def crop_to_square(image):
    """Crop the input image to a square based on the shorter side."""
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))

def update_editor(image_path):
    """
    Update the image editor with the input image.
    
    Args:
        image_path (str): Path to the input image.
        
    Returns:
        np.ndarray: Image data for the editor.
        tuple: Original size of the image.
    """
    if image_path is None:
        return None, None
    # Convert image to RGBA to ensure transparency is preserved
    img = Image.open(image_path).convert('RGBA')
    editor_img = img.resize((512, 512), Image.LANCZOS)
    return np.array(editor_img), img.size

def generate_mask(editor_output, original_size):
    """
    Generate a binary mask from the editor output.
    
    Args:
        editor_output (dict): Output from the Gradio ImageEditor.
        original_size (tuple): Original size of the input image.
        
    Returns:
        Image.Image: Binary mask image.
    """
    if editor_output is None or not isinstance(editor_output, dict):
        print("Invalid editor output")
        return None

    if 'layers' not in editor_output or not editor_output['layers']:
        print("No layers found in editor output")
        return None

    # The mask is in the first (and usually only) layer
    mask = editor_output['layers'][0]

    # Convert mask to binary (0 or 255) based on the alpha channel
    binary_mask = np.where(mask[:, :, 3] > 0, 255, 0).astype(np.uint8)

    # Resize the mask to the original image size using NEAREST to preserve binary values
    mask_image = Image.fromarray(binary_mask, mode='L').resize(original_size, Image.NEAREST)

    return mask_image

def process_and_prepare_merge(image_path, editor_output, original_size):
    """
    Apply the mask to the input image to create a processed image with transparent areas.
    
    Args:
        image_path (str): Path to the input image.
        editor_output (dict): Output from the Gradio ImageEditor.
        original_size (tuple): Original size of the input image.
        
    Returns:
        str: Status message.
        Image.Image: Generated mask image.
        Image.Image: Processed image with transparency.
    """
    if image_path is None or editor_output is None:
        return "No image or mask to process.", None, None

    # Open the original image and convert to RGBA
    original_image = Image.open(image_path).convert('RGBA')
    original_image = crop_to_square(original_image)

    # Generate the mask
    mask_image = generate_mask(editor_output, original_image.size)

    if mask_image is None:
        return "Failed to generate mask.", None, None

    # Apply the mask to the alpha channel
    processed_image = original_image.copy()
    processed_image.putalpha(mask_image)

    # Ensure the output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save the processed image for debugging
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)
    processed_path = os.path.join("outputs", f"{name}-processed.png")
    processed_image.save(processed_path)
    print(f"Processed image saved as {processed_path}")

    return f"Processed image saved as {processed_path}", mask_image, processed_image

def interactive_merge(processed_image, background_image, x_position, y_position, scale):
    """
    Merge the processed image with the background image based on user inputs.
    
    Args:
        processed_image (Image.Image): The processed image with transparency.
        background_image (str or Image.Image): Path or Image object for the background.
        x_position (float): X-axis position to paste the processed image.
        y_position (float): Y-axis position to paste the processed image.
        scale (float): Scaling factor for the processed image.
        
    Returns:
        Image.Image or str: The merged image or an error message.
    """
    if processed_image is None or background_image is None:
        return "Processed image or background image is missing."

    try:
        # Convert images to PIL if they're not already
        if isinstance(processed_image, np.ndarray):
            processed = Image.fromarray(processed_image).convert('RGBA')
        else:
            processed = processed_image.convert('RGBA')

        print(f"Processed image mode: {processed.mode}")
        print(f"Processed image size: {processed.size}")

        if isinstance(background_image, str):
            background = Image.open(background_image).convert('RGBA')
        else:
            background = background_image.convert('RGBA')

        print(f"Background image mode: {background.mode}")
        print(f"Background image size: {background.size}")

        # Scale the processed image
        new_size = (int(processed.width * scale), int(processed.height * scale))
        processed = processed.resize(new_size, Image.LANCZOS)

        print(f"Scaled processed image size: {processed.size}")

        # Debug: Save the scaled processed image
        scaled_processed_path = os.path.join("outputs", "scaled_processed.png")
        processed.save(scaled_processed_path)
        print(f"Scaled processed image saved as {scaled_processed_path}")

        # Calculate the position to paste
        paste_x = int(x_position)
        paste_y = int(y_position)

        # Ensure paste positions are within the background bounds
        paste_x = max(0, min(paste_x, background.width - processed.width))
        paste_y = max(0, min(paste_y, background.height - processed.height))

        print(f"Pasting at position: ({paste_x}, {paste_y})")

        # Create a copy of the background to paste onto
        merged = background.copy()

        # Paste the processed image onto the background using its alpha channel as mask
        merged.paste(processed, (paste_x, paste_y), processed)

        # Debug: Save the merged image
        merged_path = os.path.join("outputs", "merged.png")
        merged.save(merged_path)
        print(f"Merged image saved as {merged_path}")

        return merged
    except Exception as e:
        print(f"Error during merging: {e}")
        return f"An error occurred during merging: {e}"

# Define the Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Image Mask Generator and Merger")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image (PNG Recommended)", type="filepath")
            background_image = gr.Image(label="Background Image (PNG Recommended)", type="filepath")
            generate_btn = gr.Button("Generate Mask and Process Image")

        with gr.Column(scale=2):
            editor = gr.ImageEditor(
                label="Draw Mask",
                brush=gr.Brush(colors=["#ffffff"], default_size=70),
                type="numpy"
            )

    with gr.Row():
        mask_output = gr.Image(label="Generated Mask", type="pil")
        processed_output = gr.Image(label="Processed Image", type="pil")

    gr.Markdown("## Merge Images")
    gr.Markdown("Adjust the sliders to position and scale the processed image:")

    with gr.Row():
        x_position = gr.Slider(minimum=0, maximum=1000, step=1, label="X Position", value=0)
        y_position = gr.Slider(minimum=0, maximum=1000, step=1, label="Y Position", value=0)
        scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Scale", value=1.0)

    merge_btn = gr.Button("Merge Images")
    merged_output = gr.Image(label="Merged Image", type="pil")

    save_output = gr.Textbox(label="Save Status")

    original_size = gr.State()

    # Define the interactions
    input_image.change(
        update_editor,
        inputs=input_image,
        outputs=[editor, original_size]
    )

    generate_btn.click(
        process_and_prepare_merge,
        inputs=[input_image, editor, original_size],
        outputs=[save_output, mask_output, processed_output]
    )

    merge_btn.click(
        interactive_merge,
        inputs=[processed_output, background_image, x_position, y_position, scale],
        outputs=merged_output
    )

# Launch the app
app.launch()