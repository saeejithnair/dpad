import numpy as np
from PIL import Image

def apply_mask(original_image_path, mask_image_path, output_image_path):
    # Open the original image and the mask image
    original = Image.open(original_image_path).convert("RGBA")
    mask = Image.open(mask_image_path).convert("RGBA")

    # Ensure the mask is the same size as the original image
    if original.size != mask.size:
        print("resizing")
        original = original.resize(mask.size, Image.LANCZOS)

    # Convert images to NumPy arrays
    original_array = np.array(original)
    mask_array = np.array(mask)

    # Create a boolean mask where True represents pixels to keep (alpha == 255)
    keep_mask = mask_array[:,:,3] == 255

    # Create the output array, starting with a copy of the original
    output_array = original_array.copy()

    # Set alpha channel to 0 for pixels we don't want to keep
    output_array[~keep_mask, 3] = 0

    # Create a new image from the output array
    output = Image.fromarray(output_array)

    # Save the result
    output.save(output_image_path)
    print(f"Saving to {output_image_path}")

# Usage remains the same
original_image_path = "/Users/snair/Documents/new_builds/original.png"
mask_image_path = "/Users/snair/Documents/new_builds/mask.png"
output_image_path = "/Users/snair/Documents/new_builds/output.png"

apply_mask(original_image_path, mask_image_path, output_image_path)
