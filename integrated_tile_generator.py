import replicate
import requests
import os
import uuid
from io import BytesIO
import base64
from PIL import Image
import gradio as gr
import cohere
import json
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
PHOTOROOM_API_KEY = os.getenv("PHOTOROOM_API_KEY")
COHERE_API_TOKEN = os.getenv("COHERE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

co = cohere.ClientV2(COHERE_API_TOKEN)

prefix = "isometric pixel art map tile. the tile base is a block of material, tile content is rendered on top of the block. tile content is:"
suffix = "white background."

# File to store prompts
PROMPTS_FILE = "prompts.json"

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_prompts(prompts):
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f)

prompts = load_prompts()

def add_prompt(prompt, keywords, used_for_generation=False):
    for existing_prompt in prompts:
        if existing_prompt["prompt"] == prompt:
            if used_for_generation:
                existing_prompt["timestamp"] = datetime.now().isoformat()
                existing_prompt["used_for_generation"] = True
            save_prompts(prompts)
            return f"Updated existing prompt: {prompt}"

    prompts.append({
        "prompt": prompt,
        "keywords": keywords,
        "timestamp": datetime.now().isoformat(),
        "used_for_generation": used_for_generation
    })
    save_prompts(prompts)
    return f"Added new prompt: {prompt}"

def search_prompts(query):
    documents = [{"text": p["prompt"]} for p in prompts]
    unique_prompts = {doc["text"] for doc in documents}  # Use a set to filter out duplicates
    documents = [{"text": prompt} for prompt in unique_prompts]  # Convert back to list of dicts
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=3,
    )
    
    results = []
    idx = 0
    for r in response.results:
        results.append([
            idx,
            r.relevance_score,
            documents[r.index]["text"]
        ])
        idx+=1 
    
    return results

def generate_prompt(keywords):
    response = co.chat(
        model="command-r-plus",
        messages=[
            {
                "role": "user",
                "content": f"""Generate a prompt for an isometric 2D pixel art map tile based on these keywords: {keywords}.
                The prompt will be used with the following prefix and suffix:
                Prefix: "{prefix}"
                Suffix: "{suffix}"
                The prompt should be detailed and creative, suitable for generating a tile that extends towards the edge of the image. Do not include any background description or repeat information from the prefix/suffix.
                Output the complete prompt starting with the prefix and ending with the suffix."""
            }
        ],
        temperature=0.8,
    )
    return response.message.content[0].text.replace('"', '').strip()

def generate_tile(prompt, prompt_strength, reference_image=None):
    input_data = {
        "image": "https://replicate.delivery/yhqm/xmiV1QzaHvYYNRWfS6bSTjy6BoK7AHr1hxWv4fqbWxjqBzhTA/out-0.webp",
        "prompt": f"{prefix} {prompt} {suffix}",
        "go_fast": True,
        "guidance": 3.5,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "output_quality": 90,
        "seed": 7,
        "prompt_strength": prompt_strength,
        "num_inference_steps": 28,
    }
    
    if reference_image is not None:
        buffered = BytesIO()
        reference_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        input_data["image"] = f"data:image/png;base64,{img_str}"

    output = replicate.run(
        "black-forest-labs/flux-dev",
        input=input_data
    )
    
    image_url = output[0]
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    return img


def remove_background(input_image):
    # Convert PIL Image to base64 string
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Run the rembg model on Replicate
    output = replicate.run(
        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
        input={"image": f"data:image/png;base64,{img_str}"}
    )

    # Download the result image
    response = requests.get(output)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def process_image(prompt, prompt_strength, reference_image):
    generated_image = generate_tile(prompt, prompt_strength, reference_image)
    bg_removed_image = remove_background(generated_image)
    
    add_prompt(prompt, "", used_for_generation=True)
    
    return generated_image, bg_removed_image

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Mind Mapper")
    
    with gr.Tab("Generate Tile"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt")
                prompt_strength = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="Prompt Strength")
                reference_image = gr.Image(label="Reference Image (Optional)", type="pil")
                generate_button = gr.Button("Generate Tile")
            with gr.Column():
                generated_image = gr.Image(label="Generated Image")
                bg_removed_image = gr.Image(label="Background Removed Image")
        
        generate_button.click(process_image, inputs=[prompt_input, prompt_strength, reference_image], outputs=[generated_image, bg_removed_image])
    
    with gr.Tab("Manage Prompts"):
        with gr.Row():
            with gr.Column():
                add_prompt_input = gr.Textbox(label="Enter Prompt")
                keywords_input = gr.Textbox(label="Enter Keywords (comma-separated)")
                add_button = gr.Button("Add Prompt")
                add_output = gr.Textbox(label="Result")
            with gr.Column():
                search_input = gr.Textbox(label="Enter Search Query")
                search_button = gr.Button("Search Prompts")
                search_output = gr.Dataframe(
                    headers=["Idx", "Score", "Prompt"],
                    datatype=["number", "number", "str"],
                    label="Search Results"
                )
                selected_prompt = gr.Dropdown(label="Select a prompt", choices=[], type="index")
                use_search_prompt = gr.Button("Use Selected Prompt")
        
        def update_search_results(results):
            if len(results) == 0:
                return [], gr.Dropdown(choices=[], value=None)
            
            choices = [f"{idx}: {results['Prompt'].iloc[idx]}" for idx, result in enumerate(results)]
            return results, gr.Dropdown(choices=choices, value=choices[0] if choices else None, type="index")

        add_button.click(add_prompt, inputs=[add_prompt_input, keywords_input], outputs=add_output)
        search_button.click(
            search_prompts,
            inputs=search_input,
            outputs=search_output
        ).then(
            update_search_results,
            inputs=search_output,
            outputs=[search_output, selected_prompt]
        )
        use_search_prompt.click(
            lambda x, df: df.iloc[x, 2] if x is not None and not df.empty else "",
            inputs=[selected_prompt, search_output],
            outputs=prompt_input
        )
    
    with gr.Tab("Generate Prompt"):
        generate_input = gr.Textbox(label="Enter Keywords for Prompt Generation")
        generate_button = gr.Button("Generate Prompt")
        generate_output = gr.Textbox(label="Generated Prompt")
        use_generated_prompt = gr.Button("Use Generated Prompt")
        
        generate_button.click(generate_prompt, inputs=generate_input, outputs=generate_output)
        use_generated_prompt.click(lambda x: x, inputs=generate_output, outputs=prompt_input)

demo.launch()