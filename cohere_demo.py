import cohere
from dotenv import load_dotenv
import os
import gradio as gr
import json

# Load environment variables from .env file
load_dotenv()

COHERE_API_TOKEN = os.getenv("COHERE_API_TOKEN")

co = cohere.ClientV2(COHERE_API_TOKEN)

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

def add_prompt(prompt, keywords):
    prompts.append({"prompt": prompt, "keywords": keywords})
    save_prompts(prompts)
    return f"Added prompt: {prompt}"

def search_prompts(query):
    documents = [{"text": p["prompt"]} for p in prompts]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=3,
    )
    return "\n\n".join([f"Score: {r.relevance_score:.2f}\nPrompt: {r.document['text']}" for r in response.results])

def generate_prompt(keywords):
    response = co.chat(
        model="command-r-plus",
        message=f"Generate a Midjourney prompt based on these keywords: {keywords}. The prompt should be detailed and creative, suitable for generating an image tile.",
        temperature=0.8,
    )
    return response.text

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Tile Prompt Manager")
    
    with gr.Tab("Add Prompt"):
        prompt_input = gr.Textbox(label="Enter Midjourney Prompt")
        keywords_input = gr.Textbox(label="Enter Keywords (comma-separated)")
        add_button = gr.Button("Add Prompt")
        add_output = gr.Textbox(label="Result")
        add_button.click(add_prompt, inputs=[prompt_input, keywords_input], outputs=add_output)
    
    with gr.Tab("Search Prompts"):
        search_input = gr.Textbox(label="Enter Search Query")
        search_button = gr.Button("Search")
        search_output = gr.Textbox(label="Search Results")
        search_button.click(search_prompts, inputs=search_input, outputs=search_output)
    
    with gr.Tab("Generate Prompt"):
        generate_input = gr.Textbox(label="Enter Keywords for Prompt Generation")
        generate_button = gr.Button("Generate Prompt")
        generate_output = gr.Textbox(label="Generated Prompt")
        generate_button.click(generate_prompt, inputs=generate_input, outputs=generate_output)

demo.launch()
