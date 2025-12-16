#!/usr/bin/env python3
"""
Gradio interface for DeepSeek OCR
"""

import os
import sys
import base64
import requests
import gradio as gr
from PIL import Image
import io

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config_loader import SERVER_CONFIG, COMMON_CONFIG

# Server configuration
SERVER_URL = f"http://{SERVER_CONFIG.address}:{SERVER_CONFIG.port}/v1/ocr"

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def ocr_with_deepseek(image: Image.Image, prompt: str, level: str) -> tuple[str, str]:
    """
    Send image to DeepSeek OCR server and return the result

    Args:
        image: PIL Image object
        prompt: OCR prompt
        level: Processing level (raw, md_image, md_text, md_merged)

    Returns:
        tuple[str, str]: OCR result in raw text and markdown format
    """
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image)

        # Prepare request data
        data = {
            "image": image_base64,
            "level": level
        }

        # Add prompt if provided
        if prompt and prompt.strip():
            data["prompt"] = prompt

        # Send request to server
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status()

        # Parse response
        result = response.json()
        text_result = result.get("result", "No result returned")
        return text_result, text_result

    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to server: {str(e)}"
        return error_msg, error_msg
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return error_msg, error_msg

def toggle_output(choice, raw_text):
    """Toggle between raw text and markdown output"""
    if choice == "Origin":
        return (
            gr.Textbox(visible=True),
            gr.Markdown(visible=False)
        )
    else:  # Markdown
        return (
            gr.Textbox(visible=False),
            gr.Markdown(visible=True, value=raw_text)
        )

# Gradio interface
with gr.Blocks(title="NECJN Document->MarkDown Service UI") as demo:
    gr.Markdown("# NECJN Document->MarkDown Service UI")
    gr.Markdown("Upload an image and configure parameters to perform OCR using DeepSeek OCR.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(
                label="OCR Prompt",
                value=COMMON_CONFIG.ocr_prompt,
                lines=3,
                placeholder="Enter OCR prompt (optional)"
            )
            level_input = gr.Radio(
                choices=["raw", "md_image", "md_text", "md_merged"],
                value="md_text",
                label="Processing Level"
            )
            submit_btn = gr.Button("Process Image")
            
        with gr.Column():
            # Raw text output
            raw_output = gr.Textbox(label="OCR Result (Raw)", lines=20, interactive=False, buttons=["copy"])
            # Markdown output
            markdown_output = gr.Markdown(label="OCR Result (Markdown)", visible=False)
            # Toggle switch
            output_toggle = gr.Radio(
                choices=["Origin", "Markdown"],
                value="Origin",
                label="Output Format"
            )

    submit_btn.click(
        fn=ocr_with_deepseek,
        inputs=[image_input, prompt_input, level_input],
        outputs=[raw_output, markdown_output]
    )

    # Toggle visibility of output components
    output_toggle.change(
        fn=toggle_output,
        inputs=[output_toggle, raw_output],
        outputs=[raw_output, markdown_output]
    )

    
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
