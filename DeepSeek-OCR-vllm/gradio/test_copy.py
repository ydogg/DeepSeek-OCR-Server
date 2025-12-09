import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# Test Copy Button")
    textbox = gr.Textbox(label="Test Box", value="Test content", buttons=["copy"])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)