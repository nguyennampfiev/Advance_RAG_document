import gradio as gr
from process_file import process_file
from query import query_rag
import os

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("### File Chat Assistant (Supports PDF, HTML, Excel, CSV, PPTX)")
        gr.Markdown("**Upload a File and chat about its contents**")
        
        with gr.Row():
            file = gr.File(label="Upload File", type="filepath")
            status_text = gr.Textbox(label="Status")
        
        process_button = gr.Button("📂 Add Document")
        process_button.click(
            fn=process_file,
            inputs=file,
            outputs=status_text
        )

        chatbot_interface = gr.Chatbot(label="Chat History", height=400, type="messages")
        msg = gr.Textbox(label="Type your message", placeholder="Ask a question about the file...")

        msg.submit(
            fn=query_rag,
            inputs=[msg, chatbot_interface],
            outputs=[chatbot_interface, msg]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
