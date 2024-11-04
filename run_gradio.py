import gradio as gr
from process_file import process_file
import tempfile
import shutil
import os
from query import query_rag

def create_demo():
    
    with gr.Blocks() as demo:
        gr.Markdown("File Chat Assistant support PDF, HTML, Excel, csv, PPTX")
        gr.Markdown("Upload a File and chat about its contents")
        
        with gr.Row():
            file = gr.File(label="Upload File", type="filepath")
            status_text = gr.Textbox(label="Status")
            
        process_button = gr.Button("Add document")
        process_button.click(
            fn=process_file,
            inputs=file,
            outputs=status_text
        )

        chatbot_interface = gr.Chatbot(
            label="Chat History",
            height=400
        )
        
        msg = gr.Textbox(
            label="Type your message",
            placeholder="Ask a question about the file..."
        )
        
        send_btn = gr.Button("Send")
        
        # Set up event handlers
        # process_button.click(
        #     fn=process_pdf,
        #     inputs=pdf_file,
        #     outputs=status_text
        # )
        
        send_btn.click(
            fn=query_rag,
            inputs=[msg, chatbot_interface],
            outputs=[chatbot_interface]
        )
        # msg.submit(
        #     fn=chatbot.chat,
        #     inputs=[msg, chatbot_interface],
        #     outputs=[chatbot_interface]
        # )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()