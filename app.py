import gradio as gr
from gradio_pdf import PDF
from src.query_engine import QueryEngine
from pathlib import Path
import json


class GradioInterface:
    """Implements the complete interface for a page in Gradio."""
    def __init__(self):

        # Gradio Layout and Component Instance Variables
        self.block_params = {"title": "Research Companion", "fill_height": True, "fill_width": True}
        self.mark_style = 'background-color: #c9c8c7; padding: 0.2em 0.4em; border-radius: 5px;'
        self.desciption_style = 'text-align: center; line-height: 2.5; font-size: 16px;'
        self.top_description = f"""
        <div style="{self.desciption_style}">
        Wanna dive into a 
        <mark style="{self.mark_style}">Research Paper</mark> or breakdown 
        <mark style="{self.mark_style}">Complex Literature</mark> let's get started.<br>
        For best results 
        <mark style="{self.mark_style}">Upload the Document below</mark> and start interacting with the Document by engaging in a 
        <mark style="{self.mark_style}">Conversation</mark> through the Chat Inferface.
        </div>
        """

        # Query Engine Parameters
        self.query_engine = None


    # ==== Interface Builder ====
    def page(self) -> None:
        """Implements a complete Gradio Interface using the Components."""

        # The main block enclosing the entire interface.
        with gr.Blocks(**self.block_params) as demo:

            # Sidebar for additional features
            with gr.Sidebar(position="left"):
                user_name = gr.Markdown(value="### Hi Username")

            # Top Division for the Header
            with gr.Row():
                # Header
                _ = gr.Markdown(value="<h1 style='text-align: center; font-size: 32px;'>Research Companion 🤖</h1>")
            with gr.Row():
                # Description
                _ = gr.HTML(value=self.top_description)

            # Splitting the page into two sections
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_comp = PDF(label="Upload PDF", interactive=True)
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        label="Research Companion", bubble_full_width=False, 
                        placeholder="Upload a PDF and begin the Research Journey."
                    )
                    message_box = gr.Textbox(label="Ask questions about the Document", interactive=True)
                    voice_input = gr.Audio(label="Speak with the Document", interactive=False)
                    submit_button = gr.Button(value="Submit")

                    gr.on(
                        triggers=[message_box.submit, submit_button.click],
                        fn=self.run_query,
                        inputs=[pdf_comp, message_box, chatbot],
                        outputs=[chatbot, message_box]
                    )
        
        # Rendering the page.
        demo.launch()

    # ==== Helper Functions ====
    def run_query(self, pdf_path: str, message: str, history: list):
        """Propagates the given query through the AI agent."""
        
        # If no message was sent
        if not message:
            yield history, ""
            return

        # If pdf was not uploaded
        if not pdf_path:
            history.append((message, "Please upload a PDF document first."))
            yield history, "" 
            return
        
        # Begin Thinking Process
        history.append((message, "Thinking..."))
        yield history, ""

        # If the query engine was not created yet or the PDF has changed
        if self.query_engine is None or self.query_engine.file_path != Path(pdf_path):
            self.query_engine = QueryEngine(pdf_path)

        # Generating a response
        response_json = self.query_engine.run_query(message)
        response_data = json.loads(response_json)
        answer = response_data.get("answer", "Sorry no answer was found")

        citations = response_data.get("citations", [])
        citations_markdown = ""
        if citations:
            citations_markdown = "\n\n---\n\n**Sources & Citations**\n\n"
            for idx, citation in enumerate(response_data.get("citations")):
                source_text = citation.get('source_text', 'N/A').replace('\n', ' ')
                citations_markdown += ( 
                    f"**{idx + 1}. Source from Page {citation.get('page_number', 'N/A')}:**\n" 
                    f"> {source_text}\n\n" 
                    f"*Simplified Explanation:*\n{citation.get('simplification', 'N/A')}\n\n" 
                )
        
        final_response = f"{answer}{citations_markdown}"
        history[-1] = (message, final_response)
        yield history, ""
        return
