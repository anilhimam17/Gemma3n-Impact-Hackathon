import gradio as gr
from gradio.themes import Ocean
from gradio_pdf import PDF

from pathlib import Path
import tempfile
import json

from src.query_engine import QueryEngine
from src.audio_transcription import AudioTranscription


class GradioInterface:
    """Implements the complete interface for a page in Gradio."""
    def __init__(self):

        # Gradio Layout and Component Instance Variables
        self.block_params = {"title": "Research Companion", "fill_height": True, "fill_width": True, "theme": Ocean()}
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

        # Audio Transcription Model
        self.whisper_audio = AudioTranscription()

        # Summary Place holder
        self.summary = ""

    # ==== Interface Builder ====
    def page(self) -> None:
        """Implements a complete Gradio Interface using the Components."""

        # The main block enclosing the entire interface.
        with gr.Blocks(**self.block_params) as demo:

            # Top Division for the Header
            with gr.Row():
                # Header
                _ = gr.Markdown(value="<h1 style='text-align: center; font-size: 32px;'>Research Companion 🤖</h1>")
            with gr.Row():
                # Description
                _ = gr.HTML(value=self.top_description)

            # Splitting the page into two sections
            with gr.Row():

                # PDF Viewer Section
                with gr.Column(scale=1):
                    pdf_comp = PDF(label="Upload PDF", interactive=True)

                # Chatbox Section
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        label="Research Companion", bubble_full_width=False, type="messages",
                        placeholder="Upload a PDF and begin the Research Journey."
                    )
                    multimodal_box = gr.MultimodalTextbox(
                        show_label=False, placeholder="Chat with the Research Companion",
                        sources=["microphone", "upload"], file_types=["file"], file_count="multiple"
                    )

                    # Columns for the buttons
                    with gr.Row():
                        with gr.Column():
                            submit_button = gr.Button(value="Submit")
                        with gr.Column():
                            summary_button = gr.Button(value="Generate Summary")
                    with gr.Row():
                        download_file = gr.File(label="Summary File", interactive=False)

                    # Multiple event triggers for the Chat Interface
                    gr.on(
                        triggers=[multimodal_box.submit, submit_button.click],
                        fn=self.run_query,
                        inputs=[pdf_comp, multimodal_box, chatbot],
                        outputs=[chatbot, multimodal_box]
                    )

                    # Event Listener for the Summary
                    summary_button.click(
                        fn=self.generate_summary, inputs=chatbot, outputs=chatbot
                    ).then(
                        fn=self.output_summary_file, inputs=None, outputs=download_file
                    )
        
        # Rendering the page.
        demo.launch()

    # ==== Helper Functions ====
    def run_query(self, pdf_path: str, multimodal_chat: dict, history: list):
        """Propagates the given query through the AI agent."""
        
        # If no multimodal_chat was sent
        if not multimodal_chat["text"] and not multimodal_chat["files"]:
            yield history, {"text": ""}
            return

        # If pdf was not uploaded
        if not pdf_path:
            history.append({"role": "user", "content": multimodal_chat["text"]})
            history.append({"role": "assistant", "content": "Please upload a document to begin research."})
            yield history, {"text": ""}
            return
        
        # Utilising the Text Input
        user_prompt: str = ""
        if multimodal_chat["text"]:
            history.append({"role": "user", "content": multimodal_chat["text"]})
            user_prompt = multimodal_chat["text"]
        # Applying Whisper for Audio Transcription
        elif self.whisper_audio.check_audio(multimodal_chat["files"]):
            transcription = self.whisper_audio.transcribe(multimodal_chat["files"])
            history.append({"role": "user", "content": transcription[0]["text"]})
            user_prompt = transcription[0]["text"]
            print(transcription)

        # Begining Thinking Process
        history.append({"role": "assistant", "content": "Thinking ..."})
        yield history, {"text": ""}

        # If the query engine was not created yet or the PDF has changed
        if self.query_engine is None or self.query_engine.file_path != Path(pdf_path):
            self.query_engine = QueryEngine(pdf_path)

        # Generating a response
        response_json = self.query_engine.run_query(user_prompt)
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
        history[-1] = {"role": "assistant", "content": final_response}
        yield history, {"text": ""}
        return
    
    def generate_summary(self, history: list):
        """Generates a summary by running inference on the model for the entire chat."""
        
        # If the summary is being generated before chatting
        if not self.query_engine:
            history.append({
                "role": "assistant", 
                "content": "Failed to generate a Summary. Please begin a Conversation or Upload a document."
            })
            yield history
            return
        
        # Indication for generating a summary
        history.append({"role": "user", "content": "Generate a summary for the conversation."})
        history.append({"role": "assistant", "content": "Generating the summary ..."})
        yield history

        # Formatting the history for the prompt template
        
        summary_prompt = """You will now generate a consise summary of the entire chat history.
        You will first provide a title for the entire chat history based on the domain of the conversation and follow-up with two 'End of line characters'.
        You will then reflect on keypoints, key ideas, key analogies and key sources used through out the chat history.
        You will compile all the information for easy consumption.
        'Keep it simple' and follow the thumb rule 'every single idea is one paragraph'.
        At the end provide a conclusion if you deem it necessary based on expertise in the domain.
        Ensure to use 'End of line characters' where necessary to format your response.
        Here is the Chat history:
        ----------------------------
        Answer: """

        # Generating a summary
        response_json = self.query_engine.run_query(summary_prompt)
        response_data = json.loads(response_json)
        
        # Storing the summary for usage in the follow-up methods
        self.summary = response_data.get("answer", "Sorry couldn't generate the summary")

        history[-1] = {"role": "assistant", "content": "Summary generated successfully. You can download it now."}
        yield history
        return
    
    def output_summary_file(self):
        """Utilises the generated summary and output a temporary file."""

        # Creating a temporary file
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / "summary.txt"
        temp_file.write_text(self.summary, encoding="utf-8")

        return str(temp_file)
