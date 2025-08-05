import gradio as gr
from gradio.themes import Ocean
from gradio_pdf import PDF

from pathlib import Path
from typing import AsyncGenerator, Any
import tempfile
import json

from src.response_structures import ResponseTypes
from src.routing_agent import RoutingAgent
from src.audio_transcription import AudioTranscription


class GradioInterface:
    """Implements the complete interface for a page in Gradio."""
    def __init__(self) -> None:

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
        self.routing_agent: RoutingAgent | None = None

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
                        sources=["microphone"], file_types=["file"], file_count="multiple"
                    )

                    # Columns for the buttons
                    with gr.Row():
                        with gr.Column():
                            submit_button = gr.Button(value="Submit")
                        with gr.Column():
                            summary_button = gr.Button(value="Generate Conversation Summary")
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
    async def run_query(self, pdf_path: str, multimodal_chat: dict, history: list) -> AsyncGenerator[tuple[list, dict[str, str]], Any]:
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

        # Begining Thinking Process
        history.append({"role": "assistant", "content": "Thinking ..."})
        yield history, {"text": ""}

        # If the query engine was not created yet or the PDF has changed
        if self.routing_agent is None or self.routing_agent.query_engine.file_path != Path(pdf_path):
            self.routing_agent = RoutingAgent(pdf_path)

        # Generating a response
        response_json, response_type = await self.routing_agent.resolve_route(user_prompt)
        response_data = json.loads(str(response_json))

        if response_type == ResponseTypes.RESEARCH:
            # Primary Answer
            answer = response_data.get("answer", "Sorry, I couldn't generate an answer could you please try again.")

            # Followup Questions
            follow_up_questions = response_data.get("follow_up_questions", "Sorry no follow-up questions were found.")
            follow_up_questions_md = ""
            if follow_up_questions and isinstance(follow_up_questions, list):
                follow_up_questions_md = "\n\n---\n**Follow-Up Chain of Thought**\n"
                for question in follow_up_questions:
                    follow_up_questions_md += f"- {question}\n"

            # Citations
            citations = response_data.get("citations", [])
            citations_markdown = ""
            if citations:
                citations_markdown = "\n\n---\n**Sources & Citations**\n"
                for idx, citation in enumerate(citations):
                    source_text = citation.get('source_text', 'N/A').replace('\n', ' ')
                    citations_markdown += ( 
                        f"**{idx + 1}. Source from Page {citation.get('page_number', 'N/A')}:**\n" 
                        f"> {source_text}\n\n" 
                        f"*Simplified Explanation:*\n{citation.get('simplification', 'N/A')}\n\n" 
                    )
            
            final_response = f"{answer}{follow_up_questions_md}{citations_markdown}\n"
            history[-1] = {"role": "assistant", "content": final_response}

        elif response_type == ResponseTypes.SIMPLE:
            answer = response_data.get("answer", "Sorry, no answer was found.")
            history[-1] = {"role": "assistant", "content": answer}

        yield history, {"text": ""}
        return
    
    async def generate_summary(self, history: list) -> AsyncGenerator[list[dict[str, str]], Any]:
        """Generates a summary by running inference on the model for the entire chat."""
        
        # If the summary is being generated before chatting
        if not self.routing_agent:
            history.append({
                "role": "assistant", 
                "content": "Failed to generate a Summary. Please begin a Conversation or Upload a document."
            })
            yield history
            return
        
        # Indication for generating a summary
        history.append({"role": "user", "content": "Generate a conceptual summary for our conversation."})
        history.append({"role": "assistant", "content": "Generating the summary ..."})
        yield history

        # Generating a summary
        response_json, _ = await self.routing_agent.resolve_route(
            "Generate a concept-driven summary for our entire conversation"
        )
        response_data = json.loads(str(response_json))
        
        # Storing the summary for usage in the follow-up method
        self.summary = response_data.get("title", "Sorry the summary title couldn't be processed.")
        self.summary += "\n\n" + response_data.get("summary", "Sorry the summary couldn't be processed.")

        if "Sorry" in self.summary:
            history[-1] = {"role": "assistant", "content": "Sorry the Summary couldn't be generated. Please try again"}
        else:
            history[-1] = {"role": "assistant", "content": "Summary generated successfully. You can download the markdown file now."}
        yield history
        return
    
    def output_summary_file(self) -> str:
        """Utilises the generated summary and output a temporary file."""

        # Creating a temporary file
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / "summary.md"
        temp_file.write_text(self.summary, encoding="utf-8")

        return str(temp_file)
