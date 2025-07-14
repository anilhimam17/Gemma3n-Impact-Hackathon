import gradio as gr


class GradioInterface:
    """Implements the complete interface for a page in Gradio."""
    def __init__(self):
        self.block_params = {"title": "Research Companion", "fill_height": True, "fill_width": True}
        self.mark_style = 'background-color: #e1e9eb; padding: 0.2em 0.4em; border-radius: 5px;'
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

    # ==== Interface Builder ====
    def page(self) -> None:
        """Implements a complete Gradio Interface using the Components."""

        # The main block enclosing the entire interface.
        with gr.Blocks(**self.block_params) as demo:
            gr.Markdown(value="<h1 style='text-align: center; font-size: 32px;'>Research Companion 🤖</h1>")
            gr.HTML(value=self.top_description)
        
        # Rendering the page.
        demo.launch()

    # ==== Helper Functions ====