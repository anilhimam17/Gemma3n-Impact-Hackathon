# 🔬 Research Companion | Gemma 3n Impact Challenge | Google DeepMind Hackathon (Kaggle)

A `Multimodal`, `Conversational AI` research assistant powered by **Google's Gemma 3n**. This tool allows users to upload complex technical papers and engage in a natural dialogue to understand key concepts, simplify equations, and accelerate their research process.

## Features
- **Document Q&A:** Upload a PDF and ask direct questions about its content.
- **Concept Explanation:** Ask the AI to explain complex topics from the paper in simpler terms.
- **(Planned) Proactive Tutoring:** The AI will suggest follow-up questions and generate practice exercises.
- **(Planned) Multimodal Analysis:** Ask about figures, tables, and diagrams within the document.

## Tech Stack
- **LLM:** Google Gemma 3n
- **RAG Framework:** LlamaIndex
- **UI:** Gradio
- **Vector Store:** FAISS (in-memory)

## Setup & Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/anilhimam17/Gemma3n-Impact-Hackathon.git
   cd Gemma3n-Impact-Hackathon
   ```
### 2. Install dependencies:
- If you have **UV package manager**: 
    ```bash
    uv sync --all-groups
    ```

    > **Note:** [UV](https://github.com/astral-sh/uv)
    >
    > Checkout UV its an extremely fast Python Package and Project Manager written in Rust.
- If you have **pip**:
    - `Option 1` **(works for pip v23.1+ or newer)**. From the root path of the repository execute the following command to download all the dependencies.
    ```bash
    pip install .
    ```
    - `Option 2` **(works for older versions of pip)**. From the root path of the repository install all the dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
### 3. Create the Project Environemnt to automate the build
- To reduce the pain of debugging and ensuring modularity all the models, paths and other project specific configuration are provided in an example `.example-env` file.
- To use the example configurations that present in the repo:
    ```bash
    cp .example-env .env
    ```
- Feel free to change the configurations to your preferences by making modifications to the `.env` file.
### 4. Run the program:
- If you have **UV package manager**: 
    ```bash
    uv run main.py
    ```
- If you have **pip**:
    - `Option 1`. If you are using a Mac
    ```bash
    python3 main.py
    ```
    - `Option 2`. If you are using Windows
    ```bash
    python main.py
    ```