from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Class implements the Expected Structured response for the LLM when citating other works."""
    page_number: int = Field(
        description="The exact page number from the document where the supporting text was found."
    )
    source_text: str = Field(
        description="The verbatim text chunk from the source document that supports the answer."
    )
    simplification: str = Field(
        description="""A brief simplification of the verbatim text chunk from the source 
        document that supports the answer for better understanding."""
    )


class ResearchResponse(BaseModel):
    """Class implements the Expected Structured response from the LLM's."""
    answer: str = Field(
        description="""The primary, comprehensive answer to the user's query.
        This text should be fully formatted in Markdown, including any equations using LaTeX syntax (eg. $$ h=6.64 * 10 ^ {-34} $$)."""
    )
    citations: list[Citation] = Field(
        description="A list of all source citations from the document that were used to formulate the answer."
    )
