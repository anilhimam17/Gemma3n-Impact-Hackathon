from pydantic import BaseModel, Field
from enum import Enum


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
        This text should be fully formatted in Markdown, including any equations using LaTeX syntax (eg. $ h=6.64 * 10 ^ {-34} $)."""
    )
    follow_up_questions: list[str] = Field(
        description="""A maximum of three follow-up questions that the user's query relates to
        as chain of thought prompts."""
    )
    citations: list[Citation] = Field(
        description="A list of all source citations from the document that were used to formulate the answer."
    )


class SummaryResponse(BaseModel):
    """Class implements the expected structured response for a Summary from the LLM."""
    title: str = Field(
        description="An intuitive title for the entire conversation based on the name of the document."
    )
    summary: str = Field(
        description="""A comprehensive summary of the entire conversation.
        This text should highlight all the key points, ideas and analogies used in the conversation."""
    )


class SimpleResponse(BaseModel):
    """Class implements the expected structured response for a Simple Query to the LLM."""
    answer: str = Field(
        description="""A simple descriptive answer to the user's query."""
    )


class ResponseTypes(str, Enum):
    """Enumeration of all the response types."""
    RESEARCH = "research"
    SUMMARY = "summary"
    SIMPLE = "simple"


class ToolInput(BaseModel):
    """Input schema for the tools used by the agent."""
    query: str