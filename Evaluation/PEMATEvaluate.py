import io
import os
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

def getPEMATScore(caption : str, llm : ChatGoogleGenerativeAI):
    PEMAT = {
        1 : ("The material makes its purpose completely evident.", 1),
        2 : ("The material uses common, everyday language.", 1),
        3 : ("Medical terms are used only to familiarize audience with the terms. When used, medical terms are defined.", 1),
        4 : ("The material uses the active voice.", 1),
        5 : ('The material breaks or "chunks" information into short sections.', 1),
        6 : ("The material’s sections have informative headers.", 1),
        7 : ("The material presents information in a logical sequence.", 1),
        8 : ("The material provides a summary.", 1),
        9 : ("The material uses visual cues (e.g., arrows, boxes, bullets, bold, larger font, highlighting) to draw attention to key points.", 0),
        10 : ("Text on the screen is easy to read.", 1),
        11 : ("The material allows the user to hear the words clearly (e.g., not too fast, not garbled).", 1),
        12 : ("The material uses illustrations and photographs that are clear and uncluttered.", 1),
        13 : ("The material uses simple tables with short and clear row and column headings.", 1)
    }

    class PEMATScorer(BaseModel):
        """Always use this tool to structure your response to the user."""
        analysis: str = Field(description="The answer to the user's question")
        score: int = Field(ge=0, le=1, description="Give it a compliance: 1 if the caption is compliant with the principle, 0 otherwise.")

    structured_llm = llm.with_structured_output(PEMATScorer)

    analysis = {}
    total_score = 0
    for count, (principle, valid) in PEMAT.items():
        if valid == 1:
            prompt = f"Consider the captions attached from a YouTube video related to organ transplant. Analyze if the caption comply with the following principle:\n\n{principle}\n\nGive it a compliance score 1 if compliant, 0 otherwise.\n\nCaptions:\n\n{caption}"
            result = structured_llm.invoke(prompt)
            analysis[count] = (result.score, result.analysis)
            total_score += result.score

    return total_score, analysis