import io
import os
from google import genai
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader

def getDISCERNScore(caption : str, llm : ChatGoogleGenerativeAI):
    DISCERN = {
        1 : ("Are the aims clear?", 1),
        2 : ("Does it achieve its aims?", 1),
        3 : ("Is it relevant?", 1),
        4 : ("Is it clear what sources of information were used to compile the publication (other than the author or producer)?", 1),
        5 : ("Is it clear when the information used or reported in the publication was produced?", 1),
        6 : ("Is it balanced and unbiased?", 1),
        7 : ("Does it provide details of additional sources of support and information?", 1),
        8 : ("Does it refer to areas of uncertainty?", 1),
        9 : ("Does it describe how each treatment works?", 1),
        10 : ("Does it describe the benefits of each treatment?", 1),
        11 : ("Does it describe the risks of each treatment?", 1),
        12 : ("Does it describe what would happen if no treatment is used?", 1),
        13 : ("Does it describe how the treatment choices affect overall quality of life?", 1),
        14 : ("Is it clear that there may be more than one possible treatment choice?", 1),
        15 : ("Does it provide support for shared decision-making?", 1)
    }

    handbook_path = "Evaluation/DISCERN/discern-handbook.pdf"
    loader = PyPDFLoader(handbook_path)
    docs = loader.load()
    handbook_document = docs[0]
    handbook = handbook_document.page_content

    class DISCERNScorer(BaseModel):
        """Always use this tool to structure your response to the user."""
        analysis: str = Field(description="The answer to the user's question")
        score: int = Field(ge=1, le=5, description="Rate the caption/publication against the question. For refernce use: 5 -> Yes, 2-4 -> Partially, 1 -> No")

    structured_llm = llm.with_structured_output(DISCERNScorer)

    analysis = {}
    total_score = 0
    for count, (principle, valid) in DISCERN.items():
        if valid == 1:
            system_message = SystemMessage(content=f"Use the following DISCERN handbook to analyze and rate each caption against the questions.\n\n{handbook}")
            prompt = f"Consider the captions attached from a YouTube video related to organ transplant. Analyze if the caption comply with the following question:\n\n{principle}\n\nGive it a rating between 1 and 5.\n\nCaptions:\n\n{caption}"
            human_message = HumanMessage(content=prompt)
            messages = [system_message, human_message]
            result = structured_llm.invoke(messages)
            analysis[count] = (result.score, result.analysis)
            total_score += result.score

    return total_score, analysis