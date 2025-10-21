import io
import os
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

def getHONcodeScore(caption : str, llm : ChatGoogleGenerativeAI):
    HONcode = {
        1 : ("Any medical advice provided and hosted on this site will only be given by medically trained and qualified professionals unless a clear statement is made that a piece of advice offered is from a non-medically qualified individual/organisation.", 1),
        2 : ("The information provided on this site is designed to support, not replace, the relationship that exists between a patient/site visitor and his/her existing physician.", 1),
        3 : ("Confidentiality of data relating to individual patients and visitors to a medical Website, including their identity, is respected by this Website. The Website owners undertake to honour or exceed the legal requirements of medical information privacy that apply in the country and state where the Website and mirror sites are located.", 1),
        4 : ("Where appropriate, information contained on this site will be supported by clear references to source data and, where possible, have specific HTML links to that data. The date when a clinical page was last modified will be clearly displayed (e.g. at the bottom of the page).", 1),
        5 : ("Any claims relating to the benefits/performance of a specific treatment, commercial product or service will be supported by appropriate, balanced evidence in the manner outlined in principle.", 1),
        6 : ("The designers of this Website will seek to provide information in the clearest possible manner and provide contact addresses for visitors that seek further information or support. The web-master will display his/her e-mail address clearly throughout the Website.", 1),
        7 : ("Support for this Website will be clearly identified, including the identities of commercial and non-commercial organisations that have contributed funding, services or material for the site.", 1),
        8 : ("If advertising is a source of funding it will be clearly stated. A brief description of the advertising policy adopted by the Website owners will be displayed on the site. Advertising and other promotional material will be presented to viewers in a manner and context that facilitates differentiation between it and the original material created by the institution operating the site.", 1)
    }

    class HONcodeScorer(BaseModel):
        """Always use this tool to structure your response to the user."""
        analysis: str = Field(description="The answer to the user's question")
        score: int = Field(ge=0, le=1, description="Give it a compliance: 1 if the caption is compliant with the principle, 0 otherwise.")

    structured_llm = llm.with_structured_output(HONcodeScorer)

    analysis = {}
    total_score = 0
    for count, (principle, valid) in HONcode.items():
        if valid == 1:
            prompt = f"Consider the captions attached from a YouTube video related to heart transplant. Analyze if the caption comply with the following principle:\n\n{principle}\n\nGive it a compliance score 1 if compliant, 0 otherwise.\n\nCaptions:\n\n{caption}"
            result = structured_llm.invoke(prompt)
            analysis[count] = (result.score, result.analysis)
            total_score += result.score

    return total_score, analysis