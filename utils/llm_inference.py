import os
import openai
import json

from models.schemas import Claim
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def prepare_payload(input):
    prompt = f"""
    - You are a medical claim parser. Extract all possible fields from the following text.
    - If a field is missing from categorical columns, return null and for 0 for numeric columns(paid columns). 
    - Return only JSON.

    Expected fields:
        - InsuID, InsuPlanID, Diag1-Diag12, Proc1-Proc6, Modifier1-Modifier6, Paid2-Paid6, Proc2_Adj-Proc6_Adj.

    Important Note: 
        - Paid columns would be 1 for paid claims and 0 for Unpaid claims. Missing columns of paid series would also be 0.

    Text Input: {input}
    """
    response = openai.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical claim parser."},
            {"role": "user", "content": prompt},
        ],
    response_format=Claim,
    )

    response_content = response.choices[0].message.content
    parsed_dict = json.loads(response_content)

    parsed_claim = Claim(**parsed_dict)  

    return parsed_claim


def explain_prediction(result: dict, remarks:str) -> str:
    """
    Uses LLM to convert numeric prediction result into a natural human-friendly explanation.
    """
    try:
        prompt = f"""
        You are a helpful assistant that explains **medical claim prediction results** and their remarks in **plain, human-like language**.

        Given this result:
        {json.dumps(result, indent=2)}

        Here are the Remarks:
        {remarks}

        Your task:
        - Analyze the claim prediction result and remarks carefully.
        - Generate a short, **conversational summary (1- 2 sentences)** explaining whether the claim is **paid** or **unpaid**, based on the analysis of input values and model confidence (if available).
        - Write naturally, as if a human analyst is explaining the outcome — not like a machine.

        Remarks handling:
        - If the claim **status = 1 (paid)** → There will be only one remark: `"Missing Remarks for paid"`.  
        Include this remark naturally in your response.
        - If the claim **status = 0 (unpaid)** → There will be a list of top five matched remarks.  
        Select **the single most relevant remark** from the list based on the claim context.

        Your final output should include:
        1. A short, natural summary that references the **input analysis** (e.g., “Based on the provided input details…”).
        2. The selected remark, clearly stated but without phrases like “you might be looking for.”

        Example style:
        "The medical claim has been predicted to be **paid** with a very high confidence level of 99.93%. Based on the analysis of input values, the claim appears valid and thus approved. Remark: 'Missing Remarks for paid.'"
        """

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical claim prediction explainer."},
                {"role": "user", "content": prompt},
            ],
        )

        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        return f"Error generating explanation: {str(e)}"
