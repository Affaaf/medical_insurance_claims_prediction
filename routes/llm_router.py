from models.schemas import RawInput
from fastapi import APIRouter, HTTPException
from utils.prediction import predict_claim
from utils.llm_inference import prepare_payload, explain_prediction
from utils.remarks_predictions import search_top_k


router = APIRouter(prefix="/claims", tags=["LLM Parser"])

@router.post("/prediction")
def extract_payload(data: RawInput):
    """
    Takes raw claim text and returns a structured ClaimPayload.
    Missing fields are returned as null.
    """
    try:
        parsed_data= prepare_payload(data)
        status_response = predict_claim(parsed_data)
        inputs = parsed_data.dict()

        status = status_response["prediction"]

        if status == 0:
            remarks = search_top_k(inputs, k=5)
            print("\nTop 5 similar GeneralRemarks:", remarks)

        else:
            remarks = "Missing Remarks for paid"

        final_answer = explain_prediction(status_response, remarks)

        return final_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
