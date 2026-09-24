from fastapi import FastAPI
from routes import llm_router

app = FastAPI(title="Claim Status Prediction API with LLM")

app.include_router(llm_router.router)

@app.get("/")
def home():
    return {"message": "Claim Prediction API is running"}
