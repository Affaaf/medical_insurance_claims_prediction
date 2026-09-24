##  Medical Billing Claims Status Prediction
 ---------------------------------------------------
 This project is a machine learning-based FastAPI service that predicts whether
 a medical billing claim will be Paid or Unpaid using a trained XGBoost model.
 It also generates LLM-based remarks explaining the reasoning behind the prediction.


## INSTALLATION GUIDE

## 1️ Clone the Repository
```
git clone https://github.com/Affaaf/medical_insurance_claims_prediction.git
```
```
cd medical_insurance_claims_prediction
```

## Create Model Directory
```
mkdir ml_models
```

## Download Trained Models
 Download the pre-trained models from the provided Google Drive link and paste them into ml_models
```
https://drive.google.com/drive/folders/1DSS_wM-dVss9slIUlx2GE7p7ULhFgnPo?usp=sharing
```


## Create and Activate Virtual Environment
```
python3 -m venv venv
source venv/bin/activate      # For Linux/Mac
# venv\Scripts\activate       # For Windows
```

## Install Required Packages
```
pip install -r requirements.txt
```

## Create Environment File
```
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```
## RUNNING THE APPLICATION

## Run in Production using PM2
```
pm2 start "gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000" --name fastapi-app
```

## Or Run Locally for Development
```
uvicorn app:app --reload
```

## API USAGE

## 🔹 Endpoint
### POST /predict-claim

### 🔹 Example Request Payload
```

{
"text": "The insurance ID is 13 under plan 3. The listed diagnoses include M25.512, M25.612, M54.12,
M54.13, M19.011, M19.012, and M25.511, while the remaining diagnosis fields are blank. The first 
procedure is 97530 with modifier GO CO. It was paid, and the adjustments include CO:253, CO:45, 
CO:59, HE:N782, and HE:N851. The second procedure, 97110, also has modifier GO CO, marked as 
paid, with the same adjustment codes — CO:253, CO:45, CO:59, HE:N782, and HE:N851. The third 
procedure is 97535, again with modifier GO CO, paid as well, and carrying the same adjustment 
reasons. All remaining procedures are blank. Predict whether it's paid or unpaid"
}

```
### 🔹 Example Response
```
"Based on the provided input details, the medical claim has been predicted to be **paid** with an 
exceptionally high confidence level of 99.97%. The analysis suggests that the claim is valid and 
approved as anticipated. Remark: \"Missing Remarks for paid.\""
```

##  API DOCUMENTATION

### Swagger UI: 
```
http://IP:8000/docs#/LLM%20Parser/extract_payload_llm_extract_post
```

## API Curl
```
curl -X 'POST' \
  'http://127.0.0.1:8000/claims/prediction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The insurance ID is 13 under plan 3. The listed diagnoses include M25.512, M25.612, M54.12"
}'
```