import pandas as pd
import pickle
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv
from constants.constant import Const


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
PICKLE_FILE = Const.PICKLE_FILE
EMBEDDING_MODEL = Const.EMBEDDING_MODEL



def concatenate_row(row, exclude_cols=None, missing_value="Missing"):
    """
    Concatenate all columns of a row except the ones in exclude_cols.
    Replace missing/NaN values with a placeholder instead of skipping.
    """
    exclude_cols = exclude_cols or []
    values = [
        str(row[col]) if pd.notnull(row[col]) else missing_value
        for col in row.index
        if col not in exclude_cols
    ]
    return " ".join(values)


def get_text_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text using OpenAI embeddings API.
    """
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def save_single_record(record, filename):
    """
    Append a single record to a pickle file.
    """
    with open(filename, "ab") as f:
        pickle.dump(record, f)

def process_dataframe(df, exclude_cols=None):
    """
    Process DataFrame row-by-row to generate embeddings for each row
    and save immediately to pickle.
    """
    exclude_cols = exclude_cols or []
    counter = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        concatenated_text = concatenate_row(row, exclude_cols)
        general_remark = row.get("GeneralRemark", "")
        counter += 1
        print(f"\n\n Concatenated Text for Row:{counter}:", concatenated_text)
        print(f"general_remark for Row:{counter}:", general_remark)
        
        embedding = get_text_embedding(concatenated_text)
        
        record = {
            "index": idx,
            "embedding": embedding,
            "general_remark": general_remark,
            "metadata": {
                "Claim_status": int(row["Claim_status"]) if "Claim_status" in row else None,
                "InsuID": row.get("InsuID", None),
                "InsuPlanID": row.get("InsuPlanID", None)
            }
        }
        
        save_single_record(record, PICKLE_FILE)


if __name__ == "__main__":
    combined_df = pd.read_csv("billingpaymentprediction/unpaid_remarks_final.csv")

    exclude_columns = ["GeneralRemark"]
    
    if os.path.exists(PICKLE_FILE):
        os.remove(PICKLE_FILE)
    
    process_dataframe(combined_df, exclude_cols=exclude_columns)
    
    print(f"Processing complete! Embeddings saved row-by-row in {PICKLE_FILE}")
