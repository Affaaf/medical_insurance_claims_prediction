import pickle
import openai
import os
from dotenv import load_dotenv
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from constants.constant import Const

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

pickle_file = Const.PICKLE_FILE
embedding_model = Const.EMBEDDING_MODEL


def concatenate_input(input_dict, exclude_keys=None, missing_value="Missing"):
    """
    Concatenate all values of the input dictionary except excluded keys.
    Replace None/NaN with missing_value.
    """
    exclude_keys = exclude_keys or []
    values = [
        str(input_dict[k]) if input_dict.get(k) is not None else missing_value
        for k in input_dict.keys() if k not in exclude_keys
    ]
    return " ".join(values)

def get_text_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text using OpenAI embeddings API.
    """
    response = openai.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding

def load_embeddings_pickle(filename):
    """
    Load embeddings from pickle row-by-row.
    Returns a list of dicts: each has embedding and general_remark
    """
    records = []
    with open(filename, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
    return records

def search_top_k(input_dict, k=5, exclude_keys=None):
    """
    Take input dict, generate embedding, and return top-k most similar records
    based on cosine similarity.
    """
    exclude_keys = exclude_keys or ["GeneralRemark"]
    
    # Concatenate input values
    concatenated_text = concatenate_input(input_dict, exclude_keys)
    
    # Generate embedding for input
    input_embedding = get_text_embedding(concatenated_text)
    
    # Load saved embeddings
    saved_records = load_embeddings_pickle(pickle_file)
    
    # Compute cosine similarity
    embeddings_matrix = np.array([r["embedding"] for r in saved_records])
    input_vec = np.array(input_embedding).reshape(1, -1)
    
    similarities = cosine_similarity(input_vec, embeddings_matrix)[0]
    
    # Get top k indices
    top_k_idx = similarities.argsort()[::-1][:k]
    
    # Collect top k results
    top_results = []
    for idx in top_k_idx:
        rec = saved_records[idx]
        top_results.append({
            "general_remark": rec["general_remark"],
            "metadata": rec["metadata"],
            "similarity": float(similarities[idx])
        })
    
    return top_results
