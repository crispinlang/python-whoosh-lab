# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 10:42:53 2025

@author: agha
"""
import os
import sys
import json
import chromadb
import numpy as np
from tqdm import tqdm
from I_constants import *
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# test_data = json.load(open('../squad/squad_multiple_contexts.json', 'r'))

# 1. Get the directory where THIS script lives (e.g., inside RAG/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the path dynamically: Go up one level (..), then into 'squad'
json_path = os.path.join(current_script_dir, 'squad', 'squad_multiple_contexts.json')

# 3. Load the file
with open(json_path, 'r') as f:
    test_data = json.load(f)

def get_f1(true_doc: set, pred_doc: set) -> float:

    if not true_doc and not pred_doc:
        return 1.0
    f1 = 2 * len(true_doc & pred_doc) / (len(true_doc) + len(pred_doc))
    return f1


if __name__ == "__main__":
    version = 1
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, cache_folder=models_path)
    chroma_client = chromadb.PersistentClient(persist_path)
    db = Chroma(persist_directory=persist_path,
                embedding_function=embeddings,
                collection_name="test_collection",
                client=chroma_client
                )

    retriever = db.as_retriever(search_type="similarity",
                                search_kwargs={"k": K_source_chunks})

    f1s = []
    for entry in tqdm(test_data):
        query = entry['text']
        entry_prediction = [doc.metadata["source"].split('/')[-1] for n, doc in enumerate(retriever.invoke(query))]
        entry_true = entry['sources']
        f1s.append(get_f1(set(entry_true), set(entry_prediction)))

    print("F1: %.2f"%(np.mean(f1s)))


# F1 score testing:

# Chunk size
# initial value: 500
# Increasing to 800 or decreasing to 300 didn't change the outcome of the F1

# Chunk overlap
# initial value: 50
# increasing the value to 

# Embedding type

# K_source_chunks


