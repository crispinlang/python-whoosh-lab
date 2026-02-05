import os
import pickle
import random
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from tqdm import tqdm
import shutil

SOURCE_DATA_DIR = "pmid2contents"  # created by I_fetch_pubmed_CL.py
EVAL_INDEX_DIR = "index_mesh_eval" # New separate index for this experiment
TEST_SET_SIZE = 100 # for fast testing, can be improved when more powerful system available

def load_data_from_pickles():
    all_docs = []
    files = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith('.pkl')]
    
    for filename in tqdm(files[:5]): 
        filepath = os.path.join(SOURCE_DATA_DIR, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f) # data is {pmid: [title, abstract, mesh_list]}
            
            for pmid, content in data.items():
                # content[0] is title, content[1] is abstract, content[2] is mesh list
                title = content[0] if content[0] else ""
                abstract = content[1] if content[1] else ""
                mesh = content[2] if content[2] else []
                
                combined_text = f"{title} {abstract}"
                
                all_docs.append({
                    "pmid": str(pmid),
                    "text": combined_text,
                    "mesh": mesh
                })
                
    print(f"Loaded {len(all_docs)} total documents.")
    return all_docs

# Step
def build_eval_index(train_docs, test_docs):

    if os.path.exists(EVAL_INDEX_DIR):
        shutil.rmtree(EVAL_INDEX_DIR)
    os.mkdir(EVAL_INDEX_DIR)

    schema = Schema(
        pmid=ID(stored=True, unique=True),
        content=TEXT(stored=True) 
    )
    
    ix = index.create_in(EVAL_INDEX_DIR, schema)
    writer = ix.writer()

    full_corpus = train_docs + test_docs

    for doc in tqdm(full_corpus):
        writer.add_document(
            pmid=doc['pmid'],
            content=doc['text']
        )
    writer.commit()
    return ix

# (Step 4.3)
def evaluate_mesh_predictions(ix, test_docs):
    print(f"\nStarting evaluation on {len(test_docs)} test documents...")
    
    hits_at_1 = 0
    hits_at_5 = 0
    valid_queries = 0
    
    parser = QueryParser("content", ix.schema, group=OrGroup)

    with ix.searcher() as searcher:
        for doc in tqdm(test_docs):
            target_pmid = doc['pmid']
            mesh_terms = doc['mesh']
                
            valid_queries += 1

            query_str = " ".join(mesh_terms)
            
            try:
                query = parser.parse(query_str)
                # Get top 5 results
                results = searcher.search(query, limit=5)
                
                found_pmids = [hit['pmid'] for hit in results]
                
                if len(found_pmids) > 0 and found_pmids[0] == target_pmid:
                    hits_at_1 += 1
                
                if target_pmid in found_pmids:
                    hits_at_5 += 1
                    
            except Exception:
                pass # Ignore parsing errors for weird characters

    # 4. Results
    if valid_queries == 0:
        print("No valid test documents with MeSH terms found.")
        return

    acc_1 = hits_at_1 / valid_queries
    acc_5 = hits_at_5 / valid_queries
    
    print("\n" + "="*30)
    print("MESH EVALUATION RESULTS")
    print("="*30)
    print(f"Total Test Queries: {valid_queries}")
    print(f"Accuracy@1: {acc_1:.4f} ({acc_1*100:.2f}%)")
    print(f"Accuracy@5: {acc_5:.4f} ({acc_5*100:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    all_data = load_data_from_pickles()
    
    if len(all_data) > 0:
        random.shuffle(all_data)

        split_idx = min(TEST_SET_SIZE, len(all_data) - 1)
        
        test_set = all_data[:split_idx]
        train_set = all_data[split_idx:]

        ix = build_eval_index(train_set, test_set)

        evaluate_mesh_predictions(ix, test_set)