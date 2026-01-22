import os
import pickle
from tqdm import tqdm
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD

# Schema
schema = Schema(
    pmid=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    body=TEXT(stored=True),
    mesh=KEYWORD(stored=True, lowercase=True, commas=True)
)

# Index
index_dir = "pubmed_index"
source_data_dir = "pmid2contents"

def get_index():
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    ix = create_in(index_dir, schema)
    writer = ix.writer()

    files_to_index = [f for f in os.listdir(source_data_dir) if f.endswith('.pkl')]

    for filename in tqdm(files_to_index):
        filepath = os.path.join(source_data_dir, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        for pmid, content in data.items():
            mesh_str = ",".join(content[2]) if content[2] else ""
            writer.add_document(
                pmid=str(pmid),
                title=content[0],
                body=content[1],
                mesh=mesh_str
            )
    
    writer.commit()

if __name__ == "__main__":
    get_index()