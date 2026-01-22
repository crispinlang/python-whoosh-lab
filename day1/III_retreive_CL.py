from whoosh import index
from II_index_CL import index_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup, FuzzyTermPlugin

ix = index.open_dir(index_dir)

def run_search_experiment(ix, query_str, mode="MULTIFIELD"):
    """
    Executes a search using different logic modes without reopening the index.
    
    Modes:
    - 'AND'       : Single-field (content), Exact match, All terms required.
    - 'OR'        : Multi-field, Exact match, Any term required.
    - 'MULTIFIELD': Multi-field, Exact match, All terms required (Default).
    - 'FUZZY'     : Multi-field, Approximate match (typo tolerant).
    """
    
    # open the searcher
    with ix.searcher() as s:
        if mode == "AND":
            qp = QueryParser("body", schema=ix.schema)
            
        elif mode == "OR":
            qp = QueryParser("body", schema=ix.schema, group=OrGroup)

        elif mode == "MULTIFIELD":
            qp = MultifieldParser(["title", "content"], schema=ix.schema, group=OrGroup)
            
        elif mode == "FUZZY":
            qp = MultifieldParser(["title", "body"], schema=ix.schema)
            qp.add_plugin(FuzzyTermPlugin())

        # searching
        q = qp.parse(query_str)
        results = s.search(q, limit=5)

        # output
        print(f"\n" + "="*40)
        print(f"MODE: {mode} | QUERY: '{query_str}'")
        print(f"Parsed as: {q}")
        print(f"Total Hits: {len(results)}")
        print("-" * 40)

        for i, hit in enumerate(results):
            print(f"  {i+1}. [{hit.score:.4f}] {hit.get('title', 'content')}")


## testing:
#run_search_experiment(ix, "acid fluid", mode="AND")
#run_search_experiment(ix, "acid fluid", mode="OR")
#run_search_experiment(ix, "acid fluid", mode="MULTIFIELD")
run_search_experiment(ix, "fluiids~1", mode="FUZZY")