# Evaluation Questions

## 1. What do these accuracy values reveal about your system performance?  
- After using 100 test queries to test the system, the results reveil that the system is good at finding the right document generally (61% in the top 5) but struggles to rank it first (34%).

## 2. How do indexing choices and query formulation affect results?  
- The current search method treats queries as loose collections of words rather than specific concepts, and it fails to connect formal technical tags (like "Neoplasms") with the everyday language authors use (like "Cancer"). This results in the indexing failing to identify the top result because of a language mismatch.

## 3. What improvements could be made?
- Future improvements should enforce exact phrase searching to filter out irrelevant matches that share only single words. Performance would also improve by giving more weight to document titles and automatically adding synonyms to queries, ensuring the system can recognize a match even when the terminology differs slightly.



