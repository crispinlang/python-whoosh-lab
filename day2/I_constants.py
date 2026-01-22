# -*- coding: utf-8 -*-
"""
Task Sparse IR

Created on Tue November 11 09:14:24 2025

@author: agha
"""

# Indexer
persist_path= 'chroma_db'
models_path = 'models'
source_path = '../squad/texts'
embeddings_model_name='intfloat/multilingual-e5-base'

#chunk_size = 500
chunk_size = 500

#chunk_overlap = 50
chunk_overlap = 50
# increasing this increased the F1 score by 

# LLM generator
llm_base = 'llms'

#model_n_ctx=512
model_n_ctx=512
# did not change F1 score

#model_n_batch=32
model_n_batch=32

#K_source_chunks=3
K_source_chunks=5
# increasing this increased the F1 score by 0.05

#num_gpu_layers =20
num_gpu_layers =-1 # uses as many as are available