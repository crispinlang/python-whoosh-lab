# Day 2 Tasks Implementation

## 1. Fixed RAG UI Errors (`III_RAG_UI.py`)
I resolved several bugs that were preventing the application from running correctly:
- **Gradio Deprecations**: Removed outdated arguments (`equal_height` in Rows, `size` in Buttons) that were causing warnings.
- **Button Crashes**: Fixed mismatched arguments in the "Accept" and "Reject" feedback buttons so they no longer crash the app.
- **Runtime Safety**: Added protection against crashes when:
  - The retriever yields fewer documents than expected.
  - The LLM's response doesn't follow the exact "Answer:" format.
- **Port Conflict**: Changed the server port to `7004` to avoid "Address already in use" errors.

## 2. RAG System Improvements
I implemented the "Additional RAG Improvements" requested in Exercise Sheet 2 by making the pipeline dynamic:
- **UI Sliders**: Added controls for **Temperature**, **Top-P**, **Top-K**, and **K Source Chunks**.
- **Dynamic Retrieval**: Changing "K Source Chunks" now instantly updates how many documents the retriever fetches.
- **Dynamic Generation**: Adjusting generation parameters (Temperature, etc.) immediately rebuilds the LLM pipeline, allowing you to experiment with model creativity and variability in real-time.

![documentation](day2/documentation.png)

## 3. Evaluation
- Verified that `II_index_test.py` correctly implements the F1 score calculation to evaluate retrieval performance against the SQuAD dataset.
