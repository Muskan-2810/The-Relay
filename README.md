***# The Relay***



***\*\*An AI-powered Research Paper Question Answering and Retrieval Evaluation System\*\****



***The Relay is a research-focused application designed to answer questions using research papers. It combines document processing, text chunking, information retrieval, scope validation, and citation verification to improve the reliability of research-based answers.***



***## Key Features***



***- Research paper text extraction and processing***

***- Document chunking for retrieval***

***- Research question scope validation***

***- Relevant information retrieval***

***- Citation validation***

***- Retrieval comparison and evaluation experiments***

***- Concurrency and retry testing***



***## Tech Stack***



***- Python***

***- Retrieval-Augmented Generation (RAG)***

***- Information Retrieval***

***- REST API***

***- CSV-based experiment evaluation***

***- Automated testing with pytest***



***## Project Structure***



***```text***

***The-Relay/***

***├── app/***

***├── data/***

***│   └── questions/***

***├── experiments/***

***├── results/***

***├── test\_citation\_validator.py***

***├── test\_negative\_retrieval.py***

***├── test\_retrieval\_baseline.py***

***├── test\_scope\_gate.py***

***├── test\_scope\_negative.py***

***└── .gitignore***

***```***



***## Research Data***



***Research paper PDFs and extracted paper contents are not included in this public repository. Users must provide the relevant research papers and generate the required processed data locally.***



***## Experiments***



***The `experiments/` directory contains scripts for retrieval comparison, query expansion, concurrency, retry handling, and related evaluations.***



***The `results/` directory contains experiment output CSV files.***



***## Security***



***API keys and environment variables must be configured locally. Never commit `.env` files or expose secret credentials.***



***## Project Status***



***Developed as a research and engineering project focused on improving the reliability and evaluation of research-paper question answering systems.***

