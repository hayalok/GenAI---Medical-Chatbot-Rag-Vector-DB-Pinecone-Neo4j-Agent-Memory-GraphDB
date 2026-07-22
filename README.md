# End-to-end-Medical-Chatbot-Generative-AI
This project is GenAI project, which is RAG on PDF data. Also this project maintaines chat memory using Neo4j-Agent-Memory(neo4j_agent_memory) lib using neo4j graph db.

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate.bat
```



### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NEO4J_URI = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NEO4J_USERNAME = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NEO4J_PASS = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
NEO4J_DATABASE = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:8080
```


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone   
