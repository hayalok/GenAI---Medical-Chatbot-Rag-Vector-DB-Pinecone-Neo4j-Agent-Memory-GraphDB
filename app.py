import asyncio
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from src.prompt import *

# --- Import Neo4j Agent Memory ---
from neo4j_agent_memory import MemoryClient, MemorySettings

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASS = os.environ.get("NEO4J_PASS")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")

# -------------------------------------------------------------------------
# 1. Initialize Neo4j Memory Settings
# -------------------------------------------------------------------------
# We point the client directly to your hosted Neo4j instance.
# Shorthand string configurations point to OpenAI for LLM/Embeddings tasks.
memory_settings = MemorySettings(
    neo4j={
        "uri": NEO4J_URI,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASS,
        "database": NEO4J_DATABASE,
    },
    llm="openai/gpt-4o",
    embedding="openai/text-embedding-3-small",
)

# -------------------------------------------------------------------------
# RAG Setup (LangChain + Pinecone)
# -------------------------------------------------------------------------
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=500)

# Ensure system_prompt expects {memory} alongside {context}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


# -------------------------------------------------------------------------
# 2. Async Helper Function to Manage Memory Lifecycle
# -------------------------------------------------------------------------
async def process_chat_with_memory(user_msg, session_id):
    # Establish a temporary async context manager for the client operations
    async with MemoryClient(memory_settings) as memory:

        # A. Fetch memory block (Automatically resolves raw history + extracted facts)
        retrieved_memory = await memory.get_context(
            query=user_msg, session_id=session_id
        )

        # B. Run your synchronous LangChain pipeline using loop executor
        # This keeps the Flask loop happy while invoking LangChain
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: rag_chain.invoke(
                {"input": user_msg, "memory": str(retrieved_memory)}
            ),
        )

        # C. Store conversation context asynchronously back to the graph
        # This queues background pipelines to extract new entities/preferences
        await memory.short_term.add_message(
            session_id=session_id, role="user", content=user_msg
        )
        await memory.short_term.add_message(
            session_id=session_id, role="assistant", content=response["answer"]
        )

        return response["answer"]
    
# -------------------------------------------------------------------------
# 3. Synchronous Flask Route Execution
# -------------------------------------------------------------------------
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")

    # Set up a continuous session identifier for the user
    session_id = "session_alok_001"

    # Execute our async stack inside the synchronous route wrapper
    answer = asyncio.run(process_chat_with_memory(msg, session_id))

    print(f"Response: {answer}")
    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
