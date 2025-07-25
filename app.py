from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import System_prompt  # Make sure this is defined correctly

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional if you're also using OpenAI

# Set environment for consistency (not strictly required)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize Flask App
app = Flask(__name__)

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone Vector Store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load Gemini Model
chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or gemini-2.0-flash
    google_api_key=GEMINI_API_KEY
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", System_prompt),
    ("human", "{input}")
])

# Retrieval-Augmented Generation (RAG) Chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")  # Make sure your template is named correctly


# @app.route("/ask", methods=["POST"])
# def ask():
#     try:
#         user_input = request.json.get("question")
#         response = rag_chain.invoke({"input": user_input})
#         return jsonify({"answer": response["answer"]})
#     except Exception as e:
#         print("Error during chat:", str(e))
#         return jsonify({"answer": "Sorry, there was an error. Please try again later."})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json.get("question1")
        print("User asked:", user_input)  # Debug log

        response = rag_chain.invoke({"input": user_input})
        print("Response generated:", response)  # Debug log

        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print("Error during chat:", str(e))
        return jsonify({"answer": "Sorry, there was an error. Please try again later."})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)