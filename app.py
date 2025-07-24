from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import System_prompt  # Ensure this is a string

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set API keys as environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Flask App
app = Flask(__name__)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone vector store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load Gemini model
chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", System_prompt),
    ("human", "{input}")
])

# Create document -> answer chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")  # Ensure this file exists in templates folder

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json.get("question")
        print(f"\n🧠 User Input: {user_input}", flush=True)

        # Retrieve docs
        retrieved_docs = retriever.invoke(user_input)
        print("\n📄 Retrieved Documents:", flush=True)
        for i, doc in enumerate(retrieved_docs):
            print(f"  [{i+1}] {doc.page_content[:300]}...", flush=True)

        # Show prompt being sent to Gemini
        formatted_prompt = prompt.format_messages(input=user_input)
        print("\n📝 Gemini Prompt:", flush=True)
        for msg in formatted_prompt:
            print(f"  [{msg.type.upper()}]: {msg.content}", flush=True)

        # Generate response
        response = question_answer_chain.invoke({
            "input": user_input,
            "context": retrieved_docs
        })

        print("\n💬 Gemini Response:", flush=True)
        print(response.get("answer", "No 'answer' field returned."), flush=True)

        return jsonify({"answer": response.get("answer", "Sorry, I couldn't find a good answer.")})

    except Exception as e:
        print("❌ Error during chat:", str(e), flush=True)
        return jsonify({"answer": "An error occurred. Please try again later."})

# Start Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
