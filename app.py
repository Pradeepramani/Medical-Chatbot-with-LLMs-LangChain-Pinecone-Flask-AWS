from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import System_prompt  # Ensure this is defined correctly

# Load environment variables from .env file
load_dotenv()

# Get API Keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set as environment variables (optional but safe)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize Flask App
app = Flask(__name__)

# Load Sentence Transformer Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone Vector Store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Gemini Model (via LangChain)
chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# Prompt Template (System + Human)
prompt = ChatPromptTemplate.from_messages([
    ("system", System_prompt),
    ("human", "{input}")
])

# Chain setup
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")  # Make sure this HTML exists


@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json.get("question")
        print(f"\n🧠 User Input: {user_input}")

        # Step 1: Retrieve docs
        retrieved_docs = retriever.invoke(user_input)
        print("\n📄 Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  [{i+1}] {doc.page_content[:300]}...")  # Limit output for readability

        # Step 2: Show formatted prompt
        formatted_prompt = prompt.format_messages(input=user_input)
        print("\n📝 Gemini Prompt:")
        for msg in formatted_prompt:
            print(f"  [{msg.type.upper()}]: {msg.content}")

        # Step 3: RAG + Gemini answer
        response = question_answer_chain.invoke({
            "input": user_input,
            "context": retrieved_docs
        })

        print("\n💬 Gemini Response:")
        print(response.get("answer", "No 'answer' field returned."))

        return jsonify({"answer": response.get("answer", "Sorry, I couldn't find a good answer.")})

    except Exception as e:
        print("❌ Error during chat:", str(e))
        return jsonify({"answer": "An error occurred. Please try again later."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
