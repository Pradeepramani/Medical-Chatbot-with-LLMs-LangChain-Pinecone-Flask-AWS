from flask import Flask, render_template, request, jsonify
from src.helper import load_retriever
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
retriever = load_retriever()

# Load Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-pro")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.form['user_input']
        print(f"\n🧠 User Input: {user_input}")

        # 1. Retrieve docs
        docs = retriever.get_relevant_documents(user_input)
        print(f"\n📄 Retrieved Documents:")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.page_content[:300]}...")

        # 2. Prepare context
        context = "\n".join([doc.page_content for doc in docs])

        # 3. Prompt for Gemini
        prompt = f"""You are a helpful medical assistant. Use the following context to answer the question:
        Context: {context}
        Question: {user_input}
        Answer:"""

        # 4. Call Gemini API with try-except block
        try:
            response = model.generate_content([{"role": "user", "parts": [prompt]}])
            print(f"\n💬 Gemini Response: {response.text}")
            return jsonify({'response': response.text})
        except Exception as e:
            print(f"\n❌ Gemini Error: {e}")
            return jsonify({'response': "Sorry, I couldn't get a response from Gemini."})

    except Exception as e:
        print(f"\n❌ Internal Error: {e}")
        return jsonify({'response': "An internal error occurred."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
