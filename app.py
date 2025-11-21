import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Initialize LangChain Groq client
# Model specified by user: llama-3.1-8b-instant
llm = ChatGroq(
    temperature=0.7, 
    groq_api_key=GROQ_API_KEY, 
    model_name="llama-3.1-8b-instant"
)

def search_google_images(query):
    """
    Search for images using Google Custom Search API.
    Returns a list of image URLs (up to 2).
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("Google API Key or CSE ID not set.")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 2
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        image_urls = []
        if "items" in data:
            for item in data["items"]:
                image_urls.append(item["link"])
        return image_urls
    except Exception as e:
        print(f"Error searching images: {e}")
    return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    history_data = data.get("history", [])

    messages = []
    
    # System Prompt
    messages.append(SystemMessage(content=(
        "You are a helpful and engaging AI assistant. "
        "You can send images to the user to help visualize concepts or when explicitly asked. "
        "To send an image, include the tag [IMAGE: <search_query>] at the end of your response. "
        "For example: 'Here is a picture of a golden retriever. [IMAGE: golden retriever dog]'. "
        "Do not use the tag if an image is not relevant or necessary."
    )))

    for msg in history_data:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    
    messages.append(HumanMessage(content=user_input))

    try:
        ai_response = llm.invoke(messages)
        ai_response_content = ai_response.content
        
        image_urls = []
        match = re.search(r"\[IMAGE:\s*(.*?)\]", ai_response_content, re.IGNORECASE)
        
        clean_response = ai_response_content
        if match:
            query = match.group(1)
            print(f"Detected image request for: {query}")
            image_urls = search_google_images(query)
            clean_response = ai_response_content.replace(match.group(0), "").strip()

        return jsonify({
            "response": clean_response,
            "image_urls": image_urls,
            "assistant_message": {"role": "assistant", "content": ai_response_content}
        })

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
