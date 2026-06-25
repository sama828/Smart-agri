import os
import pickle
import threading
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# AraBERT
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
arabert_model = AutoModel.from_pretrained(MODEL_NAME)
arabert_model.eval()

# Models
with open("models/intent_classifier.pkl", "rb") as f:
    classifier_data = pickle.load(f)

with open("models/knowledge_base.pkl", "rb") as f:
    knowledge_base = pickle.load(f)

# FastAPI
app = FastAPI(title="Smart Agriculture Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    intent: str
    confidence: float
    language: str

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = arabert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def detect_language(text):
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > len(text) * 0.3 else "en"

def predict_intent(user_text, threshold=0.6):
    user_embedding = get_embedding(user_text).reshape(1, -1)
    similarities = cosine_similarity(
        user_embedding,
        classifier_data["embeddings"]
    )[0]

    top_idx = similarities.argmax()
    score = float(similarities[top_idx])
    intent = classifier_data["intents"][top_idx]

    if score < threshold:
        intent = "general"

    return intent, score

def retrieve_context(intent, lang):
    docs = knowledge_base.get(intent, knowledge_base["general"])
    context = []

    for doc in docs[:2]:
        content = doc["content_ar"] if lang == "ar" else doc["content_en"]
        context.append(f"- {doc['topic']}: {content}")

    return "\n".join(context)

def generate_response(user_text, intent, lang):
    context = retrieve_context(intent, lang)

    if lang == "ar":
        system = "أنت مساعد زراعي ذكي. اجب بالعربية بشكل محدد وعملي."
    else:
        system = "You are a smart agricultural assistant."

    prompt = f"""{system}

المعلومات المتاحة:
{context}

سؤال المستخدم:
{user_text}

الإجابة:
"""

    result = [None]

    def call_gemini():
        try:
            response = gemini_model.generate_content(prompt)
            result[0] = response.text
        except Exception:
            pass

    thread = threading.Thread(target=call_gemini)
    thread.start()
    thread.join(timeout=15)

    if result[0]:
        return result[0]

    docs = knowledge_base.get(intent, knowledge_base["general"])
    if docs:
        return docs[0]["content_ar"] if lang == "ar" else docs[0]["content_en"]

    return "حدث خطأ مؤقت."

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = detect_language(request.message)
    intent, confidence = predict_intent(request.message)
    response = generate_response(request.message, intent, lang)

    return ChatResponse(
        message=response,
        intent=intent,
        confidence=round(confidence, 3),
        language=lang
    )
