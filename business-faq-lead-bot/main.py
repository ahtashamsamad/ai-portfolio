"""
Business FAQ + Lead-Capture Chatbot (FastAPI).

What it does:
  - Answers common customer questions by matching the visitor's message
    against an editable FAQ file (faqs.json) with TF-IDF similarity.
  - When it can't answer confidently, it politely asks for the visitor's
    contact details and saves the lead to leads.csv — so no lead is lost.

Endpoints:
  GET  /        -> service info
  GET  /faqs    -> all FAQs (loaded from faqs.json, editable without code changes)
  POST /chat    -> {"message": "..."} -> {"answer", "confidence", "ask_contact"}
  POST /lead    -> {"name", "contact", "question"} -> saved to leads.csv
  GET  /widget  -> minimal embeddable HTML/JS chat widget (talks to /chat, /lead)
  GET  /docs    -> interactive API docs (provided by FastAPI)

Run locally:
  uvicorn main:app --reload
"""

import csv
import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "faqs.json")
LEADS_PATH = os.path.join(BASE_DIR, "leads.csv")

# Below this similarity a question is treated as "not in the FAQ".
CONFIDENCE_THRESHOLD = 0.25

app = FastAPI(title="Business FAQ & Lead-Capture Chatbot", version="1.0.0")


def load_faqs():
    """FAQs live in faqs.json so the business owner can edit them as plain text."""
    with open(FAQ_PATH, encoding="utf-8") as f:
        return json.load(f)


FAQS = load_faqs()
_QUESTIONS = [item["question"] for item in FAQS]
_VECTORIZER = TfidfVectorizer(stop_words="english")
_FAQ_MATRIX = _VECTORIZER.fit_transform(_QUESTIONS)


class ChatRequest(BaseModel):
    message: str


class LeadRequest(BaseModel):
    name: str
    contact: str          # phone or email
    question: str = ""    # the question we couldn't answer (optional)


def best_faq_match(message):
    """Return (faq_item, confidence) for the closest FAQ question.

    Confidence is the TF-IDF cosine similarity in [0, 1].
    """
    if not message or not message.strip():
        return None, 0.0
    query_vec = _VECTORIZER.transform([message])
    scores = cosine_similarity(query_vec, _FAQ_MATRIX)[0]
    best_idx = int(scores.argmax())
    return FAQS[best_idx], float(scores[best_idx])


def save_lead(name, contact, question):
    """Append a lead row to leads.csv (creates the file with a header)."""
    new_file = not os.path.exists(LEADS_PATH)
    with open(LEADS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "name", "contact", "question"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            name.strip(),
            contact.strip(),
            question.strip(),
        ])


@app.get("/")
def index():
    return {
        "service": "Business FAQ & Lead-Capture Chatbot",
        "endpoints": ["/faqs", "/chat", "/lead", "/widget", "/docs"],
        "faqs_loaded": len(FAQS),
    }


@app.get("/faqs")
def list_faqs():
    return {"faqs": FAQS}


@app.post("/chat")
def chat(req: ChatRequest):
    faq, confidence = best_faq_match(req.message)
    if faq is None or confidence < CONFIDENCE_THRESHOLD:
        return {
            "answer": ("I don't have an answer for that yet — but I'd be happy "
                       "to have someone get back to you. Could you share your "
                       "name and phone/email?"),
            "confidence": round(confidence, 3),
            "ask_contact": True,
        }
    return {
        "answer": faq["answer"],
        "confidence": round(confidence, 3),
        "ask_contact": False,
    }


@app.post("/lead")
def capture_lead(req: LeadRequest):
    if not req.name.strip() or not req.contact.strip():
        raise HTTPException(status_code=422, detail="name and contact are required")
    save_lead(req.name, req.contact, req.question)
    return {
        "status": "saved",
        "message": f"Thanks {req.name.strip()}! We'll get back to you shortly.",
    }


# A minimal embeddable chat widget. It uses relative URLs ("/chat", "/lead")
# so it works wherever the app itself is hosted.
WIDGET_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat with us</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 420px; margin: 40px auto; }
  #log { border: 1px solid #ccc; border-radius: 8px; height: 320px; overflow-y: auto; padding: 10px; }
  .user { text-align: right; color: #0b5ed7; margin: 6px 0; }
  .bot { text-align: left; color: #333; margin: 6px 0; }
  #controls { display: flex; gap: 6px; margin-top: 10px; }
  #msg { flex: 1; padding: 8px; }
  button { padding: 8px 14px; }
  #leadform { display: none; margin-top: 10px; border-top: 1px dashed #aaa; padding-top: 10px; }
  #leadform input { display: block; width: 95%; margin: 6px 0; padding: 8px; }
</style>
</head>
<body>
<h3>💬 Chat with us</h3>
<div id="log"></div>
<div id="controls">
  <input id="msg" placeholder="Ask a question…" onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()">Send</button>
</div>
<div id="leadform">
  <p>Leave your details and we'll follow up:</p>
  <input id="lname" placeholder="Your name">
  <input id="lcontact" placeholder="Phone or email">
  <button onclick="saveLead()">Submit</button>
</div>
<script>
let lastQuestion = "";
function add(text, cls) {
  const d = document.createElement("div");
  d.className = cls; d.textContent = text;
  document.getElementById("log").appendChild(d);
}
async function send() {
  const box = document.getElementById("msg");
  const text = box.value.trim();
  if (!text) return;
  add(text, "user"); box.value = "";
  lastQuestion = text;
  const res = await fetch("/chat", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: text})
  });
  const data = await res.json();
  add(data.answer + " (confidence " + data.confidence + ")", "bot");
  document.getElementById("leadform").style.display = data.ask_contact ? "block" : "none";
}
async function saveLead() {
  const name = document.getElementById("lname").value.trim();
  const contact = document.getElementById("lcontact").value.trim();
  const res = await fetch("/lead", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name, contact, question: lastQuestion})
  });
  const data = await res.json();
  add(data.message || data.detail, "bot");
  document.getElementById("leadform").style.display = "none";
}
</script>
</body>
</html>
"""


@app.get("/widget", response_class=HTMLResponse)
def widget():
    return WIDGET_HTML
