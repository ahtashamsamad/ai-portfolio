Live demo: _(coming soon)_

# 💬 Business FAQ & Lead-Capture Chatbot

A FastAPI backend for a small-business website chatbot. It answers common
questions (hours, prices, delivery, refunds…) by matching the visitor's
message against an **editable FAQ file** — and when it can't answer
confidently, it asks for the visitor's name + contact and **saves the lead**
to `leads.csv`, so no lead is ever lost. Includes a minimal embeddable
HTML/JS chat widget.

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs · Widget: http://127.0.0.1:8000/widget

## Deploying (Render)

New Web Service → pick the repo → **Root Directory:** `business-faq-lead-bot`
→ **Build Command:** `pip install -r requirements.txt`
→ **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API

### `GET /`
Service info.

```bash
curl http://127.0.0.1:8000/
```
```json
{"service":"Business FAQ & Lead-Capture Chatbot","endpoints":["/faqs","/chat","/lead","/widget","/docs"],"faqs_loaded":12}
```

### `GET /faqs`
All FAQs (loaded from `faqs.json` — edit that file, no code changes needed).

```bash
curl http://127.0.0.1:8000/faqs
```

### `POST /chat`
Match a visitor message against the FAQs. Returns the answer plus a
confidence score (TF-IDF cosine similarity, 0–1). Below 0.25 the bot admits
it doesn't know and asks for contact details (`ask_contact: true`).

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your opening hours?"}'
```
```json
{"answer":"We're open Monday to Saturday, 8am to 8pm, and Sundays 9am to 2pm.","confidence":1.0,"ask_contact":false}
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Do you sell rocket ships?"}'
```
```json
{"answer":"I don't have an answer for that yet — but I'd be happy to have someone get back to you. Could you share your name and phone/email?","confidence":0.0,"ask_contact":true}
```

### `POST /lead`
Save a lead. Appends to `leads.csv` (created on first use).

```bash
curl -X POST http://127.0.0.1:8000/lead \
  -H "Content-Type: application/json" \
  -d '{"name": "Sara", "contact": "sara@example.com", "question": "Do you sell rocket ships?"}'
```
```json
{"status":"saved","message":"Thanks Sara! We'll get back to you shortly."}
```

### `GET /widget`
Embeddable chat page. It calls `/chat` as you type and, when the bot can't
answer, reveals a name/contact form that posts to `/lead`.
