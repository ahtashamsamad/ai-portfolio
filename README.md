# AI Portfolio — Ahtasham

Four client-demoable AI projects in Python: a document Q&A chatbot, a business
FAQ + lead-capture chatbot API, a review sentiment dashboard, and an
ATS-busting resume screener for HR.

**Live demos**
- 📄 [Chat with your PDF](https://ai-portfolio-jwev4baznf4cjwf5hv6wdx.streamlit.app/)
- 💬 [Business FAQ & lead-capture chatbot](https://business-faq-lead-bot.onrender.com) ([chat widget](https://business-faq-lead-bot.onrender.com/widget))
- 📊 [Review sentiment dashboard](https://ai-portfolio-flhuxakr6chyv53opzkcbx.streamlit.app/)
- 📋 [ATS resume screener](https://ai-portfolio-et3o8xwtmckpkuk2rynywq.streamlit.app/)

| Project | What it does | Stack |
|---|---|---|
| [pdf-rag-chatbot](pdf-rag-chatbot/) | Upload any PDF, ask questions in plain English, see the source passages (with page numbers) behind every answer. Optional OpenAI mode for natural answers. | Streamlit, scikit-learn, pypdf, OpenAI |
| [business-faq-lead-bot](business-faq-lead-bot/) | FAQ chatbot API for business websites: answers from an editable FAQ file, captures name + contact as a lead when it can't answer, embeddable chat widget. | FastAPI, scikit-learn |
| [review-sentiment-analyzer](review-sentiment-analyzer/) | Classifies customer reviews as positive/negative — single review or batch CSV upload — with distribution charts, top complaint/praise words, and downloadable results. | Streamlit, scikit-learn, pandas, matplotlib |
| [ats-resume-screener](ats-resume-screener/) | Ranks resumes against a job description by semantic similarity + skill overlap (not just keywords), with every score explained: matched/missing skills and a fit summary per candidate. | Streamlit, scikit-learn, pypdf, pandas |

Each folder has its own README with setup, run commands, and a live demo link.
