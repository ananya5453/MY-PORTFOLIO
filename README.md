# Ananya S – AI/ML Portfolio

## 1. Start the backend (Windows PowerShell)
    cd backend
    pip install -r requirements.txt
    $env:EMBED_MODEL = "google/embeddinggemma-2"
    uvicorn main:app --port 8000
Mac/Linux: `EMBED_MODEL=google/embeddinggemma-2 uvicorn main:app --port 8000`
Check http://127.0.0.1:8000 (shows model + dimensions) or http://127.0.0.1:8000/docs.
If the model asks for access: accept its terms on Hugging Face and run `huggingface-cli login`.

## 2. Start the website (second terminal, project root)
    python -m http.server 5173
Open http://localhost:5173

## Edit your content
Everything is in the "EDIT HERE" block of the script in `index.html`
(email, LinkedIn, RAG projects, other projects, skills, certifications, API_URL).

## Optional: similarity-search step
Put `chunks.json` (a JSON list of text chunks) in `backend/` and restart the backend.

http://localhost:3000 