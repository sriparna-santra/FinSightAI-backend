from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.rag_service import process_and_store, ask_question
import google.generativeai as genai
import os

# ==============================
# 🔑 CONFIG
# ==============================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# ==============================
# 🌐 CORS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🧠 MEMORY STORE (TEMP)
# ==============================
user_data = {}

# ==============================
# 📩 REQUEST MODEL
# ==============================
class QuestionRequest(BaseModel):
    user_id: str
    question: str


# ==============================
# 📄 UPLOAD PDF
# ==============================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()

        vector_store = process_and_store(content)

        user_id = file.filename.replace(".pdf", "")
        user_data[user_id] = vector_store

        return {
            "message": "✅ PDF processed successfully",
            "user_id": user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 💬 ASK QUESTION
# ==============================
@app.post("/ask")
async def ask(req: QuestionRequest):
    try:
        if req.user_id not in user_data:
            raise HTTPException(status_code=400, detail="Upload PDF first")

        vector_store = user_data[req.user_id]

        answer = ask_question(vector_store, req.question)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 📊 INSIGHTS API
# ==============================
@app.post("/insights")
async def insights(req: QuestionRequest):
    try:
        if req.user_id not in user_data:
            raise HTTPException(status_code=400, detail="Upload PDF first")

        vector_store = user_data[req.user_id]

        docs = vector_store.similarity_search("revenue profit expenses", k=5)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Extract:
        Revenue, Profit, Expenses

        Return JSON:
        {{
          "revenue": number,
          "profit": number,
          "expenses": number
        }}

        Context:
        {context}
        """

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        res = model.generate_content(prompt).text

        import json
        try:
            parsed = json.loads(res)
        except:
            parsed = {"revenue": 0, "profit": 0, "expenses": 0}

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 🧪 HEALTH CHECK
# ==============================
@app.get("/")
def root():
    return {"status": "Backend running 🚀"}