
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import hashlib
from agent import run_agent
from tools.rag_tool import add_pdf, get_indexed_files


app = FastAPI(title="LangGraph Agentic Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(BASE_DIR, "temp_docs")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)

 
    content = await file.read()

    content_hash = hashlib.sha256(content).hexdigest()
    if content_hash in get_indexed_files():
        return JSONResponse({
            "status": "skipped",
            "message": f"'{file.filename}' is already indexed."
        })

    with open(file_path, "wb") as f:
        f.write(content)

    print(" Saved at:", file_path)
    print(" Exists:", os.path.exists(file_path))

    try:
        was_indexed, message = add_pdf(file_path)
        status = "success" if was_indexed else "skipped"

        return JSONResponse({
            "status": status,
            "message": message
        })

    except Exception as e:
        print(" ERROR in add_pdf:", e)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/chat")
async def chat(
    query: str = Form(...),
    history: str = Form(default="[]")
):
    import json
    try:
        history_list = json.loads(history)
    except:
        history_list = []

    result = run_agent(query, history_list)

  
    safe_result = {}
    for key, value in result.items():
        if isinstance(value, str):
            safe_result[key] = value
        elif hasattr(value, "content"):         
            safe_result[key] = value.content
        else:
            safe_result[key] = str(value)        

    return JSONResponse(safe_result)

@app.get("/indexed_files")
async def indexed_files():
    return get_indexed_files()