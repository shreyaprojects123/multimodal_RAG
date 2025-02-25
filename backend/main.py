from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import fitz  # PyMuPDF for PDFs
import whisper
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Load variables from .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define file storage and model directories
UPLOAD_DIR = "data/uploads"
MODEL_DIR = "models"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load AI models
whisper_model = whisper.load_model("base")  # Audio transcription
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Text embeddings
index = faiss.IndexFlatL2(384)  # FAISS vector database for storing text embeddings
text_data = []  # Store extracted text

# Function to extract text from a PDF
def process_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return ""

# Function to transcribe audio/video files
def process_audio(file_path):
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        return ""

# Function to process text files
def process_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
        return ""

# API endpoint to upload multiple files
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global text_data, index  # Ensure text_data is accessible
    responses = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
            continue

        # Process the file based on type
        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            extracted_text = process_pdf(file_path)
        elif file.filename.lower().endswith((".mp3", ".mp4", ".wav", ".m4a")):
            extracted_text = process_audio(file_path)
        elif file.filename.lower().endswith((".txt", ".html", ".md")):
            extracted_text = process_text(file_path)
        else:
            # Handle unknown file types
            extracted_text = f"File type not supported for {file.filename}"

        # Debug message
        print(f"Extracted Text from {file.filename}: {extracted_text[:200]}...")  # Print first 200 chars

        # If text is extracted, store it
        if extracted_text and extracted_text.strip():
            text_data.append(extracted_text)
            try:
                embedding = embedding_model.encode(extracted_text)
                embedding = np.array([embedding]).astype('float32')
                index.add(embedding)
                responses.append({"file": file.filename, "message": "Successfully uploaded and processed"})
            except Exception as e:
                print(f"Error encoding text from {file.filename}: {e}")
                responses.append({"file": file.filename, "message": "Uploaded but processing failed"})
        else:
            responses.append({"file": file.filename, "message": "Uploaded but no text extracted"})

    # Debug message
    print(f"Total documents stored: {len(text_data)}")

    return {"files_processed": responses}

# Question request model
class QuestionRequest(BaseModel):
    question: str

# API endpoint to ask questions
@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    question = request.question 
    
    if not text_data:
        print("❌ No data available. Upload files first!")
        return {"error": "No data available. Please upload files first."}

    print(f"✅ User asked: {question}")  # Debug log
    
    try:
        question_embedding = embedding_model.encode(question)
        question_embedding = np.array([question_embedding]).astype('float32')

        _, closest_index = index.search(question_embedding, 1)
        best_match_index = closest_index[0][0]

        if best_match_index < 0 or best_match_index >= len(text_data):
            print("❌ No relevant document found!")
            return {"error": "No relevant document found."}

        best_match_text = text_data[best_match_index]

        print(f"✅ Best match text: {best_match_text[:200]}...")  # Debug log

        # Use Gemini with instruction to format with bullet points
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            "Based on this document, answer the following question. When appropriate, format your answer with bullet points. Keep your response concise and focused.\n\n"
            f"Question: {question}\n\n"
            f"Reference:\n{best_match_text}"
        )

        return {"answer": response.text}
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"error": f"An error occurred: {str(e)}"}

# Root endpoint for testing server status
@app.get("/")
def read_root():
    return {"status": "Server is running", "endpoints": ["/upload/", "/ask/"]}