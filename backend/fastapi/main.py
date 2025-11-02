from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import yt_dlp
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import uuid
from datetime import datetime
import base64
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG + Video Verification API v2 (OpenAI)",
    description="Combined API for video verification and multimodal RAG with PDFs, images, and audio - All powered by OpenAI",
    version="2.0.1" # Incremented version for fixes
)

origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"]
)


# Initialize OpenAI client
try:
    openai_client = OpenAI()
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    openai_client = None

# Configuration
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# In-memory storage for vector stores (session-based)
vector_stores = {}

# ---
# FIXED: Simplified Verifier Prompt
# Removed references to web search capabilities that were not implemented
# and would cause errors. Forced webSearchUsed to false.
# ---
VERIFIER_SYSTEM_PROMPT = """
You are a fact-checker assistant. Your name is 'Verifier'.
Your job is to verify the content provided (which is a transcription of a video)
and determine if the claims made are factually correct based on your internal knowledge.

You must check if the content is scientifically, historically, or otherwise factually correct.
First, determine if the content contains a verifiable claim or if it is just
entertainment, opinion, or casual conversation.

Rule:
- You MUST strictly follow the JSON output format.
- The system is in JSON mode, so your entire response must be a single, valid JSON object.

Output format:
{
  "isFactualClaim": boolean,
  "isContentCorrect": "Yes" | "No" | "Half" | "N/A",
  "reason": "string",
  "webSearchUsed": false
}

Guidelines for fields:
- "isFactualClaim": true if the text makes a specific claim that can be verified.
- "isContentCorrect":
    - "Yes": If the central claim is factually correct.
    - "No": If the central claim is factually incorrect.
    - "Half": If the claim is partially correct, misleading, or lacks critical context.
    - "N/A": If "isFactualClaim" is false (e.g., it's an opinion, joke, or greeting).
- "reason": Explain your reasoning. If "isFactualClaim" is false, state that.
- "webSearchUsed": Always set this to false.
"""

# Pydantic models for request/response
class VideoURLRequest(BaseModel):
    url: HttpUrl
    
class SessionRequest(BaseModel):
    session_id: str

class VerificationRequest(BaseModel):
    content: str

class ChatRequest(BaseModel):
    session_id: str
    question: str
    k: int = 4  # Number of documents to retrieve

# ---
# REMOVED: Unused Pydantic models
# - TranscriptionRequest
# - TranslationRequest
# - VerificationRequest
# ---

class VerificationResult(BaseModel):
    isFactualClaim: bool
    isContentCorrect: str
    reason: str
    webSearchUsed: bool = False

class FullPipelineRequest(BaseModel):
    url: HttpUrl
    keep_audio: bool = False


# ============================================
# OPENAI UTILITY FUNCTIONS
# ============================================

def get_image_description_openai(image_bytes, filename):
    """Uses OpenAI Vision API to describe an image and perform OCR."""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image named '{filename}'. Describe its content in detail, extract any text (OCR), and summarize the key information. This will be used as document content for search and retrieval."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        description = response.choices[0].message.content
        return description
        
    except Exception as e:
        print(f"Error in image description: {e}")
        return None


def transcribe_audio_openai(audio_path: str) -> Optional[str]:
    """Transcribes the audio file using OpenAI Whisper."""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


# ============================================
# VIDEO VERIFICATION UTILITY FUNCTIONS
# ============================================

def sanitize_filename(filename: str) -> str:
    """Removes characters that are illegal in Windows/Linux/macOS filenames."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = re.sub(r'\.+', '.', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip(' .')
    return sanitized[:200]


def download_audio_from_url(video_url: str) -> Optional[str]:
    """Downloads the best audio from a given URL and converts it to MP3."""
    try:
        info_opts = {
            'quiet': True,
            'noplaylist': True,
            'simulate': True,
        }
        with yt_dlp.YoutubeDL(info_opts) as ydl_info:
            info_dict = ydl_info.extract_info(video_url, download=False)
            title = info_dict.get('title', 'downloaded_audio')
            sanitized_title = sanitize_filename(title)
            
            unique_id = str(uuid.uuid4())[:8]
            final_filename_base = f"{sanitized_title}_{unique_id}"
            final_filepath_mp3 = f"{final_filename_base}.mp3"

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{final_filename_base}.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            'noprogress': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_url])
            
            if error_code == 0:
                full_path = os.path.abspath(final_filepath_mp3)
                return full_path
            else:
                return None

    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# ---
# REMOVED: Redundant `transcribe_audio` wrapper function.
# Endpoints will now call `transcribe_audio_openai` directly.
# ---

def translate_to_english(text_to_translate: str) -> Optional[str]:
    """Translates the given text to English using OpenAI."""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        translation = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a translation assistant that accurately converts any language to English."},
                {"role": "user", "content": f"Translate the following text into English:\n\n{text_to_translate}"}
            ]
        )
        english_text = translation.choices[0].message.content
        return english_text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

# ---
# FIXED: `verify_content` function
# Removed the broken `tools` parameter and the complex `if/else` logic
# for handling tool calls. This function now makes one simple call.
# ---
def verify_content(content_text: str) -> Optional[dict]:
    """
    Uses the enhanced prompt to fact-check the content using internal knowledge.
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": content_text}
            ]
        )
        
        json_string = response.choices[0].message.content
        result = json.loads(json_string)
        
        # Ensure 'webSearchUsed' exists, although prompt forces it to false
        if "webSearchUsed" not in result:
             result["webSearchUsed"] = False
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Verification error: {e}")
        return None


def cleanup_file(file_path: str):
    """Helper function to clean up downloaded files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Could not delete file {file_path}: {e}")


# ============================================
# RAG DOCUMENT PROCESSING FUNCTIONS
# ============================================

def process_files_for_rag(
    pdf_files: Optional[List[UploadFile]] = None,
    image_files: Optional[List[UploadFile]] = None,
    audio_file: Optional[UploadFile] = None,
    audio_transcript: str = None
) -> Optional[FAISS]:
    """
    Processes uploaded files and creates a FAISS vector store using OpenAI embeddings.
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process audio transcript
    if audio_transcript:
        audio_filename = audio_file.filename if audio_file else "audio_transcript"
        audio_doc = Document(
            page_content=f"Transcript of {audio_filename}: {audio_transcript}",
            metadata={
                'source': audio_filename,
                'chunk_index': 0,
                'page_label': 'Transcript'
            }
        )
        all_chunks.append(audio_doc)
        print(f"Added audio transcript: {audio_filename}")
    
    # Process PDF files
    if pdf_files:
        for pdf_file in pdf_files:
            try:
                # Need to reset file pointer just in case
                pdf_file.file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(pdf_file.file.read())
                    temp_pdf_path = temp_pdf.name
                
                loader = PyPDFLoader(file_path=temp_pdf_path)
                docs = loader.load()
                
                chunks = text_splitter.split_documents(documents=docs)
                for i, chunk in enumerate(chunks):
                    chunk.metadata['source'] = pdf_file.filename
                    chunk.metadata['chunk_index'] = i
                    # PyPDFLoader adds 'page' to metadata
                    chunk.metadata['page_label'] = f"Page {chunk.metadata.get('page', 'N/A')}"
                
                all_chunks.extend(chunks)
                print(f"Processed PDF: {pdf_file.filename} - {len(chunks)} chunks")
                os.remove(temp_pdf_path)
                
            except Exception as e:
                print(f"Error processing PDF {pdf_file.filename}: {e}")
    
    # Process image files using OpenAI Vision
    if image_files:
        for image_file in image_files:
            try:
                # Need to reset file pointer
                image_file.file.seek(0)
                image_bytes = image_file.file.read()
                
                description = get_image_description_openai(
                    image_bytes,
                    image_file.filename
                )
                
                if description:
                    image_doc = Document(
                        page_content=description,
                        metadata={
                            'source': image_file.filename,
                            'chunk_index': 0,
                            'page_label': 'Image Description'
                        }
                    )
                    all_chunks.append(image_doc)
                    print(f"Processed image: {image_file.filename}")
                    
            except Exception as e:
                print(f"Error processing image {image_file.filename}: {e}")
    
    # Create vector store with OpenAI embeddings
    if all_chunks:
        try:
            # Use OpenAI embeddings
            embedding_model = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL
            )
            
            vector_store = FAISS.from_documents(
                documents=all_chunks,
                embedding=embedding_model
            )
            
            print(f"Vector store created with {len(all_chunks)} chunks")
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None
    
    return None


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal RAG + Video Verification API v2 (OpenAI Powered)",
        "version": "2.0.1",
        "ai_provider": "OpenAI",
        "models": {
            "chat": OPENAI_CHAT_MODEL,
            "embeddings": OPENAI_EMBEDDING_MODEL,
            "transcription": "whisper-1",
            "vision": "gpt-4o"
        },
        "endpoints": {
            "video_verification": {
                "POST /verify-video": "Simple: URL -> Verification result",
                "POST /full-pipeline": "Detailed: URL -> All steps + verification"
            },
            "rag_system": {
                "POST /create-rag-session": "Create RAG session with documents",
                "POST /transcribe-audio": "Transcribe audio using Whisper",
                "POST /chat": "Ask questions about indexed documents",
                "DELETE /delete-session": "Delete RAG session",
                "GET /sessions": "List all active sessions"
            }
        }
    }


# ============================================
# TEXT VERIFICATION ENDPOINTS
# ============================================


@app.post("/verify")
async def verify(request: VerificationRequest):
    """Verify content for factual accuracy."""
    if not request.content or len(request.content.strip()) < 5:
        raise HTTPException(status_code=400, detail="Content is too short or empty")
    
    verification_result = verify_content(request.content)
    
    if not verification_result:
        raise HTTPException(status_code=500, detail="Verification failed")
    
    return {
        "success": True,
        "verification": verification_result
    }

# ============================================
# VIDEO VERIFICATION ENDPOINTS
# ============================================

@app.post("/verify-video")
async def verify_video(request: VideoURLRequest, background_tasks: BackgroundTasks):
    """
    Single endpoint to verify video content.
    Downloads audio, transcribes, translates, and verifies in one call.
    Returns only the verification result.
    """
    audio_path = None
    try:
        video_url = str(request.url)
        cleaned_url = video_url.split('?')[0].strip()
        
        audio_path = download_audio_from_url(cleaned_url)
        if not audio_path:
            raise HTTPException(status_code=400, detail="Failed to download audio from URL")
        
        # FIXED: Call transcribe_audio_openai directly
        transcription = transcribe_audio_openai(audio_path)
        if not transcription:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        if len(transcription.strip()) < 5:
            raise HTTPException(status_code=400, detail="Transcription is too short to process")
        
        english_text = translate_to_english(transcription)
        if not english_text:
            raise HTTPException(status_code=500, detail="Translation failed")
        
        verification_result = verify_content(english_text)
        if not verification_result:
            raise HTTPException(status_code=500, detail="Verification failed")
        
        return {
            "success": True,
            "url": str(request.url),
            "verification": verification_result,
            "transcript": english_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if audio_path:
            background_tasks.add_task(cleanup_file, audio_path)


# ---
# FIXED: Refactored /full-pipeline to use a clean `finally` block
# for all cleanup logic, just like /verify-video.
# ---
@app.post("/full-pipeline")
async def full_pipeline(request: FullPipelineRequest, background_tasks: BackgroundTasks):
    """
    Complete pipeline: download audio, transcribe, translate, and verify content.
    Returns detailed information from each step.
    """
    audio_path = None
    try:
        video_url = str(request.url)
        cleaned_url = video_url.split('?')[0].strip()
        
        audio_path = download_audio_from_url(cleaned_url)
        if not audio_path:
            raise HTTPException(status_code=400, detail="Failed to download audio")
        
        # FIXED: Call transcribe_audio_openai directly
        transcription = transcribe_audio_openai(audio_path)
        if not transcription:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        if len(transcription.strip()) < 5:
            raise HTTPException(status_code=400, detail="Transcription is too short")
        
        english_text = translate_to_english(transcription)
        if not english_text:
            raise HTTPException(status_code=500, detail="Translation failed")
        
        verification_result = verify_content(english_text)
        if not verification_result:
            raise HTTPException(status_code=500, detail="Verification failed")
        
        return {
            "success": True,
            "url": str(request.url),
            "audio_path": audio_path if request.keep_audio else "deleted",
            "transcription": transcription,
            "translated_text": english_text,
            "verification": verification_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # This cleanup logic now runs regardless of success or failure
        if audio_path and not request.keep_audio:
            background_tasks.add_task(cleanup_file, audio_path)


# ============================================
# RAG SYSTEM ENDPOINTS
# ============================================

# ---
# FIXED: Corrected typing for optional file uploads
# ---
@app.post("/create-rag-session")
async def create_rag_session(
    pdf_files: Optional[List[UploadFile]] = File(None),
    image_files: Optional[List[UploadFile]] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Create a new RAG session by uploading PDFs, images, and/or audio.
    Returns a session_id to use for chatting.
    All processing uses OpenAI (Whisper for audio, GPT-4o Vision for images).
    """
    try:
        session_id = str(uuid.uuid4())
        audio_transcript = None
        
        # Transcribe audio if provided using OpenAI Whisper
        if audio_file:
            audio_bytes = await audio_file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            audio_transcript = transcribe_audio_openai(temp_audio_path)
            os.remove(temp_audio_path)
            
            if not audio_transcript:
                raise HTTPException(status_code=500, detail="Audio transcription failed")
        
        # Process all files and create vector store
        vector_store = process_files_for_rag(
            pdf_files=pdf_files,
            image_files=image_files,
            audio_file=audio_file,
            audio_transcript=audio_transcript
        )
        
        if not vector_store:
            # Check if any files were provided at all
            if not pdf_files and not image_files and not audio_file:
                raise HTTPException(status_code=400, detail="No files provided. Please upload at least one PDF, image, or audio file.")
            raise HTTPException(status_code=400, detail="No valid documents were processed from the uploaded files.")
        
        # Store vector store in memory
        vector_stores[session_id] = {
            "vector_store": vector_store,
            "created_at": datetime.now().isoformat(),
            "audio_transcript": audio_transcript
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "RAG session created successfully (OpenAI powered)",
            "audio_transcribed": audio_transcript is not None,
            "pdf_files_processed": len(pdf_files) if pdf_files else 0,
            "image_files_processed": len(image_files) if image_files else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating RAG session: {str(e)}")


@app.post("/transcribe-audio")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    Transcribe audio using OpenAI Whisper.
    Returns the transcript text.
    """
    try:
        audio_bytes = await audio_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        transcript = transcribe_audio_openai(temp_audio_path)
        os.remove(temp_audio_path)
        
        if not transcript:
            raise HTTPException(status_code=500, detail="Transcription failed")
        
        return {
            "success": True,
            "filename": audio_file.filename,
            "transcript": transcript,
            "model": "whisper-1"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Ask a question about the indexed documents in a RAG session.
    Uses OpenAI for both retrieval and generation.
    """
    try:
        if request.session_id not in vector_stores:
            raise HTTPException(status_code=4404, detail="Session not found. Please create a RAG session first.")
        
        session_data = vector_stores[request.session_id]
        vector_store = session_data["vector_store"]
        
        # Retrieve relevant documents
        search_results = vector_store.similarity_search(query=request.question, k=request.k)
        
        context_list = []
        for result in search_results:
            context_list.append(
                f"Page Content: {result.page_content.strip()}\n"
                f"Source: {result.metadata.get('source', 'Unknown File')}\n"
                f"Location: {result.metadata.get('page_label', 'N/A')}"
            )

        context = "\n\n---\n\n".join(context_list)
        
        # Create system prompt with context
        system_prompt = f"""
        You are a helpful AI assistant. Answer the user's question based ONLY on the provided context, which is retrieved from uploaded files and audio transcripts.
        
        You MUST cite the source information (Source, Location) for every part of your answer. If the context does not contain the answer, state clearly that the information is not available in the documents.
        
        Context:
        {context}
        """
        
        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ]
        )
        
        answer = response.choices[0].message.content
        
        return {
            "success": True,
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "sources_count": len(search_results),
            "model": OPENAI_CHAT_MODEL,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


@app.delete("/delete-session")
async def delete_session(request: SessionRequest):
    """Delete a RAG session and free up memory."""
    try:
        if request.session_id not in vector_stores:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del vector_stores[request.session_id]
        
        return {
            "success": True,
            "message": f"Session {request.session_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active RAG sessions."""
    sessions = []
    for session_id, data in vector_stores.items():
        sessions.append({
            "session_id": session_id,
            "created_at": data["created_at"],
            "has_audio_transcript": data["audio_transcript"] is not None
        })
    
    return {
        "success": True,
        "active_sessions": len(sessions),
        "sessions": sessions
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    openai_status = "ready" if openai_client else "not initialized"
    
    return {
        "status": "healthy",
        "ai_provider": "OpenAI",
        "openai_client": openai_status,
        "models": {
            "chat": OPENAI_CHAT_MODEL,
            "embeddings": OPENAI_EMBEDDING_MODEL,
            "transcription": "whisper-1",
            "vision": "gpt-4o"
        },
        "active_rag_sessions": len(vector_stores),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
