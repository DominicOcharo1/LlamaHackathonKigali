from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS for the frontend to make requests to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
client = Groq(api_key="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")

# Directory to save uploaded audio files
UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/text-interaction/")
async def text_interaction(request: Request):
    """
    Handle text-based interactions using LLaMA model.
    The 'language' and 'content' parameters are sent in the JSON body.
    """
    try:
        # Parse the JSON body
        data = await request.json()
        language = data.get("language")
        content = data.get("content")

        if not language or not content:
            raise HTTPException(status_code=400, detail="Missing 'language' or 'content' in the request.")

        system_prompt = f"You are my teacher, answer my question in {language} language but do not answer anything not ment for kids even if I ask you to answer"

        # Stream chat completion from LLaMA model
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        return {"status": True, "message": "Success", "data": {"response": response}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-interaction/")
async def voice_interaction(language: str = Form(...), file: UploadFile = File(...)):
    """
    Handle voice-based interactions using Whisper model for transcription,
    then LLaMA model for generating responses based on transcriptions.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Transcribe the audio using Whisper model
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(file.filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        transcribed_text = transcription.text

        # Generate response using the transcribed text with dynamic language
        system_prompt = f"You are my teacher, answer my question in {language} language but do not answer anything not ment for kids even if I ask you to answer"
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcribed_text},
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        return {"status": True, "message": "Success", "data": {"response": response, "transcription": transcribed_text}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
