import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

app = FastAPI()

# 1. Security: Allow your website (Silverback) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to "https://silverbackweb.com" in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configure AI Clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    history: list = []

class ImageRequest(BaseModel):
    prompt: str

# --- Endpoints ---

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Handles the conversation logic.
    """
    try:
        # Construct the conversation history for context
        messages = [
            {"role": "system", "content": "You are a helpful Deck Design Consultant for Silverback Web Design. concise, professional, and focus on gathering requirements for a deck (size, material, budget)."}
        ]
        # Append past history (simplified for this demo)
        for msg in req.history:
             messages.append(msg)
        
        messages.append({"role": "user", "content": req.message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize")
async def visualize_endpoint(req: ImageRequest):
    """
    Uses Google's 'Nano Banana' (Imagen/Gemini) to generate the deck image.
    """
    try:
        # Using the Imagen 3 model (often accessed via the Gemini API endpoints now)
        # Note: Exact model ID changes rapidly; 'imagen-3.0-generate-001' is current standard
        model = genai.GenerativeModel("imagen-3.0-generate-001") 
        
        enhanced_prompt = (
            f"Photorealistic architectural render of a residential deck: {req.prompt}. "
            "Sunny day, suburban backyard, 8k resolution, highly detailed materials."
        )

        # Generate the image
        result = model.generate_images(
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio="16:9"
        )

        # Retrieve the image (Google SDK usually returns a PIL image or bytes)
        # We need to convert it to base64 to send to the frontend
        img_bytes = result.images[0].image_bytes
        b64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        return {"image_data": f"data:image/jpeg;base64,{b64_string}"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

# 3. Serve the Frontend (The Iframe Content)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
