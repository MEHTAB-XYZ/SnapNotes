from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import io
import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI(title="Smart Notes API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to Smart Notes API"}


def extract_text_from_image(image_bytes):
    try:
        # Initialize the EasyOCR reader (only need to do this once)
        reader = easyocr.Reader(['en'])
        
        # Open the image and convert to numpy array for EasyOCR
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Perform OCR using EasyOCR
        results = reader.readtext(image_np)
        
        # Extract and combine the text from results
        extracted_text = ' '.join([text[1] for text in results])
        
        if not extracted_text or extracted_text.isspace():
            raise HTTPException(status_code=400, detail="No text could be extracted from the image")
            
        return extracted_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


def generate_notes(topic, extracted_text):
    try:
        prompt = f"""You are a helpful academic assistant. A student is studying the topic: "{topic}". 
        The extracted notes from their screenshot are:
        "{extracted_text}"

        Based on both, generate detailed and well-formatted notes suitable for exam revision.
        
        Respond directly with clean HTML without any code block markers. Use:
        - <h1>, <h2>, <h3> tags for headings and subheadings
        - <ul> and <li> for bullet points
        - <ol> and <li> for numbered lists
        - <strong> or <b> for important terms
        - <em> or <i> for emphasis
        - <p> tags for paragraphs
        - Proper spacing and indentation for readability
        - Clear sections and subsections
        - Highlight key concepts and definitions
        
        Make sure the notes are comprehensive, well-organized, and visually structured."""
        
        response = model.generate_content(prompt)
        # Clean any potential code block markers from the response
        cleaned_response = response.text.replace('```html', '').replace('```', '').strip()
        return cleaned_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.post("/generate-notes")
async def process_image(file: UploadFile = File(...), topic: str = Form(...)):
    # Check file size (5MB limit)
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > 5 * 1024 * 1024:  # 5MB in bytes
        return JSONResponse(
            status_code=400,
            content={"error": "File size exceeds the 5MB limit"}
        )
    
    # Check file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPEG, JPG and PNG files are supported"}
        )
    
    try:
        # Extract text using OCR
        extracted_text = extract_text_from_image(file_content)
        
        if not extracted_text or extracted_text.isspace():
            return JSONResponse(
                status_code=400,
                content={"error": "No text could be extracted from the image"}
            )
        
        # Generate notes using Gemini API
        notes = generate_notes(topic, extracted_text)
        
        return {"notes": notes}
    
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)