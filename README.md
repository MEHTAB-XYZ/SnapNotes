# Smart Notes from Screenshots

A web application that helps students generate clean, readable notes from screenshots of lecture slides, handwritten notes, or YouTube video snapshots.

## Features

- Upload screenshots (PNG, JPG, JPEG)
- Extract text using OCR
- Generate detailed, formatted notes using Google's Gemini AI
- Export notes to PDF

## Project Structure

```
├── frontend/         # React frontend
└── backend/          # FastAPI backend
```

## Prerequisites

- Node.js and npm
- Python 3.8+
- Tesseract OCR installed on your system
- Google Gemini API key

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:

   ```
   cd backend
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt   ```



3. Create a `.env` file in the backend directory and add your Gemini API key:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Start the backend server:
   ```
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```
   cd frontend
   ```

2. Install the required npm packages:

   ```
   npm install
   ```

3. Start the development server:

   ```
   npm run dev
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:5173)

## Usage

1. Upload a screenshot of lecture slides, handwritten notes, or a YouTube video
2. Enter the topic or subject you're studying
3. Click "Generate Notes"
4. View the AI-generated notes
5. Download the notes as a PDF if desired

## Technologies Used

- Frontend: React, Tailwind CSS, react-to-pdf
- Backend: FastAPI, pytesseract, Google Generative AI (Gemini Flash)
