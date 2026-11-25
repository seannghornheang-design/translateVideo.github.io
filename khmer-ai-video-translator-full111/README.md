# Khmer AI Video Translator (Frontend + Backend)

## Project structure
- `static/index.html`: frontend (HTML/CSS/JavaScript)
- `main.py`: FastAPI backend and `/api/translate` endpoint
- `requirements.txt`: Python packages
- `render.yaml`: Render deployment configuration

## How the frontend connects to the backend
The browser JavaScript sends the selected video to `POST /api/translate` using `fetch()` and `FormData`. FastAPI receives it, runs speech transcription, Khmer translation, Khmer text-to-speech and FFmpeg, then returns a download URL. Because the frontend is served by the same FastAPI app, no separate API URL or CORS setup is needed.

## Required environment variables
Set these in your hosting provider's Environment settings (do not commit API keys):
- `OPENAI_API_KEY`
- `AZURE_SPEECH_KEY`
- `AZURE_SPEECH_REGION`

Azure Khmer voices:
- Male: `km-KH-PisethNeural`
- Female: `km-KH-SreymomNeural`

## Deploy on Render
1. Push this folder's contents to a GitHub repository.
2. In Render, create a new Web Service and connect that repository.
3. Render can use `render.yaml`; set the three environment variables in the service settings.
4. Deploy and open the generated `onrender.com` URL.

## Important limitations
- This starter currently accepts uploaded videos, not YouTube links.
- Select one Khmer voice for the whole video. Automatic male/female speaker diarization for videos with multiple speakers is not implemented yet.
- This is a starter project. Long videos may exceed free hosting time, memory, or request limits. Test with a short clip first.
- Use only videos you own or have permission to process, and comply with platform terms.
- API usage may incur charges.
