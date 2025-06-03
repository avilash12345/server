from fastapi import FastAPI, Query, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse # type: ignore
from pytubefix import YouTube # type: ignore
import os

app = FastAPI()

# Setup CORS

app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # For production, set this to your frontend URL
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "YouTube Downloader API with FastAPI and pytubefix"}

@app.get("/download/video")
def download_video(url: str = Query(..., description="YouTube video URL")):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        output_path = os.path.join(DOWNLOAD_DIR, f"{yt.title}.mp4")
        stream.download(output_path=DOWNLOAD_DIR, filename=f"{yt.title}.mp4")
        return FileResponse(output_path, media_type="video/mp4", filename=f"{yt.title}.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/audio")
def download_audio(url: str = Query(..., description="YouTube video URL")):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        output_path = os.path.join(DOWNLOAD_DIR, f"{yt.title}.mp3")
        temp_file = stream.download(output_path=DOWNLOAD_DIR)
        os.rename(temp_file, output_path)
        return FileResponse(output_path, media_type="audio/mpeg", filename=f"{yt.title}.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# http://127.0.0.1:8000/download/audio?url=https%3A%2F%2Fyoutu.be%2FKnzmnTagueM%3Fsi%3DHVDuAAoJwrW8zMR0

# http://127.0.0.1:8000/download/audio?url=https%3A%2F%2Fyoutu.be%2FKnzmnTagueM%3Fsi%3DHVDuAAoJwrW8zMR0