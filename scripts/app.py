import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .text_analysis import AI
from .speech_to_text import Whisper
from .utils import write_audio
from .video_manager import VideoDownloader

# Create instances of various services and utilities
downloader = VideoDownloader()  # For downloading YouTube videos
whisper_api = Whisper()          # For speech-to-text conversion
openai_services = AI()           # For AI-powered services

# Define metadata for API tags
tags_metadata = [
    {
        "name": "diarization_api",
        "description": "Operations with the diarization API. Click on the `Try it out` button to test the API with **YouTube video** or **audio file**.",
    },
]

# Initialize the FastAPI app with metadata and description
app = FastAPI(
    title="GPT-Diarization Project",
    description="Conversational Speaker Diarization using AI Language Models",
    version="0.1.0",
    openapi_tags=tags_metadata,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging settings for the application
logging.basicConfig(level=logging.INFO)


def process_audio(audio_data: bytes, filename: str, chunk_seconds: Optional[int]):
    """
    Process audio data and perform speaker diarization.

    :param audio_data: Audio data in bytes.
    :param filename: Name of the audio file.
    :param chunk_seconds: Optional backend chunk size in seconds. Use <= 0 to disable backend chunking when client-side chunking is already enabled.
    :return: Diarization result containing transcript and dialogue.
    """
    audio_output_path = write_audio(audio_data, filename)
    if chunk_seconds is not None and chunk_seconds <= 0:
        transcript = whisper_api.transcribe_single_pass(audio_output_path)
    else:
        transcript = whisper_api.transcribe_chunked(audio_output_path, chunk_seconds=chunk_seconds)

    dialogue = openai_services.extract_dialogue(transcript)
    result = {
        "transcript": transcript,
        "diarization_result": dialogue
    }
    return result


def process_youtube_video(video_id: str, chunk_seconds: Optional[int]):
    """
    Process a YouTube video and perform speaker diarization.

    :param video_id: ID of the YouTube video.
    :param chunk_seconds: Optional backend chunk size in seconds. Use <= 0 to disable backend chunking when client-side chunking is already enabled.
    :return: Diarization result containing transcript and dialogue.
    """
    audio_output_path = downloader.download_video(video_id)
    if chunk_seconds is not None and chunk_seconds <= 0:
        transcript = whisper_api.transcribe_single_pass(audio_output_path)
    else:
        transcript = whisper_api.transcribe_chunked(audio_output_path, chunk_seconds=chunk_seconds)

    dialogue = openai_services.extract_dialogue(transcript)
    result = {
        "transcript": transcript,
        "diarization_result": dialogue
    }
    return result


def parse_chunk_seconds(chunk_seconds: Optional[str]) -> int:
    """
    Parse and validate the optional chunk_seconds form value.

    Values <= 0 disable backend chunking.
    """
    if chunk_seconds is None:
        return whisper_api.default_chunk_seconds

    try:
        parsed_value = int(chunk_seconds)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid chunk_seconds: must be an integer (use <= 0 to disable backend chunking).",
        )

    if parsed_value <= 0:
        return parsed_value

    return whisper_api._clamp_chunk_seconds(parsed_value)


@app.post("/speaker-diarization", tags=["diarization_api"])
async def speaker_diarization(
    audio_file: UploadFile = File(None),
    youtube_video_id: Optional[str] = Form(None),
    chunk_seconds: Optional[str] = Form(None),
):
    """
    Endpoint to perform speaker diarization on audio file or YouTube video.

    :param audio_file: Uploaded audio file (if provided).
    :param youtube_video_id: ID of the YouTube video (if provided).
    :param chunk_seconds: Optional backend chunk size in seconds. Use <= 0 to disable backend chunking when client-side chunking is already enabled.
    :return: Diarization result containing transcript and dialogue.
    """
    try:
        chosen_chunk_seconds = parse_chunk_seconds(chunk_seconds)
        logging.info("/speaker-diarization chunk_seconds=%s", chosen_chunk_seconds)

        if audio_file is not None:
            audio_data = await audio_file.read()
            filename = audio_file.filename
            response = process_audio(audio_data, filename, chunk_seconds=chosen_chunk_seconds)
            return response

        if youtube_video_id:
            response = process_youtube_video(youtube_video_id, chunk_seconds=chosen_chunk_seconds)
            return response

        raise HTTPException(status_code=400, detail="Provide either audio_file or youtube_video_id")
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"/speaker-diarization:/500, {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
