# Import dependencies
from fastapi import APIRouter, Body, Header
from dotenv import load_dotenv, find_dotenv
import os
import base64
import tempfile
from pydub import AudioSegment
from io import BytesIO

from logger import logger

# Create a router object
router = APIRouter()

# Loading env variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


@router.post("/join_verif_audios")
async def join_verif_audios(requestor: str = Header(None), data: dict = Body(...)):
    try:
        AUDIOS_B64 = data["audio_data"]
        USER_ID = data["user_id"]

        # Convert all the audios from base64 opus to 16khz wav
        # then join them with a small gap between each audio
        # then return the audio as base64 wav

        audio_segments = []

        for audio_object in AUDIOS_B64:
            audio = audio_object["blob"]
            format, data = audio.split(",", 1)

            # Decode base64-encoded WebM audio to binary data
            binary_data = base64.b64decode(data)

            # Load binary data as an audio segment
            audiosegment = AudioSegment.from_file(BytesIO(binary_data), format="webm")
            
            audio_segments.append(audiosegment)

        # create silence AudioSegment
        silence = AudioSegment.silent(duration=500)

        # join audio_segments with silence using a for loop
        full_audio = AudioSegment.empty()
        
        for i, audio_segment in enumerate(audio_segments):
            full_audio += audio_segment
            if i != len(audio_segments) - 1:
                full_audio += silence

        full_audio_bytes = BytesIO()
        full_audio.export(full_audio_bytes, format='webm')
        
        base64_string = 'data:audio/webm;codecs=opus;base64,' + base64.b64encode(full_audio_bytes.getvalue()).decode('utf-8')

        # return the new audio

        return {
            "audio": base64_string,
            "user_id": USER_ID,
        }

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"error": str(e)}
