import torch
import whisper
import pydub
from pydub import AudioSegment
from io import BytesIO
from fastapi import FastAPI, Header, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import os
import tempfile

from openai import OpenAI

from logger import logger

import time

from routes.verification_audios_routes import router as verification_audios_router

# import pyannote.audio as pa
# from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Pipeline
from dotenv import load_dotenv, find_dotenv
from os import environ as env
# Ensure ffmpeg is used by pydub


# Loading env variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = FastAPI()

app.include_router(
    verification_audios_router,
    prefix="/verification_audios",
    tags=["verification_audios"],
)

allowed_origins = [
    # "http://localhost:*", "https://*.projectus.ai",
    "http://localhost:3000",
    "http://localhost:8089",  # for locust load testing
    "https://internaldev.projectus.ai",
    "https://staging.projectus.ai",
    "https://beta.projectus.ai",
    "https://api.projectus.ai",
    "https://nlp-api.projectus.ai",
    "https://whisper.projectus.ai",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pyinstrument import Profiler


PROFILING = False  # Set this from a settings model

if PROFILING:

    @app.middleware("http")
    async def pyinstrument_middleware(request, call_next):
        profiler = Profiler()
        profiler.start()

        response = await call_next(request)

        profiler.stop()
        if env.get("ENVIRONMENT") == "local":
            print(profiler.output_text(unicode=True, color=True))
        # else:
        #     logger.info(profiler.output_text())

        return response


openai_client = OpenAI()
whisper_model = whisper.load_model("medium")
# whisper_model = whisper.load_model("base")

# vad_pipeline = Pipeline.from_pretrained(
#     "pyannote/voice-activity-detection", use_auth_token=PYANNOTE_TOKEN
# )

PYANNOTE_TOKEN = os.getenv("PYANNOTE_TOKEN")
diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=PYANNOTE_TOKEN
)

silero_model, silero_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=False
)
(
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
) = silero_utils
FRAMERATE = 16000

VERIF_THRESHOLD = 0.85
MIN_DURATION_MS = 400

# IMPORT THIS LAST. There seems to be some conflict with pyannote and msft wavlm
from utils.verification_audios_util import similarity_fn


@app.get("/")
def check_health():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        return f"Hello! If you are seeing this, the server is running. And using the GPU :). Cuda version: {cuda_version}"
    else:
        return "Hello! If you are seeing this, the server is running. But not using the GPU :("


@app.post("/speech_to_text")
async def speech_to_text(requestor: str = Header(None), data: dict = Body(...)):
    try:
        # Extract the required properties from the incoming data
        AUDIO_B64 = data["audio_file"]
        CONV_ID = data["room"]
        USER = data["user"]
        UCID = data["ucid"]

        # SILENCE_THRESHOLD = 2000

        full_transcription = []
        full_airtime = 0
        segments = []
        text = ""
        full_transcript_language = "en"

        if AUDIO_B64 and "," in AUDIO_B64:
            format, data = AUDIO_B64.split(",", 1)

            # Decode base64-encoded WebM audio to binary data
            binary_data = base64.b64decode(data)

            # Load binary data as an audio segment
            audio = AudioSegment.from_file(BytesIO(binary_data), format="webm")

            # Create a temporary file to save the WAV audio data
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

            # Convert audio segment to WAV format and save to the temporary file
            wav_file_path = temp.name
            audio.export(wav_file_path, format="wav")

            # Perform the function with the file path
            # text = whisper_model.transcribe(wav_file_path)

            wav = AudioSegment.from_wav(wav_file_path)

            # get speech timestamps from full audio file
            wav_silero = read_audio(wav_file_path, sampling_rate=FRAMERATE)
            speech_timestamps = get_speech_timestamps(
                wav_silero, silero_model, sampling_rate=FRAMERATE
            )

            # voice activity detection test with pyannote
            # vad_res = vad_pipeline(wav_file_path)
            # timeline = vad_res.get_timeline().support()

            # initialization of variables

            # logger.info(f"Timeline: {timeline}")

            # loop through the timeline
            # for seg in timeline:
            for seg in speech_timestamps:
                # segment timestamps
                # t_start = (seg.start * 1000)
                # t_end = (seg.end * 1000)

                t_end = int((seg["end"] / FRAMERATE) * 1000)
                t_start = int((seg["start"] / FRAMERATE) * 1000)

                if t_start < 0:
                    t_start = 0

                start_sec = t_start / 1000
                end_sec = t_end / 1000

                airtime = end_sec - start_sec
                full_airtime += airtime

                # extract the segment
                audio_seg = wav[t_start:t_end]

                # Create a temporary file to save the WAV audio data
                temp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

                audio_seg.set_frame_rate(16000)
                audio_seg.set_channels(1)
                audio_seg.export(temp_seg.name, format="wav")

                # whisper transcribe
                seg_text = whisper_model.transcribe(temp_seg.name)

                if seg_text["language"] != "en":
                    full_transcript_language = seg_text["language"]

                # logger.info(f'\n\nWhisper response: {seg_text}\n\n')

                if (
                    seg_text["text"].replace(" ", "") == ""
                ):  # if the segment is empty or contain whitespace only
                    os.remove(temp_seg.name)
                    continue

                # append the segment transcription to the full transcription
                full_transcription.append(seg_text["text"])

                segments.append(
                    {
                        "text": seg_text["text"],
                        "start": start_sec,
                        "end": end_sec,
                        "language": seg_text["language"],
                    }
                )

                # logger.info(f"SEGMENT: {seg_text}")

                # delete the temporary file
                os.remove(temp_seg.name)

            # join the full transcription
            text = " ".join(full_transcription)

            # Delete the temporary WAV file after it's done
            os.remove(wav_file_path)

            segments = dict(enumerate(segments))

        full_result = {
            "text": text,
            "airtime": full_airtime,
            "segments": segments,
            "language": full_transcript_language,
        }

        return {
            "whisper_result": full_result,
            "ucid": UCID,
            "user": USER,
            "room": CONV_ID,
        }

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return {"error": str(e)}


@app.post("/speech_to_text_set_language")
async def speech_to_text_set_language(
    requestor: str = Header(None), data: dict = Body(...)
):
    try:
        # Extract the required properties from the incoming data
        AUDIO_B64 = data["audio_file"]
        CONV_ID = data["room"]
        USER = data["user"]
        UCID = data["ucid"]

        SET_LANG = data.get("set_lang", "en")
        DETECT_LANG = data.get("detect_lang", False)

        AUDIO_INDEX = data.get("audioIndex", None)

        # logger.info(f"SET_LANG: {SET_LANG}")

        # SILENCE_THRESHOLD = 2000

        full_transcription = []
        full_airtime = 0
        segments = []
        text = ""
        full_transcript_language = "en"

        if AUDIO_B64 and "," in AUDIO_B64:
            format, data = AUDIO_B64.split(",", 1)

            logger.warning(f"PROFILE:API_CALL:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}")

            # Decode base64-encoded WebM audio to binary data
            binary_data = base64.b64decode(data)

            # Load binary data as an audio segment
            audio = AudioSegment.from_file(BytesIO(binary_data), format="webm")

            # Create a temporary file to save the WAV audio data
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

            # Convert audio segment to WAV format and save to the temporary file
            wav_file_path = temp.name
            audio.export(wav_file_path, format="wav")

            wav = AudioSegment.from_wav(wav_file_path)

            processing_start = time.time()
            # Get the duration of the audio in milliseconds
            duration_ms = len(wav)

            # Optionally, convert the duration to seconds
            duration_seconds = round(duration_ms / 1000.0, 2)

            # get speech timestamps from full audio file
            wav_silero = read_audio(wav_file_path, sampling_rate=FRAMERATE)
            speech_timestamps = get_speech_timestamps(
                wav_silero, silero_model, sampling_rate=FRAMERATE
            )

            # loop through the timeline
            # for seg in timeline:
            for seg in speech_timestamps:
                # segment timestamps
                t_end = int((seg["end"] / FRAMERATE) * 1000)
                t_start = int((seg["start"] / FRAMERATE) * 1000)

                if t_start < 0:
                    t_start = 0

                start_sec = t_start / 1000
                end_sec = t_end / 1000

                airtime = end_sec - start_sec

                if airtime >= 1:
                    full_airtime += airtime

                    # extract the segment
                    audio_seg = wav[t_start:t_end]

                    # Create a temporary file to save the WAV audio data
                    temp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

                    audio_seg.set_frame_rate(16000)
                    audio_seg.set_channels(1)
                    audio_seg.export(temp_seg.name, format="wav")

                    PROMPT_EN = "The following is a transcript from a person speaking in English. Some words may be in Spanish."
                    PROMPT_ES = "La siguiente es una transcripción de una persona que habla en español. Algunas palabras pueden estar en inglés."

                    prompt = PROMPT_EN if SET_LANG == "en" else PROMPT_ES

                    # decode_options = {"language": SET_LANG, "initial_prompt": prompt}
                    decode_options = {"language": SET_LANG}

                    # log here when the transcription starts
                    whisper_start = time.time()

                    # whisper transcribe
                    seg_text = whisper_model.transcribe(
                        audio=temp_seg.name, **decode_options
                    )  # TODO add initial prompt

                    whisper_time = round(time.time() - whisper_start, 2)

                    audio_time = round((end_sec - start_sec), 2)

                    logger.warning(
                        f"PROFILE:WHISPER:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}:audio_time_{audio_time}:{whisper_time}"
                    )
                    # TODO: check if the audio index is in the data to include it and have better tracking

                    if seg_text["language"] != "en":
                        full_transcript_language = seg_text["language"]

                    if (
                        seg_text["text"].replace(" ", "") == ""
                    ):  # if the segment is empty or contain whitespace only
                        os.remove(temp_seg.name)
                        continue

                    # append the segment transcription to the full transcription
                    full_transcription.append(seg_text["text"])

                    segments.append(
                        {
                            "text": seg_text["text"],
                            "start": start_sec,
                            "end": end_sec,
                            "language": seg_text["language"],
                        }
                    )

                    # delete the temporary file
                    os.remove(temp_seg.name)

            processing_time = round(time.time() - processing_start, 2)

            logger.warning(
                f"PROFILE:FULL_PROCESSING:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}:audio_time_{duration_seconds}:{processing_time}"
            )

            # join the full transcription
            text = " ".join(full_transcription)

            # Delete the temporary WAV file after it's done
            os.remove(wav_file_path)

            segments = dict(enumerate(segments))

        full_result = {
            "text": text,
            "airtime": full_airtime,
            "segments": segments,
            "language": full_transcript_language,
        }

        return JSONResponse(
            status_code=200,
            content={
                "whisper_result": full_result,
                "ucid": UCID,
                "user": USER,
                "room": CONV_ID,
            },
        )

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/speech_to_text_set_language_api")
async def speech_to_text_set_language_api(
    requestor: str = Header(None), data: dict = Body(...)
):
    try:
        # Extract the required properties from the incoming data
        AUDIO_B64 = data["audio_file"]
        CONV_ID = data["room"]
        USER = data["user"]
        UCID = data["ucid"]

        SET_LANG = data.get("set_lang", "en")
        DETECT_LANG = data.get("detect_lang", False)

        # logger.info(f"SET_LANG: {SET_LANG}")

        # SILENCE_THRESHOLD = 2000

        full_transcription = []
        full_airtime = 0
        segments = []
        text = ""
        full_transcript_language = "en"

        if AUDIO_B64 and "," in AUDIO_B64:
            format, data = AUDIO_B64.split(",", 1)

            # Decode base64-encoded WebM audio to binary data
            binary_data = base64.b64decode(data)

            # Load binary data as an audio segment
            audio = AudioSegment.from_file(BytesIO(binary_data), format="webm")

            # Create a temporary file to save the WAV audio data
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

            # Convert audio segment to WAV format and save to the temporary file
            wav_file_path = temp.name
            audio.export(wav_file_path, format="wav")

            wav = AudioSegment.from_wav(wav_file_path)

            # get speech timestamps from full audio file
            wav_silero = read_audio(wav_file_path, sampling_rate=FRAMERATE)
            speech_timestamps = get_speech_timestamps(
                wav_silero, silero_model, sampling_rate=FRAMERATE
            )

            # loop through the timeline
            # for seg in timeline:
            for seg in speech_timestamps:
                # segment timestamps
                t_end = int((seg["end"] / FRAMERATE) * 1000)
                t_start = int((seg["start"] / FRAMERATE) * 1000)

                if t_start < 0:
                    t_start = 0

                start_sec = t_start / 1000
                end_sec = t_end / 1000

                airtime = end_sec - start_sec

                if airtime >= 1:
                    full_airtime += airtime

                    # extract the segment
                    audio_seg = wav[t_start:t_end]

                    # Create a temporary file to save the WAV audio data
                    temp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

                    audio_seg.set_frame_rate(16000)
                    audio_seg.set_channels(1)
                    audio_seg.export(temp_seg.name, format="wav")

                    PROMPT_EN = "The following is a transcript from a person speaking in English. Some words may be in Spanish."
                    PROMPT_ES = "La siguiente es una transcripción de una persona que habla en español. Algunas palabras pueden estar en inglés."

                    prompt = PROMPT_EN if SET_LANG == "en" else PROMPT_ES

                    # decode_options = {"language": SET_LANG, "initial_prompt": prompt}
                    # decode_options = {"language": SET_LANG}
                    # whisper transcribe
                    # seg_text = whisper_model.transcribe(
                    #     audio=temp_seg.name, **decode_options
                    # )  # TODO add initial prompt

                    temp_seg_file = open(temp_seg.name, "rb")

                    api_seg_text = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=temp_seg_file,
                        response_format="text",
                        language=SET_LANG,
                    )

                    temp_seg_file.close()

                    # if seg_text["language"] != "en":
                    #     full_transcript_language = seg_text["language"]

                    if (
                        api_seg_text.replace(" ", "") == ""
                    ):  # if the segment is empty or contain whitespace only
                        os.remove(temp_seg.name)
                        continue

                    # append the segment transcription to the full transcription
                    full_transcription.append(api_seg_text)

                    segments.append(
                        {
                            "text": api_seg_text,
                            "start": start_sec,
                            "end": end_sec,
                            "language": SET_LANG,
                        }
                    )

                    # delete the temporary file
                    os.remove(temp_seg.name)

            # join the full transcription
            text = " ".join(full_transcription)

            # Delete the temporary WAV file after it's done
            os.remove(wav_file_path)

            segments = dict(enumerate(segments))

        full_result = {
            "text": text,
            "airtime": full_airtime,
            "segments": segments,
            "language": full_transcript_language,
        }

        return JSONResponse(
            status_code=200,
            content={
                "whisper_result": full_result,
                "ucid": UCID,
                "user": USER,
                "room": CONV_ID,
            },
        )

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/speech_to_text_set_language_batch")
async def speech_to_text_set_language_batch(
    requestor: str = Header(None), data: dict = Body(...)
):
    SILENCE_DURATION_MS = 200

    try:
        # Extract the required properties from the incoming data
        BATCH_LIST = data["audio_batch"]
        CONV_ID = data["conv_id"]
        USER = data["username"]
        # UCID = data["ucid"]
        UCID = f'{USER}-{CONV_ID}'

        SET_LANG = data.get("set_lang", "en")

        AUDIO_INDEX = data.get("audioIndex", None)

        full_transcription = []
        full_airtime = 0
        segments = []
        text = ""
        full_transcript_language = "en"
        audio_segments = []

        logger.warning(f"PROFILE:BATCH_API_CALL:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}")
        logger.warning(
            f"PROFILE:BATCH_API_CALL:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}:BATCH_LIST_{len(BATCH_LIST)}"
        )

        for AUDIO_B64 in BATCH_LIST:
            if AUDIO_B64 and "," in AUDIO_B64:
                format, data = AUDIO_B64.split(",", 1)

                # Decode base64-encoded WebM audio to binary data
                binary_data = base64.b64decode(data)

                # Load binary data as an audio segment
                audio = AudioSegment.from_file(BytesIO(binary_data), format="webm")

                # Create a temporary file to save the WAV audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:

                # Convert audio segment to WAV format and save to the temporary file
                    wav_file_path = temp.name
                    audio.export(wav_file_path, format="wav")

                wav = AudioSegment.from_wav(wav_file_path)

                duration_ms = len(wav)

                wav_silero = read_audio(wav_file_path, sampling_rate=FRAMERATE)
                speech_timestamps = get_speech_timestamps(
                    wav_silero, silero_model, sampling_rate=FRAMERATE
                )

                for seg in speech_timestamps:
                    t_end = int((seg["end"] / FRAMERATE) * 1000)
                    t_start = int((seg["start"] / FRAMERATE) * 1000)

                    if t_start < 0:
                        t_start = 0

                    start_sec = t_start / 1000
                    end_sec = t_end / 1000

                    airtime = end_sec - start_sec

                    if airtime >= 1:
                        full_airtime += airtime

                        audio_seg = wav[t_start:t_end]

                        audio_segments.append(
                            {
                                "audio": audio_seg,
                                "start": start_sec,
                                "end": end_sec,
                            }
                        )

                os.remove(wav_file_path)

        silence_seg = AudioSegment.silent(duration=SILENCE_DURATION_MS)

        # combined_segments = sum(
        #     [(seg["audio"] + silence_seg) for seg in audio_segments],
        #     AudioSegment.empty(),
        # )
        # this adds a silence at the end which causes hallucinations
        
        # create list of interwoven audio segments and silence
        interwoven_segments = []
        for seg in audio_segments:
            interwoven_segments.append(seg["audio"])
            interwoven_segments.append(silence_seg)

        # remove the last appended silence
        if interwoven_segments:
            interwoven_segments.pop()

        combined_segments = sum(interwoven_segments, AudioSegment.empty())

        # Create a temporary file to save the combination of all segments
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stitched_temp:
            combined_segments.export(stitched_temp.name, format="wav")

        # combined_segments.export("stitched.wav", format="wav")

        PROMPT_EN = "The following is a transcript from a person speaking in English. Some words may be in Spanish."
        PROMPT_ES = "La siguiente es una transcripción de una persona que habla en español. Algunas palabras pueden estar en inglés."

        prompt = PROMPT_EN if SET_LANG == "en" else PROMPT_ES

        decode_options = {"language": SET_LANG}

        # full_text = whisper_model.transcribe(audio="stitched.wav", **decode_options)

        start_time = time.time()

        full_text = whisper_model.transcribe(audio=stitched_temp.name, **decode_options)

        transcript_time = round(time.time() - start_time, 2)
        duration_ms = len(combined_segments)
        duration_s = round(duration_ms / 1000, 2)

        logger.warning(
            f"PROFILE:BATCH_WHISPER:{USER}:{CONV_ID}:ID_{AUDIO_INDEX}:audio_time_{duration_s}:{transcript_time}"
        )

        full_transcript_language = full_text["language"]

        text = full_text["text"]

        full_result = {
            "text": text,
            "airtime": full_airtime,
            "language": full_transcript_language,
        }

        # os.remove("stitched.wav")
        # Delete the temporary file after it's done
        os.remove(stitched_temp.name)

        return JSONResponse(
            status_code=200,
            content={
                "whisper_result": full_result,
                "ucid": UCID,
                "user": USER,
                "room": CONV_ID,
            },
        )

    except Exception as e:
        logger.error(f"ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/speech_to_text_verification")
async def speech_to_text_verification(
    requestor: str = Header(None), data: dict = Body(...)
):
    SILENCE_DURATION_MS = 200
    MIN_DURATION_MS = 1000  # Minimum duration for a valid segment in ms
    VERIF_THRESHOLD = 0.75  # Example threshold, adjust based on your requirement

    try:
        # Extract the required properties from the incoming data
        BATCH_LIST = data["audio_batch"]
        CONV_ID = data["conv_id"]
        USER = data["username"]
        UCID = f'{USER}-{CONV_ID}'
        VERIF_AUDIO = data["verif_audio"]  # Verification audio segment

        SET_LANG = data.get("set_lang", "en")
        AUDIO_INDEX = data.get("audioIndex", None)

        logger.warning(f"Starting speech-to-text verification for {UCID}")
        logger.warning(f"{UCID}: Set language: {SET_LANG}, Audio index: {AUDIO_INDEX}")

        # Prepare verification audio
        temp_verif = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        verif_audio = AudioSegment.from_file(
            BytesIO(base64.b64decode(VERIF_AUDIO.split(",", 1)[1])), format="webm"
        )
        verif_audio.set_frame_rate(16000)
        verif_audio.set_channels(1)
        verif_audio.export(temp_verif.name, format="wav")

        logger.warning(f"{UCID}: Verification audio prepared.")

        full_transcription = []
        full_airtime = 0
        audio_segments = []

        logger.warning(f"{UCID}: Batch list length: {len(BATCH_LIST)}")

        for AUDIO_B64 in BATCH_LIST:
            if AUDIO_B64 and "," in AUDIO_B64:
                format, data = AUDIO_B64.split(",", 1)

                logger.warning(f"{UCID}: Processing batch item with format: {format}")

                # Decode base64-encoded WebM audio to binary data
                binary_data = base64.b64decode(data)

                # Load binary data as an audio segment
                audio = AudioSegment.from_file(BytesIO(binary_data), format="webm")

                # Create a temporary file to save the WAV audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                    # Convert audio segment to WAV format and save to the temporary file
                    wav_file_path = temp.name
                    audio.export(wav_file_path, format="wav")

                logger.warning(f"{UCID}: Converted audio to WAV format: {wav_file_path}")

                wav = AudioSegment.from_wav(wav_file_path)

                # Perform voice activity detection to get speech timestamps
                wav_silero = read_audio(wav_file_path, sampling_rate=FRAMERATE)
                speech_timestamps = get_speech_timestamps(
                    wav_silero, silero_model, sampling_rate=FRAMERATE
                )

                logger.warning(f"{UCID}: Speech timestamps found: {speech_timestamps}")

                for seg in speech_timestamps:
                    t_end = int((seg["end"] / FRAMERATE) * 1000)
                    t_start = int((seg["start"] / FRAMERATE) * 1000)

                    if t_start < 0:
                        t_start = 0

                    start_sec = t_start / 1000
                    end_sec = t_end / 1000

                    # Extract the segment
                    audio_seg = wav[t_start:t_end]

                    # Create a temporary file to save the WAV audio data
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_seg:
                        audio_seg.set_frame_rate(16000)
                        audio_seg.set_channels(1)
                        audio_seg.export(temp_seg.name, format="wav")

                        logger.warning(f"{UCID}: Exported segment for diarization: {temp_seg.name}")

                        # Perform diarization on the segment
                        diar_results = diar_pipeline(temp_seg.name)

                        for turn, track, speaker in diar_results.itertracks(yield_label=True):
                            logger.warning(f"{UCID}: Diarization: start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

                            diar_start = int(turn.start * 1000)
                            diar_end = int(turn.end * 1000)
                            diar_segment = audio_seg[diar_start:diar_end]
                            duration_ms = len(diar_segment)

                            # If the segment is too short, skip it
                            if duration_ms < MIN_DURATION_MS:
                                logger.warning(f"{UCID}: Segment too short, skipping: {duration_ms}ms")
                                continue

                            diar_start_sec = diar_start / 1000
                            diar_end_sec = diar_end / 1000
                            
                            # Now that the diarization segment has been extracted, export it with a random filename
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_diar_seg:
                                diar_segment.export(temp_diar_seg.name, format="wav")

                                similarity = similarity_fn(temp_diar_seg.name, temp_verif.name)
                                logger.warning(f"{UCID}: Similarity score for segment: {similarity}")

                                if similarity >= VERIF_THRESHOLD:
                                    airtime = diar_end_sec - diar_start_sec
                                    full_airtime += airtime

                                    audio_segments.append(
                                        {
                                            "audio": diar_segment,
                                            "start": diar_start_sec,
                                            "end": diar_end_sec,
                                        }
                                    )
                    
                    os.remove(temp_seg.name)
                
                os.remove(wav_file_path)

        silence_seg = AudioSegment.silent(duration=SILENCE_DURATION_MS)

        interwoven_segments = []
        for seg in audio_segments:
            interwoven_segments.append(seg["audio"])
            interwoven_segments.append(silence_seg)

        if interwoven_segments:
            interwoven_segments.pop()

        combined_segments = sum(interwoven_segments, AudioSegment.empty())

        # Create a temporary file to save the combination of all segments
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stitched_temp:
            combined_segments.export(stitched_temp.name, format="wav")

        logger.info("All segments stitched into a single audio file.")

        PROMPT_EN = "The following is a transcript from a person speaking in English. Some words may be in Spanish."
        PROMPT_ES = "La siguiente es una transcripción de una persona que habla en español. Algunas palabras pueden estar en inglés."

        prompt = PROMPT_EN if SET_LANG == "en" else PROMPT_ES

        decode_options = {"language": SET_LANG}

        start_time = time.time()

        full_text = whisper_model.transcribe(audio=stitched_temp.name, **decode_options)

        transcript_time = round(time.time() - start_time, 2)
        duration_ms = len(combined_segments)
        duration_s = round(duration_ms / 1000, 2)

        logger.warning(f"Transcription completed in {transcript_time}s for {duration_s}s of audio.")

        full_transcript_language = full_text["language"]
        if full_text["text"] != "":
            text = full_text["text"]
            logger.info(f"Transcript: {text}")

            full_result = {
                "text": text,
                "airtime": full_airtime,
                "language": full_transcript_language,
            }
        else:
             full_result = {
                "text": '',
                "airtime": full_airtime,
                "language": full_transcript_language,
            }

        #os.remove(stitched_temp.name)
        #os.remove(temp_verif.name)
        logger.warning(f"Returning transcription result for {UCID}")

        return JSONResponse(
            status_code=200,
            content={
                "whisper_result": full_result,
                "ucid": UCID,
                "user": USER,
                "room": CONV_ID,
            },
        )

    except Exception as e:
        logger.error(f"ERROR in speech_to_text_verification: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8050)