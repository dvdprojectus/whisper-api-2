import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from logger import logger
from pydub import AudioSegment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

def load_audio(file_name):
    audio = AudioSegment.from_file(file_name)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.apply_gain(-1.0)
    audio = audio.strip_silence(silence_len=100, silence_thresh=-50)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

def pad_audio(arr, target_length):
    if len(arr) < target_length:
        pad_width = target_length - len(arr)
        arr = np.pad(arr, (0, pad_width), 'constant')
    return arr

def similarity_fn(path1, path2):
    if not (path1 and path2):
        logger.error("ERROR: one of the audio paths does not exist")
        return None

    try:
        wav1, sr1 = load_audio(path1)
        wav2, sr2 = load_audio(path2)

        # Pad audio to ensure it meets the minimum length requirement
        target_length = 16000 * 10  # 10 seconds at 16000 Hz
        wav1 = pad_audio(wav1, target_length)
        wav2 = pad_audio(wav2, target_length)

        input1 = feature_extractor(
            wav1, return_tensors="pt", sampling_rate=16000
        ).input_values.to(device)
        input2 = feature_extractor(
            wav2, return_tensors="pt", sampling_rate=16000
        ).input_values.to(device)

        with torch.no_grad():
            emb1 = model(input1).embeddings
            emb2 = model(input2).embeddings

        emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
        emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()
        similarity = cosine_sim(emb1, emb2).numpy()[0]

        return similarity
    except Exception as e:
        logger.error(f"ERROR: {e}")
        return None