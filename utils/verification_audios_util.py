import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from logger import logger
from pydub import AudioSegment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "microsoft/wavlm-base-plus-sv"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMForXVector.from_pretrained(model_name).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

def similarity_fn(audio1_array, audio2_array, sr1, sr2):
    try:
        # Ensure both audio arrays have the same sampling rate
        if sr1 != 16000:
            logger.warning("Resampling audio1 to 16000 Hz")
            audio1_segment = AudioSegment(audio1_array.tobytes(), frame_rate=sr1, sample_width=audio1_array.dtype.itemsize, channels=1)
            audio1_segment = audio1_segment.set_frame_rate(16000)
            audio1_array = np.array(audio1_segment.get_array_of_samples(), dtype=np.float32) / (1 << (8 * audio1_segment.sample_width - 1))
        
        if sr2 != 16000:
            logger.warning("Resampling audio2 to 16000 Hz")
            audio2_segment = AudioSegment(audio2_array.tobytes(), frame_rate=sr2, sample_width=audio2_array.dtype.itemsize, channels=1)
            audio2_segment = audio2_segment.set_frame_rate(16000)
            audio2_array = np.array(audio2_segment.get_array_of_samples(), dtype=np.float32) / (1 << (8 * audio2_segment.sample_width - 1))

        # Pad audio to ensure it meets the minimum length requirement
        target_length = 16000 * 10  # 10 seconds at 16000 Hz
        audio1_array = pad_audio(audio1_array, target_length)
        audio2_array = pad_audio(audio2_array, target_length)

        # Feature extraction
        inputs = feature_extractor([audio1_array, audio2_array], sampling_rate=16000, return_tensors="pt", padding=True)

        # Move inputs to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            emb1, emb2 = outputs.embeddings[0], outputs.embeddings[1]

        # Normalize embeddings
        emb1 = torch.nn.functional.normalize(emb1, dim=-1).cpu()
        emb2 = torch.nn.functional.normalize(emb2, dim=-1).cpu()

        # Calculate cosine similarity
        similarity = cosine_sim(emb1, emb2).item()

        return similarity

    except Exception as e:
        logger.error(f"ERROR in similarity_fn: {e}")
        return None

def pad_audio(arr, target_length):
    if len(arr) < target_length:
        pad_width = target_length - len(arr)
        arr = np.pad(arr, (0, pad_width), 'constant')
    return arr
