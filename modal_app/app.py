"""Modal app with NVIDIA Parakeet TDT transcription service."""

import io
import json
import os
import sys
from typing import Iterator, Dict, Any, Optional
from pathlib import Path

import modal

# =============================================================================
# MODAL CONFIG - Embedded to avoid import issues in Modal containers
# Keep in sync with config.py
# =============================================================================

# Modal configuration
MODAL_APP_NAME = "transcodio-app"
MODAL_VOLUME_NAME = "parakeet-models"
MODAL_GPU_TYPE = "L4"
MODAL_CONTAINER_IDLE_TIMEOUT = 20
MODAL_TIMEOUT = 9000
MODAL_MEMORY_MB = 8192

# Cold start optimization flags
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = False
EXTENDED_IDLE_TIMEOUT_SECONDS = 300

# STT Model configuration
STT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000

# Silence detection
SILENCE_THRESHOLD_DB = -40
SILENCE_MIN_LENGTH_MS = 700

# Speaker diarization
ENABLE_SPEAKER_DIARIZATION = True
DIARIZATION_MODEL = "nvidia/speakerverification_en_titanet_large"
DIARIZATION_MIN_SPEAKERS = 1
DIARIZATION_MAX_SPEAKERS = 5
DIARIZATION_WINDOW_LENGTHS = [1.5, 1.0, 0.5]
DIARIZATION_SHIFT_LENGTH = 0.75

# Voice cloning
QWEN_TTS_SOURCE_REF = (
    "git+https://github.com/QwenLM/Qwen3-TTS.git"
    "@1ab0dd75353392f28a0d05d9ca960c9954b13c83"
)
TTS_MODELS = {
    "qwen": {
        "name": "Qwen3-TTS",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sample_rate": 24000,
        "gpu_type": "L4",
        "memory_mb": 8192,
        "description": "Fast, good quality (1.7B params)",
    },
}
TTS_CONTAINER_IDLE_TIMEOUT = 20
TTS_TIMEOUT = 900
TTS_ENABLE_CHUNKING = True
TTS_CHUNK_MAX_WORDS = 80          # max words per chunk (pysbd-based splitting)
TTS_SHORT_SENTENCE_WORDS = 3      # sentences <= this get merged with a neighbor
TTS_ENABLE_BATCHING = True
TTS_BATCH_SIZE = 8
TTS_CHUNK_MAX_RETRIES = 3                # retries per chunk after first attempt
TTS_SLOW_CHUNK_MULTIPLIER = 2.5          # retry when chunk runtime exceeds avg * multiplier
TTS_SLOW_BATCH_MULTIPLIER = 2.5          # fallback batch->single when batch runtime exceeds expected
TTS_MIN_BASELINE_CHUNKS = 3              # chunk samples before avg-based slow detection
TTS_SLOW_CHUNK_ABSOLUTE_S = 45.0         # absolute slow threshold before avg baseline is stable
TTS_SLOW_BATCH_ABSOLUTE_S = 120.0        # absolute slow threshold for whole batch
# Gap inserted between stitched chunks based on the boundary type
TTS_GAP_PARAGRAPH_MS = 400        # blank line / paragraph break
TTS_GAP_SENTENCE_MS = 180         # sentence-end punctuation (. ! ?)
TTS_GAP_CLAUSE_MS = 60            # clause boundary (, ; : — –)
TTS_EDGE_TRIM_MS = 120
TTS_EDGE_SILENCE_THRESHOLD_DB = -42.0   # dB below peak RMS for edge trim
TTS_SILENCE_GAP_LIMIT_MS = 500          # cap any internal silence gap longer than this
TTS_LUFS_TARGET = -18.0                 # loudness normalization target (LUFS)
VOICES_INDEX_FILE = "index.json"
MAX_SAVED_VOICES = 50

# Image generation
IMAGE_GENERATION_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_GPU_TYPE = "L4"
IMAGE_MEMORY_MB = 16384
IMAGE_CONTAINER_IDLE_TIMEOUT = 20

# =============================================================================

# Create the container image with all required dependencies
stt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": "/models",  # Use Modal volume for HuggingFace cache
        "DEBIAN_FRONTEND": "noninteractive",
        "CXX": "g++",
        "CC": "g++",
    })
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface-hub==0.36.0",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "numpy<2",
        "pydub==0.25.1",
        "scikit-learn>=1.3.0",  # For spectral clustering
        "soundfile>=0.12.1",    # For audio file I/O
    )
)

# Image generation image for FLUX.1-schnell
flux_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_HOME": "/models",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .pip_install(
        "torch",
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "sentencepiece",
        "protobuf",
        "huggingface-hub>=0.24.0",
    )
)

# TTS image for Qwen3-TTS voice cloning
qwen_tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({
        "HF_HOME": "/models",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .apt_install("ffmpeg", "git", "libsndfile1", "sox")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
    )
    .run_commands("pip install psutil")
    .run_commands("pip install ninja")
    .run_commands("pip install flash-attn==2.8.3 --no-build-isolation")
    .pip_install(
        QWEN_TTS_SOURCE_REF,
        "chardet==5.2.0",
        "ffmpeg-python==0.2.0",
        "librosa==0.11.0",
        "num2words==0.5.14",
        "pyloudnorm==0.1.1",
        "psutil",
        "pydantic",
        "pysbd==0.3.4",
        "soundfile==0.13.1",
        "spacy==3.8.4",
    )
    .run_commands("python -m spacy download en_core_web_sm")
)


# Create Modal app
app = modal.App(MODAL_APP_NAME)

# Create Modal Volume for persistent model storage
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Build decorator arguments based on optimization flags
decorator_kwargs = {
    "image": stt_image,
    "gpu": MODAL_GPU_TYPE,
    "scaledown_window": (
        EXTENDED_IDLE_TIMEOUT_SECONDS
        if EXTENDED_IDLE_TIMEOUT
        else MODAL_CONTAINER_IDLE_TIMEOUT
    ),
    "timeout": MODAL_TIMEOUT,
    "memory": MODAL_MEMORY_MB,
    "volumes": {"/models": volume},
}

# Print configuration summary
print(f"GPU Type: {MODAL_GPU_TYPE}")
print(f"Memory: {MODAL_MEMORY_MB}MB")
print(f"Container idle timeout: {decorator_kwargs['scaledown_window']}s")

# Add CPU memory snapshot if enabled
if ENABLE_CPU_MEMORY_SNAPSHOT:
    decorator_kwargs["enable_memory_snapshot"] = True
    print(f"CPU Memory Snapshots: ENABLED")

# Add GPU memory snapshot if enabled (requires CPU snapshots)
if ENABLE_GPU_MEMORY_SNAPSHOT:
    if not ENABLE_CPU_MEMORY_SNAPSHOT:
        raise ValueError(
            "ENABLE_GPU_MEMORY_SNAPSHOT requires ENABLE_CPU_MEMORY_SNAPSHOT to be True"
        )
    decorator_kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
    print(f"GPU Memory Snapshots: ENABLED (Experimental)")


class NoStdStreams:
    """Context manager to suppress NeMo's verbose stdout/stderr."""

    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


def align_speakers_to_segments(transcription_segments: list, speaker_timeline: list) -> list:
    """
    Assign speaker labels to transcription segments based on maximum overlap.

    Args:
        transcription_segments: List of dicts with {id, start, end, text}
        speaker_timeline: List of dicts with {start, end, speaker}

    Returns:
        List of segments with speaker field added
    """
    def calculate_overlap(range1, range2):
        """Calculate overlap duration between two time ranges."""
        start1, end1 = range1
        start2, end2 = range2
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        return max(0, overlap_end - overlap_start)

    for segment in transcription_segments:
        max_overlap = 0
        assigned_speaker = None

        for speaker_seg in speaker_timeline:
            overlap = calculate_overlap(
                (segment["start"], segment["end"]),
                (speaker_seg["start"], speaker_seg["end"])
            )

            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = speaker_seg["speaker"]

        # Assign speaker label (convert to "Speaker 1", "Speaker 2", etc.)
        if assigned_speaker is not None:
            segment["speaker"] = f"Speaker {assigned_speaker + 1}"
        else:
            segment["speaker"] = "Speaker 1"  # Fallback for no overlap

    return transcription_segments


@app.cls(**decorator_kwargs)
class ParakeetSTTModel:
    """NVIDIA Parakeet TDT model for streaming GPU-accelerated transcription."""

    @modal.enter(snap=ENABLE_CPU_MEMORY_SNAPSHOT or ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load Parakeet TDT model once per container (runs on container startup)."""
        import logging
        import nemo.collections.asr as nemo_asr
        import time

        start_time = time.time()

        # Suppress NeMo's verbose logging
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        print(f"Loading Parakeet TDT model: {STT_MODEL_ID}...")

        # NeMo uses HuggingFace Hub internally, respects HF_HOME env var
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=STT_MODEL_ID
        )
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f}s")

        # Optional: Warm up model with dummy forward pass
        if ENABLE_MODEL_WARMUP:
            self._warmup_model()

        total_time = time.time() - start_time
        print(f"Total initialization time: {total_time:.2f}s")

        # Print optimization status
        if ENABLE_CPU_MEMORY_SNAPSHOT:
            print("-> This state will be captured in CPU memory snapshot")
        if ENABLE_GPU_MEMORY_SNAPSHOT:
            print("-> GPU state (including loaded model) will be captured in snapshot")

    def _warmup_model(self):
        """Warmup model with dummy audio to compile CUDA kernels."""
        import numpy as np
        import time

        print("Warming up model with dummy forward pass...")
        warmup_start = time.time()

        # Create 1 second of silence at 16kHz (Parakeet's native sample rate)
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        # Run transcription to compile kernels (suppress output)
        with NoStdStreams():
            self.model.transcribe([dummy_audio])

        warmup_time = time.time() - warmup_start
        print(f"Model warm-up completed in {warmup_time:.2f}s")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe complete audio file (non-streaming).

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)

        Returns:
            Dict with transcription results
        """
        import numpy as np

        # Convert bytes to numpy array (int16 → float32)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        duration = len(audio_data) / SAMPLE_RATE

        print(f"Transcribing audio: {len(audio_data)} samples ({duration:.2f}s)")

        # Transcribe with NeMo (suppress verbose logs)
        with NoStdStreams():
            output = self.model.transcribe([audio_data])

        # Extract text from NeMo output
        text = output[0].text if output and hasattr(output[0], 'text') else ""

        return {
            "text": text,
            "language": "en",  # Parakeet TDT 0.6B v3 is English-only
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                }
            ],
        }

    @modal.method()
    def transcribe_stream(self, audio_bytes: bytes, actual_duration: float = 0.0) -> Iterator[str]:
        """
        Transcribe audio with real progressive streaming using silence detection.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)
            actual_duration: Actual duration from FFmpeg

        Yields:
            JSON strings: metadata → segment(s) → complete
        """
        import numpy as np
        from pydub import AudioSegment, silence

        try:
            # Calculate duration
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            duration = actual_duration if actual_duration > 0 else len(audio_array) / SAMPLE_RATE

            # Yield metadata first
            yield json.dumps({
                "type": "metadata",
                "language": "en",
                "duration": duration,
            })

            # Create AudioSegment for silence detection
            audio_segment = AudioSegment(
                data=audio_bytes,
                channels=1,
                sample_width=2,  # 16-bit
                frame_rate=SAMPLE_RATE,
            )

            # Detect silent windows
            silent_windows = silence.detect_silence(
                audio_segment,
                min_silence_len=SILENCE_MIN_LENGTH_MS,
                silence_thresh=SILENCE_THRESHOLD_DB,
            )

            print(f"Detected {len(silent_windows)} silent windows")

            # Accumulate all segment texts for final transcription
            all_segments = []

            # If no silence detected, transcribe entire audio at once
            if not silent_windows:
                audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                with NoStdStreams():
                    output = self.model.transcribe([audio_float])
                text = output[0].text if output and hasattr(output[0], 'text') else ""

                if text.strip():
                    all_segments.append(text)

                yield json.dumps({
                    "type": "segment",
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                })
            else:
                # Transcribe segments progressively based on silence boundaries
                segment_id = 0
                current_pos = 0

                for window_start, window_end in silent_windows:
                    # Extract segment up to end of silence
                    segment_audio = audio_segment[current_pos:window_end]

                    # Skip very short segments (< 100ms)
                    if len(segment_audio) < 100:
                        continue

                    # Transcribe segment
                    segment_float = np.frombuffer(
                        segment_audio.raw_data,
                        dtype=np.int16
                    ).astype(np.float32)

                    with NoStdStreams():
                        output = self.model.transcribe([segment_float])

                    text = output[0].text if output and hasattr(output[0], 'text') else ""

                    # Only yield if there's actual text
                    if text.strip():
                        start_time = current_pos / 1000.0  # ms → seconds
                        end_time = window_end / 1000.0

                        all_segments.append(text)

                        yield json.dumps({
                            "type": "segment",
                            "id": segment_id,
                            "start": start_time,
                            "end": end_time,
                            "text": text,
                        })

                        segment_id += 1

                    current_pos = window_end

                # Process any remaining audio after the last silence
                if current_pos < len(audio_segment):
                    remaining_audio = audio_segment[current_pos:]
                    remaining_float = np.frombuffer(
                        remaining_audio.raw_data,
                        dtype=np.int16
                    ).astype(np.float32)

                    with NoStdStreams():
                        output = self.model.transcribe([remaining_float])

                    text = output[0].text if output and hasattr(output[0], 'text') else ""

                    if text.strip():
                        all_segments.append(text)

                        yield json.dumps({
                            "type": "segment",
                            "id": segment_id,
                            "start": current_pos / 1000.0,
                            "end": duration,
                            "text": text,
                        })

            # Yield completion with full transcription
            full_transcription = " ".join(all_segments)
            yield json.dumps({
                "type": "complete",
                "text": full_transcription,
            })

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            yield json.dumps({
                "type": "error",
                "error": str(e),
            })


@app.cls(**decorator_kwargs)
class SpeakerDiarizerModel:
    """NVIDIA TitaNet + clustering for speaker diarization."""

    @modal.enter(snap=ENABLE_CPU_MEMORY_SNAPSHOT or ENABLE_GPU_MEMORY_SNAPSHOT)
    def load_model(self):
        """Load TitaNet speaker embedding model."""
        import logging
        import nemo.collections.asr as nemo_asr
        import time

        start_time = time.time()
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        print(f"Loading TitaNet model: {DIARIZATION_MODEL}...")

        self.embedding_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=DIARIZATION_MODEL
        )

        load_time = time.time() - start_time
        print(f"TitaNet model loaded in {load_time:.2f}s")

    @modal.method()
    def diarize(self, audio_bytes: bytes, duration: float) -> list:
        """
        Perform speaker diarization on audio.

        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono WAV)
            duration: Audio duration in seconds

        Returns:
            List of speaker segments: [{"start": 0.0, "end": 5.2, "speaker": 0}, ...]
        """
        import numpy as np
        from sklearn.cluster import SpectralClustering
        import tempfile
        import soundfile as sf

        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Save to temporary WAV file (NeMo requires file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, SAMPLE_RATE)
                audio_path = tmp.name

            # Extract single-scale embeddings (fixes multi-scale artifact issue)
            embeddings_list = []
            timestamps_list = []

            # Use only the first (longest) window length to avoid multi-scale artifacts
            window_length = DIARIZATION_WINDOW_LENGTHS[0]
            window_samples = int(window_length * SAMPLE_RATE)
            shift_samples = int(DIARIZATION_SHIFT_LENGTH * SAMPLE_RATE)

            num_windows = max(1, (len(audio_array) - window_samples) // shift_samples + 1)

            for i in range(num_windows):
                start_sample = i * shift_samples
                end_sample = min(start_sample + window_samples, len(audio_array))

                window_audio = audio_array[start_sample:end_sample]

                # Skip windows that are too short
                if len(window_audio) < window_samples * 0.5:
                    continue

                # Save window to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_win:
                    sf.write(tmp_win.name, window_audio, SAMPLE_RATE)

                    # Get embedding
                    with NoStdStreams():
                        embedding = self.embedding_model.get_embedding(tmp_win.name)

                    embeddings_list.append(embedding.cpu().numpy().flatten())
                    timestamps_list.append({
                        "start": start_sample / SAMPLE_RATE,
                        "end": end_sample / SAMPLE_RATE
                    })

                    import os
                    os.unlink(tmp_win.name)

            # Cluster embeddings to identify speakers
            embeddings_matrix = np.array(embeddings_list)

            # Normalize embeddings for cosine similarity
            from sklearn.preprocessing import normalize
            embeddings_matrix = normalize(embeddings_matrix)

            # Determine optimal number of speakers using multiple metrics with complexity penalty
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            from sklearn.cluster import AgglomerativeClustering

            # Try different numbers of speakers and pick the best
            max_speakers_to_try = min(DIARIZATION_MAX_SPEAKERS, len(embeddings_matrix) // 5)
            min_speakers_to_try = DIARIZATION_MIN_SPEAKERS

            best_score = -np.inf
            best_n_speakers = 1
            best_labels = None

            # Only try clustering if we have enough segments
            if len(embeddings_matrix) >= 10 and max_speakers_to_try >= 2:
                for n in range(min_speakers_to_try, min(max_speakers_to_try + 1, 6)):  # Cap at 5 speakers
                    try:
                        # Use AgglomerativeClustering with cosine distance (better for speaker embeddings)
                        clustering = AgglomerativeClustering(
                            n_clusters=n,
                            metric='cosine',
                            linkage='average'
                        )
                        labels = clustering.fit_predict(embeddings_matrix)

                        # Calculate silhouette score (measures cluster quality)
                        silhouette = silhouette_score(embeddings_matrix, labels, metric='cosine')

                        # Calculate Calinski-Harabasz score (higher is better, measures cluster separation)
                        calinski = calinski_harabasz_score(embeddings_matrix, labels)

                        # Normalize calinski to 0-1 range (approximately)
                        calinski_norm = calinski / (calinski + 100)

                        # Apply complexity penalty: prefer fewer speakers (BIC-inspired)
                        # Penalty increases with number of speakers
                        complexity_penalty = 0.15 * (n - 1)  # Each additional speaker costs 0.15 points

                        # Combined score: weighted average of silhouette and calinski, minus penalty
                        combined_score = (0.6 * silhouette + 0.4 * calinski_norm) - complexity_penalty

                        print(f"Trying {n} speakers: silhouette={silhouette:.3f}, calinski_norm={calinski_norm:.3f}, penalty={complexity_penalty:.3f}, combined={combined_score:.3f}")

                        if combined_score > best_score:
                            best_score = combined_score
                            best_n_speakers = n
                            best_labels = labels
                    except Exception as e:
                        print(f"Failed to cluster with {n} speakers: {e}")
                        continue

                if best_labels is not None:
                    speaker_labels = best_labels
                    n_speakers = best_n_speakers
                    print(f"Selected {n_speakers} speakers with combined score {best_score:.3f}")
                else:
                    # Fallback: assume single speaker
                    n_speakers = 1
                    speaker_labels = np.zeros(len(embeddings_matrix), dtype=int)
            else:
                # Too few segments, assume single speaker
                n_speakers = 1
                speaker_labels = np.zeros(len(embeddings_matrix), dtype=int)

            # Create speaker timeline by merging consecutive windows with same speaker
            speaker_segments = []
            current_speaker = speaker_labels[0]
            current_start = timestamps_list[0]["start"]
            current_end = timestamps_list[0]["end"]

            for i in range(1, len(speaker_labels)):
                if speaker_labels[i] == current_speaker:
                    # Extend current segment
                    current_end = timestamps_list[i]["end"]
                else:
                    # Save previous segment and start new one
                    speaker_segments.append({
                        "start": current_start,
                        "end": current_end,
                        "speaker": int(current_speaker)
                    })
                    current_speaker = speaker_labels[i]
                    current_start = timestamps_list[i]["start"]
                    current_end = timestamps_list[i]["end"]

            # Add final segment
            speaker_segments.append({
                "start": current_start,
                "end": current_end,
                "speaker": int(current_speaker)
            })

            # Clean up temp file
            import os
            os.unlink(audio_path)

            print(f"Diarization complete: {len(speaker_segments)} speaker segments, {n_speakers} speakers")

            return speaker_segments

        except Exception as e:
            import traceback
            print(f"Diarization error: {e}")
            print(traceback.format_exc())
            return []


@app.cls(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/models": volume},
    scaledown_window=10,  # CPU-only container; aggressively terminate since voice saves are infrequent
)
class VoiceStorage:
    """Manage saved voices in Modal Volume."""

    # UUID format regex for voice_id validation
    _UUID_RE = __import__("re").compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    )

    def _validate_voice_id(self, voice_id: str) -> bool:
        """Validate voice_id is a proper UUID to prevent path traversal."""
        return bool(self._UUID_RE.match(voice_id))

    def _get_voices_dir(self) -> Path:
        """Get the voices directory path."""
        voices_dir = Path("/models") / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        return voices_dir

    def _get_index_path(self) -> Path:
        """Get the index file path."""
        return self._get_voices_dir() / VOICES_INDEX_FILE

    def _load_index(self) -> list:
        """Load the voices index."""
        index_path = self._get_index_path()
        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)
        return []

    def _save_index(self, index: list):
        """Save the voices index."""
        index_path = self._get_index_path()
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        volume.commit()

    @modal.method()
    def list_voices(self) -> list:
        """List all saved voices."""
        return self._load_index()

    @modal.method()
    def get_voice(self, voice_id: str) -> Dict[str, Any]:
        """Get a voice by ID including audio bytes."""
        if not self._validate_voice_id(voice_id):
            return {"success": False, "error": "Invalid voice ID format"}

        voices_dir = self._get_voices_dir()
        voice_dir = voices_dir / voice_id

        # Verify resolved path stays within voices_dir (defense in depth)
        if not voice_dir.resolve().is_relative_to(voices_dir.resolve()):
            return {"success": False, "error": "Invalid voice ID"}

        if not voice_dir.exists():
            return {"success": False, "error": "Voice not found"}

        # Load metadata
        metadata_path = voice_dir / "metadata.json"
        if not metadata_path.exists():
            return {"success": False, "error": "Voice metadata not found"}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load audio
        audio_path = voice_dir / "ref_audio.wav"
        if not audio_path.exists():
            return {"success": False, "error": "Voice audio not found"}

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        return {
            "success": True,
            "metadata": metadata,
            "audio_bytes": audio_bytes,
        }

    @modal.method()
    def save_voice(
        self,
        voice_id: str,
        name: str,
        ref_text: str,
        language: str,
        prompt_bytes: bytes,
    ) -> Dict[str, Any]:
        """Save a new voice with pre-computed voice prompt.
        
        Args:
            voice_id: UUID for the voice
            name: User-friendly name
            ref_text: Reference audio transcription (for info only)
            language: Voice language
            prompt_bytes: Pre-computed torch-serialized VoiceClonePromptItem bytes
        
        Returns:
            Success status with voice_id
        """
        from datetime import datetime

        if not self._validate_voice_id(voice_id):
            return {"success": False, "error": "Invalid voice ID format"}

        # Check max voices limit
        index = self._load_index()
        if len(index) >= MAX_SAVED_VOICES:
            return {"success": False, "error": f"Maximum {MAX_SAVED_VOICES} voices reached"}

        # Check for duplicate name
        if any(v["name"].lower() == name.lower() for v in index):
            return {"success": False, "error": f"Voice with name '{name}' already exists"}

        # Create voice directory
        voices_dir = self._get_voices_dir()
        voice_dir = voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Save voice prompt (no audio stored — prompt contains embeddings only)
        prompt_path = voice_dir / "prompt.pt"
        with open(prompt_path, "wb") as f:
            f.write(prompt_bytes)

        # Save metadata
        metadata = {
            "id": voice_id,
            "name": name,
            "ref_text": ref_text,
            "language": language,
            "created_at": datetime.utcnow().isoformat(),
        }
        metadata_path = voice_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update index
        index.append({
            "id": voice_id,
            "name": name,
            "language": language,
            "ref_text": ref_text,
            "created_at": metadata["created_at"],
        })
        self._save_index(index)
        volume.commit()

        print(f"Voice saved: {name} ({voice_id})")
        return {"success": True, "voice_id": voice_id}

    @modal.method()
    def get_voice_with_prompt(self, voice_id: str) -> Dict[str, Any]:
        """Get a voice by ID including metadata and cached voice prompt.

        Returns metadata + prompt_bytes. For legacy voices created before prompt
        caching, this may also include audio_bytes so callers can auto-migrate by
        computing and saving prompt.pt on first synthesis.
        """
        if not self._validate_voice_id(voice_id):
            return {"success": False, "error": "Invalid voice ID format"}

        voices_dir = self._get_voices_dir()
        voice_dir = voices_dir / voice_id

        if not voice_dir.resolve().is_relative_to(voices_dir.resolve()):
            return {"success": False, "error": "Invalid voice ID"}

        if not voice_dir.exists():
            return {"success": False, "error": "Voice not found"}

        metadata_path = voice_dir / "metadata.json"
        if not metadata_path.exists():
            return {"success": False, "error": "Voice metadata not found"}

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load cached voice prompt
        prompt_path = voice_dir / "prompt.pt"
        prompt_bytes = None
        if prompt_path.exists():
            with open(prompt_path, "rb") as f:
                prompt_bytes = f.read()

        # Backward compatibility: old voices may still have ref_audio.wav.
        audio_bytes = None
        audio_path = voice_dir / "ref_audio.wav"
        if audio_path.exists():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

        # If neither prompt nor audio exists, the voice is unusable.
        if prompt_bytes is None and audio_bytes is None:
            return {"success": False, "error": "Voice prompt not found"}

        return {
            "success": True,
            "metadata": metadata,
            "prompt_bytes": prompt_bytes,
            "audio_bytes": audio_bytes,
        }

    @modal.method()
    def save_voice_prompt(self, voice_id: str, prompt_bytes: bytes) -> Dict[str, Any]:
        """Save precomputed voice prompt bytes for faster future synthesis."""
        if not self._validate_voice_id(voice_id):
            return {"success": False, "error": "Invalid voice ID format"}

        voices_dir = self._get_voices_dir()
        voice_dir = voices_dir / voice_id

        if not voice_dir.resolve().is_relative_to(voices_dir.resolve()):
            return {"success": False, "error": "Invalid voice ID"}

        if not voice_dir.exists():
            return {"success": False, "error": "Voice not found"}

        prompt_path = voice_dir / "prompt.pt"
        with open(prompt_path, "wb") as f:
            f.write(prompt_bytes)
        volume.commit()
        print(f"Voice prompt cached: {voice_id}")
        return {"success": True}

    @modal.method()
    def delete_voice(self, voice_id: str) -> Dict[str, Any]:
        """Delete a voice."""
        import shutil

        if not self._validate_voice_id(voice_id):
            return {"success": False, "error": "Invalid voice ID format"}

        voices_dir = self._get_voices_dir()
        voice_dir = voices_dir / voice_id

        # Verify resolved path stays within voices_dir (defense in depth)
        if not voice_dir.resolve().is_relative_to(voices_dir.resolve()):
            return {"success": False, "error": "Invalid voice ID"}

        if not voice_dir.exists():
            return {"success": False, "error": "Voice not found"}

        # Remove directory
        shutil.rmtree(voice_dir)

        # Update index
        index = self._load_index()
        index = [v for v in index if v["id"] != voice_id]
        self._save_index(index)

        print(f"Voice deleted: {voice_id}")
        return {"success": True}


@app.cls(
    image=qwen_tts_image,
    gpu=TTS_MODELS["qwen"]["gpu_type"],
    scaledown_window=30,  # Aggressive: terminate 30s after last request (saves cost for personal use)
    timeout=TTS_TIMEOUT,
    memory=TTS_MODELS["qwen"]["memory_mb"],
    volumes={"/models": volume},
)
class Qwen3TTSVoiceCloner:
    """Qwen3-TTS model for voice cloning."""

    @modal.enter()
    def load_model(self):
        """Load Qwen3-TTS model once per container."""
        import torch
        import time
        import os
        import inspect
        import warnings

        model_config = TTS_MODELS["qwen"]
        start_time = time.time()
        print(f"Loading Qwen3-TTS model: {model_config['model_id']}...")

        def _cache_stats(path: str) -> tuple[int, int]:
            """Return (file_count, total_bytes) for a cache path."""
            if not os.path.exists(path):
                return 0, 0
            file_count = 0
            total_bytes = 0
            for root, _dirs, files in os.walk(path):
                for filename in files:
                    file_count += 1
                    file_path = os.path.join(root, filename)
                    try:
                        total_bytes += os.path.getsize(file_path)
                    except OSError:
                        continue
            return file_count, total_bytes

        hf_home = os.getenv("HF_HOME", "/models")
        hf_cache_path = os.path.join(hf_home, "hub")
        before_files, before_bytes = _cache_stats(hf_cache_path)
        print(
            f"Qwen load cache before: path={hf_cache_path}, files={before_files}, "
            f"size_mb={before_bytes / (1024 * 1024):.1f}"
        )

        from qwen_tts import Qwen3TTSModel

        # Qwen3-TTS forwards kwargs through to AutoModel.from_pretrained(...).
        # On this pinned commit + transformers combination, FA2 can still emit
        # "dtype not specified" unless torch_dtype is passed explicitly.
        model_kwargs_dtype = {
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
        }
        model_kwargs_torch_dtype = {
            "device_map": "cuda:0",
            "torch_dtype": torch.bfloat16,
        }

        from_pretrained_sig = inspect.signature(Qwen3TTSModel.from_pretrained)
        print(f"Qwen3TTSModel.from_pretrained signature: {from_pretrained_sig}")

        try:
            import flash_attn  # noqa: F401
            print("flash-attn detected; trying attn_implementation=flash_attention_2")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
                        category=UserWarning,
                    )
                    self.model = Qwen3TTSModel.from_pretrained(
                        model_config["model_id"],
                        attn_implementation="flash_attention_2",
                        **model_kwargs_torch_dtype,
                    )
                print("flash-attn enabled with torch_dtype=bfloat16 (compat path)")
            except Exception as flash_td_error:
                print(f"flash-attn + torch_dtype path failed ({flash_td_error}); retrying with dtype")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_config["model_id"],
                    attn_implementation="flash_attention_2",
                    **model_kwargs_dtype,
                )
                print("flash-attn enabled with dtype=bfloat16")
        except ImportError:
            print("flash-attn not installed; loading Qwen3-TTS without flash attention")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_config["model_id"],
                    **model_kwargs_dtype,
                )
            except TypeError as td_error:
                print(f"dtype path failed without flash-attn ({td_error}); retrying with torch_dtype")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_config["model_id"],
                    **model_kwargs_torch_dtype,
                )
        except Exception as flash_type_error:
            print(f"flash-attn path failed ({flash_type_error}); falling back to default attention")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_config["model_id"],
                    **model_kwargs_dtype,
                )
            except TypeError as fallback_td_error:
                print(f"fallback dtype path failed ({fallback_td_error}); retrying with torch_dtype")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_config["model_id"],
                    **model_kwargs_torch_dtype,
                )

        # Ensure inference-only behavior.
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Persist newly downloaded HF artifacts for reuse across cold starts.
        try:
            volume.commit()
            print("Committed /models volume after Qwen model load")
        except Exception as commit_error:
            print(f"Volume commit skipped/failed: {commit_error}")

        after_files, after_bytes = _cache_stats(hf_cache_path)
        print(
            f"Qwen load cache after: path={hf_cache_path}, files={after_files}, "
            f"size_mb={after_bytes / (1024 * 1024):.1f}, "
            f"delta_mb={(after_bytes - before_bytes) / (1024 * 1024):.1f}"
        )

        load_time = time.time() - start_time
        print(f"Qwen3-TTS model loaded in {load_time:.2f}s")

    @modal.method()
    def compute_voice_prompt(
        self,
        ref_audio_bytes: bytes,
        ref_text: str,
        language: str = "English",
    ) -> bytes:
        """
        Pre-compute and serialize a voice prompt for reuse.

        Computes VoiceClonePromptItem (embeddings + speech tokens) from reference audio
        and transcription. Result is torch.save'd to bytes for storage in Modal Volume.

        Args:
            ref_audio_bytes: Reference audio bytes (WAV format, 24kHz mono)
            ref_text: Transcription of the reference audio (required for ICL mode)
            language: Voice language (for logging)

        Returns:
            Serialized bytes of List[VoiceClonePromptItem] ready to cache and reuse
        """
        import io
        import tempfile
        import torch

        try:
            # Write reference audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                tmp_ref.write(ref_audio_bytes)
                ref_audio_path = tmp_ref.name

            # Compute voice prompt with ICL mode (requires ref_text)
            effective_ref_text = ref_text.strip() if ref_text else None
            voice_prompt_items = self.model.create_voice_clone_prompt(
                ref_audio_path, effective_ref_text
            )

            # Serialize for storage
            buf = io.BytesIO()
            torch.save(voice_prompt_items, buf)
            prompt_bytes = buf.getvalue()
            buf.close()

            print(
                f"Voice prompt computed: language={language} "
                f"mode={'icl' if effective_ref_text else 'x_vector_only'} "
                f"size_bytes={len(prompt_bytes)}"
            )
            return prompt_bytes

        finally:
            # Clean up temp file
            import os
            if 'ref_audio_path' in locals() and os.path.exists(ref_audio_path):
                try:
                    os.unlink(ref_audio_path)
                except OSError:
                    pass

    @modal.method()
    def generate_voice_clone_stream(
        self,
        ref_audio_bytes: Optional[bytes],
        ref_text: str,
        target_text: str,
        language: str,
        request_trace: str = "",
        voice_prompt_bytes: Optional[bytes] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream live synthesis status lines and final result.

        This wraps generate_voice_clone and forwards log/status lines while the
        synthesis worker is running, then yields a final `complete` event with
        the same result payload as generate_voice_clone.
        """
        import queue
        import threading
        import sys
        import time

        log_queue: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, str] = {}

        class _QueueWriter:
            def __init__(self):
                self._buf = ""

            def write(self, text: str) -> int:
                if not text:
                    return 0
                self._buf += text
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        log_queue.put(line)
                return len(text)

            def flush(self) -> None:
                if self._buf.strip():
                    log_queue.put(self._buf.strip())
                self._buf = ""

        def _worker() -> None:
            writer = _QueueWriter()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = writer
            sys.stderr = writer
            try:
                result_holder["result"] = self._generate_voice_clone_impl(
                    ref_audio_bytes,
                    ref_text,
                    target_text,
                    language,
                    request_trace,
                    voice_prompt_bytes,
                )
            except Exception as e:
                error_holder["error"] = str(e)
            finally:
                try:
                    writer.flush()
                except Exception:
                    pass
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        yield {
            "type": "status",
            "stage": "start",
            "message": f"[tts:{request_trace or 'modal'}] synthesis started",
        }

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        last_heartbeat = time.monotonic()
        heartbeat_seconds = 8.0

        while worker.is_alive() or not log_queue.empty():
            try:
                line = log_queue.get(timeout=0.25)
                yield {
                    "type": "status",
                    "stage": "progress",
                    "message": line,
                }
                last_heartbeat = time.monotonic()
            except queue.Empty:
                now = time.monotonic()
                if worker.is_alive() and (now - last_heartbeat) >= heartbeat_seconds:
                    yield {
                        "type": "status",
                        "stage": "heartbeat",
                        "message": f"[tts:{request_trace or 'modal'}] still processing",
                    }
                    last_heartbeat = now
                continue

        if "error" in error_holder:
            yield {
                "type": "error",
                "error": error_holder["error"],
            }
            return

        result = result_holder.get("result")
        if result is None:
            yield {
                "type": "error",
                "error": "Synthesis worker ended without a result",
            }
            return

        yield {
            "type": "complete",
            "result": result,
        }

    @modal.method()
    def generate_voice_clone(
        self,
        ref_audio_bytes: Optional[bytes],
        ref_text: str,
        target_text: str,
        language: str,
        request_trace: str = "",
        voice_prompt_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Generate audio with cloned voice."""
        return self._generate_voice_clone_impl(
            ref_audio_bytes,
            ref_text,
            target_text,
            language,
            request_trace,
            voice_prompt_bytes,
        )

    def _generate_voice_clone_impl(
        self,
        ref_audio_bytes: Optional[bytes],
        ref_text: str,
        target_text: str,
        language: str,
        request_trace: str = "",
        voice_prompt_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Generate audio with cloned voice.

        Args:
            ref_audio_bytes: Reference audio bytes (WAV format); may be None when
                voice_prompt_bytes is provided (cached prompt already encodes the speaker).
            ref_text: Transcription of the reference audio. Empty string triggers
                x_vector_only_mode (speaker-embedding-only, no ICL codes).
            target_text: Text to synthesize with cloned voice
            language: Target language
            request_trace: Optional trace ID for correlated logging.
            voice_prompt_bytes: Pre-serialized List[VoiceClonePromptItem] from a previous
                call. When present, ref_audio/ref_text processing is bypassed entirely.

        Returns:
            Dict with audio_bytes, metadata, and prompt_bytes (populated only when a new
            voice prompt was computed; None when an existing cache was used).
        """
        import io
        import soundfile as sf
        import tempfile
        import time
        import os
        import re
        import numpy as np
        import torch
        import pysbd

        def _prepare_text_for_tts_document(text: str, language: str = "en") -> str:
            """
            Document-level text preparation before chunking.

            This stage mirrors the first half of commercial TTS pipelines: clean the
            source, expand things that materially affect spoken length, then hand the
            chunker text that already resembles what will be spoken.
            """
            import unicodedata
            import html as _html
            from num2words import num2words

            lang_map = {
                "english": "en", "spanish": "es", "french": "fr", "german": "de",
                "russian": "ru", "portuguese": "pt", "italian": "it",
                "chinese": "zh", "japanese": "ja", "korean": "ko",
            }
            nw_lang = lang_map.get(language.lower(), "en")

            # ------------------------------------------------------------------ #
            # 1. Unicode + smart punctuation
            # ------------------------------------------------------------------ #
            text = unicodedata.normalize("NFKC", text)
            text = _html.unescape(text)
            for src, dst in {
                "\u2018": "'", "\u2019": "'",
                "\u201C": '"', "\u201D": '"',
                "\u2013": ", ",
                "\u2014": ", ",
                "\u2026": "...",
                "\u00B7": " ",
                "\u2022": "",
                "\u00A0": " ",
                "\u00AD": "",      # soft hyphen
            }.items():
                text = text.replace(src, dst)

            # ------------------------------------------------------------------ #
            # 2. Strip HTML
            # ------------------------------------------------------------------ #
            text = re.sub(r"<[^>]+>", " ", text)

            # ------------------------------------------------------------------ #
            # 2b. Strip Markdown
            # ------------------------------------------------------------------ #
            text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
            text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
            text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)
            text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
            text = re.sub(r"`([^`]+)`", r"\1", text)
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
            text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", " ", text)
            text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
            text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
            text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
            text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

            # ------------------------------------------------------------------ #
            # 3. URLs and emails
            # ------------------------------------------------------------------ #
            text = re.sub(r"https?://[^\s,;)\"']+", "link", text)

            def _expand_alnum_token(token: str) -> str:
                """Expand mixed alphanumeric tokens into speakable chunks."""
                pieces: list[str] = []
                for part in re.findall(r"[A-Za-z]+|\d+", token):
                    if part.isdigit():
                        pieces.extend(num2words(int(d), lang=nw_lang) for d in part)
                    else:
                        pieces.append(part)
                return " ".join(pieces) if pieces else token

            def _expand_email(m: re.Match) -> str:
                raw = m.group(0)
                local, domain = raw.split("@", 1)

                def _expand_local(local_part: str) -> str:
                    chunks: list[str] = []
                    token = ""
                    for ch in local_part:
                        if ch.isalnum():
                            token += ch
                            continue
                        if token:
                            chunks.append(_expand_alnum_token(token))
                            token = ""
                        if ch == ".":
                            chunks.append("dot")
                        elif ch == "_":
                            chunks.append("underscore")
                        elif ch == "-":
                            chunks.append("dash")
                        elif ch == "+":
                            chunks.append("plus")
                    if token:
                        chunks.append(_expand_alnum_token(token))
                    return " ".join(chunks)

                def _expand_domain(domain_part: str) -> str:
                    labels = [lbl for lbl in domain_part.split(".") if lbl]
                    spoken_labels = []
                    for label in labels:
                        label_chunks: list[str] = []
                        token = ""
                        for ch in label:
                            if ch.isalnum():
                                token += ch
                                continue
                            if token:
                                label_chunks.append(_expand_alnum_token(token))
                                token = ""
                            if ch == "-":
                                label_chunks.append("dash")
                            elif ch == "_":
                                label_chunks.append("underscore")
                        if token:
                            label_chunks.append(_expand_alnum_token(token))
                        spoken_labels.append(" ".join(label_chunks) if label_chunks else label)
                    return " dot ".join(spoken_labels)

                return f"{_expand_local(local)} at {_expand_domain(domain)}"

            text = re.sub(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                _expand_email,
                text,
            )

            # ------------------------------------------------------------------ #
            # 4. Social handles and hashtags
            # ------------------------------------------------------------------ #
            text = re.sub(r"@([A-Za-z0-9_]+)", lambda m: "at " + m.group(1), text)
            text = re.sub(r"#([A-Za-z0-9_]+)", lambda m: m.group(1), text)

            # ------------------------------------------------------------------ #
            # 5. Repeated punctuation collapse
            # ------------------------------------------------------------------ #
            text = re.sub(r"!{2,}", "!", text)
            text = re.sub(r"\?{2,}", "?", text)
            text = re.sub(r"\.{3,}", "...", text)   # keep ellipsis as exactly 3
            text = re.sub(r",{2,}", ",", text)

            # ------------------------------------------------------------------ #
            # 6. Abbreviation / honorific expansion (English)
            # ------------------------------------------------------------------ #
            ABBREV_EN = {
                r"\bMr\.": "Mister",     r"\bMrs\.": "Missus",
                r"\bMs\.":  "Miss",      r"\bDr\.":  "Doctor",
                r"\bProf\.":"Professor", r"\bSgt\.": "Sergeant",
                r"\bCpl\.": "Corporal",  r"\bLt\.":  "Lieutenant",
                r"\bCol\.": "Colonel",   r"\bGen\.": "General",
                r"\bSt\.":  "Saint",     r"\bAve\.": "Avenue",
                r"\bBlvd\.":"Boulevard", r"\bDept\.":"Department",
                r"\bvs\.":  "versus",    r"\bvs\b":  "versus",
                r"\betc\.": "et cetera", r"\be\.g\.":"for example",
                r"\bi\.e\.":"that is",   r"\bw/\b":  "with",
                r"\bw/o\b": "without",   r"\b&\b":   "and",
                r"\b%\b":   "percent",   r"\bapprox\.": "approximately",
                r"\bappt\.":"appointment",r"\binfo\.": "information",
                r"\bFYI\b": "for your information",
                r"\bIMHO\b":"in my humble opinion",
                r"\bIRL\b": "in real life",
                r"\bAKA\b": "also known as",
                r"\bASAP\b":"as soon as possible",
                r"\bTBD\b": "to be determined",
                r"\bTBA\b": "to be announced",
                r"\bRSVP\b":"please respond",
            }
            if nw_lang == "en":
                for pattern, expansion in ABBREV_EN.items():
                    text = re.sub(pattern, expansion, text)

            # ------------------------------------------------------------------ #
            # 7. All-caps sentence de-casing
            # Common short words / known acronyms that STAY uppercase
            # ------------------------------------------------------------------ #
            KEEP_UPPER = {
                "I", "AM", "PM", "OK", "TV", "PC", "ID",
                "US", "UK", "EU", "UN", "AI", "ML", "IT",
            }
            def _decap_allcaps(m: re.Match) -> str:
                word = m.group(0)
                if word in KEEP_UPPER:
                    return word
                # Already handled by acronym spacing step below if len >= 3
                return word.capitalize()
            # Only target ALL-CAPS words of 4+ chars that aren't already known acronyms
            text = re.sub(r"\b[A-Z]{4,}\b", _decap_allcaps, text)

            # ------------------------------------------------------------------ #
            # 8. Roman numerals  (I–XXXIX, conservative to avoid false positives)
            # ------------------------------------------------------------------ #
            ROMAN = {
                "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
                "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
                "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
                "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
                "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30,
                "XXXI": 31, "XXXII": 32, "XXXIII": 33, "XXXIV": 34, "XXXV": 35,
                "XXXVI": 36, "XXXVII": 37, "XXXVIII": 38, "XXXIX": 39,
                "XL": 40, "L": 50,
            }
            # Only expand when preceded by a word like Chapter/Part/Section/Act/Volume/Book
            roman_pat = r"\b(Chapter|Part|Section|Act|Volume|Book|Episode|Phase|Stage)\s+(" + \
                        "|".join(sorted(ROMAN.keys(), key=len, reverse=True)) + r")\b"
            def _expand_roman(m: re.Match) -> str:
                try:
                    return f"{m.group(1)} {num2words(ROMAN[m.group(2)], to='ordinal', lang=nw_lang)}"
                except Exception:
                    return m.group(0)
            text = re.sub(roman_pat, _expand_roman, text)

            # ------------------------------------------------------------------ #
            # 9. Phone numbers
            # ------------------------------------------------------------------ #
            def _expand_phone(m: re.Match) -> str:
                digits = re.sub(r"\D", "", m.group(0))
                # Group: (NXX) NXX-XXXX or NXX-NXX-XXXX
                if len(digits) == 10:
                    groups = [digits[:3], digits[3:6], digits[6:]]
                elif len(digits) == 11 and digits[0] == "1":
                    groups = [digits[1:4], digits[4:7], digits[7:]]
                else:
                    return m.group(0)
                spoken = []
                for g in groups:
                    spoken.append(", ".join(num2words(int(d), lang=nw_lang) for d in g))
                return ", ".join(spoken)
            text = re.sub(
                r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
                _expand_phone, text,
            )

            # ------------------------------------------------------------------ #
            # 10. Fractions
            # ------------------------------------------------------------------ #
            FRAC_MAP = {
                "1/2": "one half",       "1/3": "one third",
                "2/3": "two thirds",     "1/4": "one quarter",
                "3/4": "three quarters", "1/5": "one fifth",
                "2/5": "two fifths",     "3/5": "three fifths",
                "4/5": "four fifths",    "1/8": "one eighth",
                "3/8": "three eighths",  "5/8": "five eighths",
                "7/8": "seven eighths",
            }
            for frac, spoken in FRAC_MAP.items():
                text = re.sub(r"(?<!\d)" + re.escape(frac) + r"(?!\d)", spoken, text)
            # Generic N/M not already caught
            def _expand_generic_frac(m: re.Match) -> str:
                try:
                    n, d = int(m.group(1)), int(m.group(2))
                    if d == 0:
                        return m.group(0)
                    numer = num2words(n, lang=nw_lang)
                    denom = num2words(d, to="ordinal", lang=nw_lang)
                    if n != 1:
                        denom += "s"
                    return f"{numer} {denom}"
                except Exception:
                    return m.group(0)
            text = re.sub(r"(?<!\d)(\d+)/(\d+)(?!\d)", _expand_generic_frac, text)

            # ------------------------------------------------------------------ #
            # 11. Units of measure
            # ------------------------------------------------------------------ #
            UNITS = {
                # Distance
                r"km\b": "kilometers",   r"mi\b": "miles",
                r"m\b":  "meters",       r"cm\b": "centimeters",
                r"mm\b": "millimeters",  r"ft\b": "feet",
                r"in\b": "inches",       r"yd\b": "yards",
                # Weight
                r"kg\b": "kilograms",    r"g\b":  "grams",
                r"mg\b": "milligrams",   r"lb\b": "pounds",
                r"lbs\b":"pounds",       r"oz\b": "ounces",
                # Volume
                r"L\b":  "liters",       r"mL\b": "milliliters",
                r"ml\b": "milliliters",  r"gal\b":"gallons",
                r"fl oz\b": "fluid ounces",
                # Temperature
                r"°F\b": "degrees Fahrenheit",
                r"°C\b": "degrees Celsius",
                r"°K\b": "Kelvin",
                # Speed
                r"mph\b": "miles per hour",
                r"kph\b": "kilometers per hour",
                r"km/h\b":"kilometers per hour",
                r"m/s\b": "meters per second",
                # Data
                r"KB\b": "kilobytes",    r"MB\b": "megabytes",
                r"GB\b": "gigabytes",    r"TB\b": "terabytes",
                r"Kb\b": "kilobits",     r"Mb\b": "megabits",
                r"Gb\b": "gigabits",
                # Time
                r"ms\b": "milliseconds", r"µs\b": "microseconds",
                r"ns\b": "nanoseconds",
                # Misc
                r"MHz\b":"megahertz",    r"GHz\b":"gigahertz",
                r"kHz\b":"kilohertz",    r"Hz\b": "hertz",
                r"kW\b": "kilowatts",    r"MW\b": "megawatts",
                r"W\b":  "watts",        r"V\b":  "volts",
                r"A\b":  "amps",
            }
            for unit_pat, unit_word in UNITS.items():
                text = re.sub(r"(\d)\s*" + unit_pat, r"\1 " + unit_word, text)

            # ------------------------------------------------------------------ #
            # 12. Currencies
            # ------------------------------------------------------------------ #
            def _expand_currency(m: re.Match) -> str:
                symbol = m.group(1)
                amount_str = m.group(2).replace(",", "")
                currency_map = {
                    "$": ("dollar", "dollars", "cent", "cents"),
                    "€": ("euro",   "euros",   "cent", "cents"),
                    "£": ("pound",  "pounds",  "penny","pence"),
                    "¥": ("yen",    "yen",     "sen",  "sen"),
                }
                names = currency_map.get(symbol, ("unit", "units", "subunit", "subunits"))
                try:
                    if "." in amount_str:
                        major, minor = amount_str.split(".", 1)
                        minor = minor[:2].ljust(2, "0")
                        major_val = int(major) if major else 0
                        minor_val = int(minor)
                        parts = []
                        if major_val or not minor_val:
                            parts.append(f"{num2words(major_val, lang=nw_lang)} "
                                         f"{names[0] if major_val == 1 else names[1]}")
                        if minor_val:
                            parts.append(f"{num2words(minor_val, lang=nw_lang)} "
                                         f"{names[2] if minor_val == 1 else names[3]}")
                        return " and ".join(parts)
                    else:
                        val = int(amount_str)
                        return f"{num2words(val, lang=nw_lang)} {names[0] if val == 1 else names[1]}"
                except Exception:
                    return m.group(0)
            text = re.sub(r"([$€£¥])([\d,]+(?:\.\d+)?)", _expand_currency, text)

            # ------------------------------------------------------------------ #
            # 13. Ordinals
            # ------------------------------------------------------------------ #
            def _expand_ordinal(m: re.Match) -> str:
                try:
                    return num2words(int(m.group(1)), to="ordinal", lang=nw_lang)
                except Exception:
                    return m.group(0)
            text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _expand_ordinal, text)

            # ------------------------------------------------------------------ #
            # 14. Dates
            # ------------------------------------------------------------------ #
            MONTHS = (
                "January|February|March|April|May|June|July|August|"
                "September|October|November|December|"
                "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
            )
            def _expand_named_date(month: str, day: str, year: str) -> str:
                day_word = num2words(int(day), to="ordinal", lang=nw_lang)
                return f"{month} {day_word}, {year}" if year and year.strip() else f"{month} {day_word}"
            text = re.sub(
                rf"\b({MONTHS})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b",
                lambda m: _expand_named_date(m.group(1), m.group(2), m.group(3) or ""),
                text,
            )
            def _expand_numeric_date(m: re.Match) -> str:
                try:
                    mo, day, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    month_names = ["January","February","March","April","May","June",
                                   "July","August","September","October","November","December"]
                    if 1 <= mo <= 12 and 1 <= day <= 31:
                        day_word = num2words(day, to="ordinal", lang=nw_lang)
                        return f"{month_names[mo-1]} {day_word}, {yr}"
                    return m.group(0)
                except Exception:
                    return m.group(0)
            text = re.sub(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b", _expand_numeric_date, text)

            # ------------------------------------------------------------------ #
            # 15. Times
            # ------------------------------------------------------------------ #
            def _expand_time(hour: int, minute: int, ampm: str) -> str:
                h_word = num2words(hour, lang=nw_lang)
                if minute == 0:
                    t_str = f"{h_word} o'clock"
                elif minute < 10:
                    t_str = f"{h_word} oh {num2words(minute, lang=nw_lang)}"
                else:
                    t_str = f"{h_word} {num2words(minute, lang=nw_lang)}"
                ampm = ampm.strip().upper().replace(".", "")
                if ampm == "AM":
                    t_str += " AM"
                elif ampm == "PM":
                    t_str += " PM"
                return t_str
            def _time_match(m: re.Match) -> str:
                try:
                    return _expand_time(int(m.group(1)), int(m.group(2) or 0), m.group(3) or "")
                except Exception:
                    return m.group(0)
            text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm|A\.M\.|P\.M\.)?\b", _time_match, text)
            def _short_time_match(m: re.Match) -> str:
                try:
                    return _expand_time(int(m.group(1)), 0, m.group(2))
                except Exception:
                    return m.group(0)
            text = re.sub(r"\b(\d{1,2})(am|pm|AM|PM)\b", _short_time_match, text)

            # ------------------------------------------------------------------ #
            # 16. Plain numbers → words
            # ------------------------------------------------------------------ #
            def _expand_number(m: re.Match) -> str:
                try:
                    raw = m.group(0).replace(",", "")
                    if "." in raw:
                        int_part, dec_part = raw.split(".", 1)
                        words = num2words(int(int_part), lang=nw_lang)
                        if dec_part and int(dec_part) != 0:
                            digit_words = " ".join(num2words(int(d), lang=nw_lang) for d in dec_part)
                            words += " point " + digit_words
                        return words
                    else:
                        val = int(raw)
                        if abs(val) > 999_999_999:
                            return m.group(0)
                        return num2words(val, lang=nw_lang)
                except Exception:
                    return m.group(0)
            text = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", _expand_number, text)

            # ------------------------------------------------------------------ #
            # ------------------------------------------------------------------ #
            # 17. Final whitespace cleanup
            # ------------------------------------------------------------------ #
            text = re.sub(r" {2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            return text

        def _polish_tts_chunk_text(text: str, language: str = "en") -> str:
            """
            Chunk-local polish immediately before synthesis.

            This is where pronunciation-sensitive cleanup lives: acronym expansion,
            small punctuation cleanup, and conservative homograph rewrites that rely
            on sentence-level context.
            """
            lang_map = {
                "english": "en", "spanish": "es", "french": "fr", "german": "de",
                "russian": "ru", "portuguese": "pt", "italian": "it",
                "chinese": "zh", "japanese": "ja", "korean": "ko",
            }
            nw_lang = lang_map.get(language.lower(), "en")
            keep_upper = {
                "I", "AM", "PM", "OK", "TV", "PC", "ID",
                "US", "UK", "EU", "UN", "AI", "ML", "IT",
            }

            def _match_case(source: str, replacement: str) -> str:
                if source.isupper():
                    return replacement.upper()
                if source and source[0].isupper():
                    return replacement.capitalize()
                return replacement

            text = re.sub(
                r"\b[A-Z]{3,}\b",
                lambda m: " ".join(m.group(0)) if m.group(0) not in keep_upper else m.group(0),
                text,
            )

            if nw_lang == "en":
                try:
                    import spacy

                    nlp = getattr(self, "_tts_en_nlp", None)
                    if nlp is None:
                        try:
                            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                            self._tts_en_nlp = nlp
                        except OSError:
                            self._tts_en_nlp = None
                            nlp = None

                    if nlp is not None:
                        doc = nlp(text)
                        tokens_out = []
                        for token in doc:
                            replacement = None
                            if token.lower_ == "read" and token.pos_ == "VERB" and token.tag_ in {"VBD", "VBN"}:
                                replacement = "red"
                            elif token.lower_ == "tear" and token.pos_ == "VERB":
                                replacement = "tare"
                            elif token.lower_ == "wind" and token.pos_ == "VERB":
                                replacement = "wynd"
                            elif token.lower_ == "minute" and token.pos_ == "ADJ":
                                replacement = "mynoot"
                            elif token.lower_ == "bass" and token.pos_ == "ADJ":
                                replacement = "base"
                            elif token.lower_ == "live" and token.pos_ == "ADJ":
                                replacement = "lyve"

                            if replacement is None:
                                tokens_out.append(token.text_with_ws)
                            else:
                                tokens_out.append(_match_case(token.text, replacement) + token.whitespace_)
                        text = "".join(tokens_out)
                except Exception as exc:
                    print(f"Chunk-level homograph polish skipped: {exc}")

            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            text = re.sub(r" {2,}", " ", text)
            return text.strip()

        # Boundary types — used later during stitching to pick the right gap length.
        _BOUNDARY_PARAGRAPH = "paragraph"
        _BOUNDARY_SENTENCE  = "sentence"
        _BOUNDARY_CLAUSE    = "clause"

        _CLAUSE_RE = re.compile(r"[,;:\u2013\u2014]$")  # , ; : – —

        def _word_count(s: str) -> int:
            return len(s.split())

        def _chunk_text(text: str, max_words: int, short_threshold: int) -> list[tuple[str, str]]:
            """
            Split *text* into (chunk_text, boundary_type) pairs.

            boundary_type is the boundary that *follows* the chunk:
              - 'paragraph' : chunk ends at a blank-line paragraph break
              - 'sentence'  : chunk ends at a sentence boundary (. ! ?)
              - 'clause'    : chunk ends at a clause boundary (, ; : – —)

            Strategy:
              1. Split on paragraph breaks first.
              2. Within each paragraph use pysbd for sentence boundaries.
              3. Merge consecutive sentences until max_words would be exceeded.
              4. Merge very-short isolated sentences (≤ short_threshold words)
                 with their neighbor so TTS gets enough context.
              5. If a single sentence exceeds max_words, split on clause
                 boundaries or, as a last resort, on word boundaries near
                 the middle of the sentence.
            """
            segmenter = pysbd.Segmenter(language="en", clean=False)
            result: list[tuple[str, str]] = []

            # Step 1 – paragraph split (two or more newlines)
            paragraphs = re.split(r"\n{2,}", text.strip())

            for para_idx, para in enumerate(paragraphs):
                para = " ".join(para.split())
                if not para:
                    continue

                # Step 2 – sentence segmentation inside paragraph
                raw_sentences = [s.strip() for s in segmenter.segment(para) if s.strip()]

                # Step 3 – greedy merge: accumulate sentences until max_words
                # Each item: (text, boundary_type)
                groups: list[tuple[str, str]] = []
                buf = ""
                for i, sent in enumerate(raw_sentences):
                    is_last = (i == len(raw_sentences) - 1)
                    candidate = (buf + " " + sent).strip() if buf else sent
                    if _word_count(candidate) <= max_words:
                        buf = candidate
                    else:
                        if buf:
                            # Determine trailing boundary type of what we're flushing
                            btype = _BOUNDARY_CLAUSE if _CLAUSE_RE.search(buf.rstrip()) else _BOUNDARY_SENTENCE
                            groups.append((buf, btype))
                        buf = sent
                if buf:
                    btype = _BOUNDARY_PARAGRAPH if not is_last else _BOUNDARY_PARAGRAPH
                    groups.append((buf, _BOUNDARY_PARAGRAPH if para_idx < len(paragraphs) - 1 else _BOUNDARY_SENTENCE))

                # Step 4 – merge short sentences with a neighbor
                merged: list[tuple[str, str]] = []
                for text_g, btype_g in groups:
                    if merged and _word_count(text_g) <= short_threshold:
                        prev_text, prev_btype = merged[-1]
                        candidate = prev_text + " " + text_g
                        if _word_count(candidate) <= max_words:
                            # absorb into previous, keep previous boundary or upgrade
                            merged[-1] = (candidate, btype_g)
                            continue
                    merged.append((text_g, btype_g))

                # Second pass: try merging short groups with next neighbor
                merged2: list[tuple[str, str]] = []
                i = 0
                while i < len(merged):
                    text_g, btype_g = merged[i]
                    if _word_count(text_g) <= short_threshold and i + 1 < len(merged):
                        next_text, next_btype = merged[i + 1]
                        candidate = text_g + " " + next_text
                        if _word_count(candidate) <= max_words:
                            merged2.append((candidate, next_btype))
                            i += 2
                            continue
                    merged2.append((text_g, btype_g))
                    i += 1
                merged = merged2

                # Step 5 – split any group that still exceeds max_words
                final_para: list[tuple[str, str]] = []
                for text_g, btype_g in merged:
                    if _word_count(text_g) <= max_words:
                        final_para.append((text_g, btype_g))
                        continue
                    # Try clause boundaries first
                    clause_parts = re.split(r"(?<=[,;:\u2013\u2014])\s+", text_g)
                    sub_buf = ""
                    for j, part in enumerate(clause_parts):
                        candidate = (sub_buf + " " + part).strip() if sub_buf else part
                        if _word_count(candidate) <= max_words:
                            sub_buf = candidate
                        else:
                            if sub_buf:
                                final_para.append((sub_buf, _BOUNDARY_CLAUSE))
                            sub_buf = part
                        if j == len(clause_parts) - 1 and sub_buf:
                            # Last part carries the original boundary type
                            if _word_count(sub_buf) <= max_words:
                                final_para.append((sub_buf, btype_g))
                            else:
                                # Hard word-boundary split near the middle
                                words = sub_buf.split()
                                half = max(1, len(words) // 2)
                                final_para.append((" ".join(words[:half]), _BOUNDARY_CLAUSE))
                                final_para.append((" ".join(words[half:]), btype_g))
                            sub_buf = ""
                result.extend(final_para)

            return result

        def _normalize_wavs(wavs_obj):
            """Normalize model output into a list of 1-D numpy arrays."""
            if isinstance(wavs_obj, np.ndarray):
                if wavs_obj.ndim == 1:
                    return [wavs_obj]
                if wavs_obj.ndim == 2:
                    return [wavs_obj[i] for i in range(wavs_obj.shape[0])]
            if isinstance(wavs_obj, (list, tuple)):
                return list(wavs_obj)
            raise TypeError(f"Unexpected wav output type: {type(wavs_obj)}")

        def _rms_frames(wav: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
            """Compute per-frame RMS energy."""
            import librosa
            return librosa.feature.rms(y=wav, frame_length=frame_len, hop_length=hop_len)[0]

        def _trim_chunk_edges(wav: np.ndarray, sr: int) -> np.ndarray:
            """Trim near-silent edges using peak-relative RMS (adapts to voice loudness)."""
            import librosa
            if wav.size == 0:
                return wav

            max_trim_samples = int((TTS_EDGE_TRIM_MS / 1000.0) * sr)
            if max_trim_samples <= 0:
                return wav

            frame_len = int(0.030 * sr)   # 30ms frames
            hop_len   = int(0.010 * sr)   # 10ms hop

            try:
                rms = _rms_frames(wav, frame_len, hop_len)
                peak_rms = float(np.max(rms))
                if peak_rms == 0:
                    return wav
                threshold = peak_rms * (10 ** (TTS_EDGE_SILENCE_THRESHOLD_DB / 20.0))

                # Left trim
                max_trim_frames = int(max_trim_samples / hop_len)
                left_frames = rms[:max_trim_frames]
                above = np.where(left_frames > threshold)[0]
                trim_left_frames = int(above[0]) if above.size else 0
                trim_left = trim_left_frames * hop_len

                # Right trim (reverse the tail)
                right_frames = rms[-max_trim_frames:]
                above_r = np.where(right_frames > threshold)[0]
                trim_right_frames = (len(right_frames) - int(above_r[-1]) - 1) if above_r.size else 0
                trim_right = trim_right_frames * hop_len

                if trim_left + trim_right >= wav.size:
                    return wav
                end = wav.size - trim_right if trim_right > 0 else wav.size
                return wav[trim_left:end]
            except Exception:
                return wav

        def _limit_silence_gaps(wav: np.ndarray, sr: int, max_silence_ms: int) -> np.ndarray:
            """Cap any internal silence longer than max_silence_ms to that duration."""
            import librosa
            if wav.size == 0 or max_silence_ms <= 0:
                return wav

            frame_len = int(0.030 * sr)
            hop_len   = int(0.010 * sr)

            try:
                rms = _rms_frames(wav, frame_len, hop_len)
                peak_rms = float(np.max(rms))
                if peak_rms == 0:
                    return wav
                threshold = peak_rms * (10 ** (TTS_EDGE_SILENCE_THRESHOLD_DB / 20.0))

                min_sil_frames = max_silence_ms / 10  # hop is 10ms
                is_silent = np.concatenate(([False], rms < threshold, [False]))
                diff = np.diff(is_silent.astype(int))
                starts = np.where(diff == 1)[0]
                ends   = np.where(diff == -1)[0]

                # Collect silence segments that exceed the limit
                excess: list[tuple[int,int,int]] = []  # (sample_start, sample_end, keep_samples)
                for s, e in zip(starts, ends):
                    dur_frames = e - s
                    if dur_frames > min_sil_frames:
                        s_sample = int(librosa.frames_to_time(s, sr=sr, hop_length=hop_len) * sr)
                        e_sample = int(librosa.frames_to_time(e, sr=sr, hop_length=hop_len) * sr)
                        s_sample = max(0, min(s_sample, wav.size))
                        e_sample = max(0, min(e_sample, wav.size))
                        keep = int((max_silence_ms / 1000.0) * sr)
                        excess.append((s_sample, e_sample, keep))

                if not excess:
                    return wav

                # Rebuild audio, trimming each excess silence to keep_samples centred
                pieces: list[np.ndarray] = []
                prev = 0
                for (s_samp, e_samp, keep) in excess:
                    pieces.append(wav[prev:s_samp])
                    mid = (s_samp + e_samp) // 2
                    half = keep // 2
                    pieces.append(wav[max(0, mid-half):min(wav.size, mid+half)])
                    prev = e_samp
                pieces.append(wav[prev:])
                return np.concatenate([p for p in pieces if p.size > 0])
            except Exception:
                return wav

        chunk_duration_samples: list[float] = []

        def _average_chunk_seconds() -> Optional[float]:
            if not chunk_duration_samples:
                return None
            if len(chunk_duration_samples) < TTS_MIN_BASELINE_CHUNKS:
                return None
            return float(sum(chunk_duration_samples) / len(chunk_duration_samples))

        def _is_slow_chunk(elapsed_s: float, avg_chunk_s: Optional[float]) -> bool:
            if avg_chunk_s is None:
                return elapsed_s >= TTS_SLOW_CHUNK_ABSOLUTE_S
            return elapsed_s >= max(TTS_SLOW_CHUNK_ABSOLUTE_S, avg_chunk_s * TTS_SLOW_CHUNK_MULTIPLIER)

        def _is_slow_batch(elapsed_s: float, avg_chunk_s: Optional[float], chunk_count: int) -> bool:
            if chunk_count <= 0:
                return False
            if avg_chunk_s is None:
                return elapsed_s >= TTS_SLOW_BATCH_ABSOLUTE_S
            expected_s = avg_chunk_s * chunk_count
            return elapsed_s >= max(TTS_SLOW_BATCH_ABSOLUTE_S, expected_s * TTS_SLOW_BATCH_MULTIPLIER)

        ref_audio_path = None
        try:
            start_time = time.time()
            t0 = time.perf_counter()
            trace = request_trace or "modal"

            def _status(message: str) -> None:
                print(f"[tts:{trace}] {message}")

            # ------------------------------------------------------------------
            # Voice prompt setup: use pre-serialised cache or compute fresh.
            # ------------------------------------------------------------------
            t_write_ref_start = time.perf_counter()
            new_prompt_bytes: Optional[bytes] = None  # set only when newly computed

            if voice_prompt_bytes is not None:
                # Fast path: deserialise cached VoiceClonePromptItem list.
                voice_prompt_items = torch.load(
                    io.BytesIO(voice_prompt_bytes),
                    map_location="cuda:0",
                    weights_only=False,
                )
                _status("voice-prompt cache_hit")
            else:
                # Write reference audio to temp file for speaker/ICL encoding.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                    tmp_ref.write(ref_audio_bytes)
                    ref_audio_path = tmp_ref.name
                # Compute voice prompt. Empty ref_text → x_vector_only_mode (speaker
                # embedding only; no ICL codes needed).
                effective_ref_text = ref_text.strip() if ref_text else None
                voice_prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio_path, effective_ref_text
                )
                # Serialise for caller to cache.
                buf = io.BytesIO()
                torch.save(voice_prompt_items, buf)
                new_prompt_bytes = buf.getvalue()
                mode_label = "icl" if effective_ref_text else "x_vector_only"
                _status(f"voice-prompt computed mode={mode_label}")

            t_write_ref = time.perf_counter() - t_write_ref_start

            _status(f"start chars={len(target_text)} language={language}")

            # Stage 1: document preparation before chunking.
            target_text = _prepare_text_for_tts_document(target_text, language)
            _status(f"document-prep chars={len(target_text)}")

            if TTS_ENABLE_CHUNKING:
                tagged_chunks = _chunk_text(target_text, TTS_CHUNK_MAX_WORDS, TTS_SHORT_SENTENCE_WORDS)
            else:
                tagged_chunks = [(" ".join(target_text.split()), _BOUNDARY_SENTENCE)]

            # Stage 2: chunk-local polish right before inference.
            chunks = [_polish_tts_chunk_text(c, language) for c, _ in tagged_chunks]
            boundaries = [b for _, b in tagged_chunks]

            batch_size = max(1, TTS_BATCH_SIZE if TTS_ENABLE_BATCHING else 1)
            total_chunks = len(chunks)
            planned_batch_sizes = [
                len(chunks[i:i + batch_size])
                for i in range(0, total_chunks, batch_size)
            ]
            total_batches = len(planned_batch_sizes)
            configured_concurrency = batch_size if TTS_ENABLE_BATCHING else 1
            print(
                "TTS chunking config: "
                f"enabled={TTS_ENABLE_CHUNKING}, "
                f"chunks={total_chunks}, "
                f"max_words={TTS_CHUNK_MAX_WORDS}, "
                f"batching={TTS_ENABLE_BATCHING}, "
                f"batch_size={batch_size}, "
                f"planned_batches={total_batches}, "
                f"planned_batch_sizes={planned_batch_sizes}, "
                f"configured_max_concurrency={configured_concurrency}"
            )
            _status(
                f"chunk-plan chunks={total_chunks} batch_size={batch_size} "
                f"planned_batches={total_batches}"
            )

            telemetry_events: list[dict[str, Any]] = []
            telemetry_events.append({
                "type": "chunk_plan",
                "chunks_total": total_chunks,
                "batch_size": batch_size,
                "batching_enabled": TTS_ENABLE_BATCHING,
                "planned_batches": total_batches,
                "planned_batch_sizes": planned_batch_sizes,
                "configured_max_concurrency": configured_concurrency,
            })

            # Inference mode avoids autograd overhead and keeps output quality unchanged.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            with torch.inference_mode():
                all_wavs = []
                sr = None
                batched_requests = 0
                batch_fallbacks = 0
                completed_chunks = 0
                effective_max_concurrency = 1
                fallback_chunks = 0
                for i in range(0, total_chunks, batch_size):
                    batch_texts = chunks[i:i + batch_size]
                    batch_index = (i // batch_size) + 1
                    batch_start_chunk = i + 1
                    batch_end_chunk = i + len(batch_texts)

                    _status(
                        f"batch-start index={batch_index}/{total_batches} "
                        f"chunks={batch_start_chunk}-{batch_end_chunk} "
                        f"size={len(batch_texts)}"
                    )

                    telemetry_events.append({
                        "type": "batch_start",
                        "batch_index": batch_index,
                        "batches_total": total_batches,
                        "batch_chunk_count": len(batch_texts),
                        "chunk_range": [batch_start_chunk, batch_end_chunk],
                    })

                    if TTS_ENABLE_BATCHING and len(batch_texts) > 1:
                        try:
                            batched_requests += 1
                            effective_max_concurrency = max(effective_max_concurrency, len(batch_texts))
                            telemetry_events.append({
                                "type": "batch_call",
                                "batch_index": batch_index,
                                "mode": "batched",
                                "in_flight_chunks": len(batch_texts),
                                "chunk_range": [batch_start_chunk, batch_end_chunk],
                            })
                            t_batch_start = time.perf_counter()
                            batch_wavs_obj, batch_sr = self.model.generate_voice_clone(
                                text=batch_texts,
                                language=language,
                                voice_clone_prompt=voice_prompt_items,
                            )
                            t_batch_elapsed = time.perf_counter() - t_batch_start
                            avg_chunk_s = _average_chunk_seconds()
                            if _is_slow_batch(t_batch_elapsed, avg_chunk_s, len(batch_texts)):
                                raise RuntimeError(
                                    f"slow_batch_elapsed={t_batch_elapsed:.2f}s avg_chunk={avg_chunk_s} "
                                    f"chunks={len(batch_texts)}"
                                )

                            per_chunk_elapsed = t_batch_elapsed / max(1, len(batch_texts))
                            chunk_duration_samples.extend([per_chunk_elapsed] * len(batch_texts))

                            batch_wavs = _normalize_wavs(batch_wavs_obj)
                            if len(batch_wavs) != len(batch_texts):
                                raise ValueError(
                                    f"Expected {len(batch_texts)} wavs from batched call, got {len(batch_wavs)}"
                                )
                            all_wavs.extend(batch_wavs)
                            sr = sr or batch_sr

                            for rel_idx in range(len(batch_wavs)):
                                completed_chunks += 1
                                progress_pct = (completed_chunks / total_chunks) * 100 if total_chunks else 100.0
                                _status(
                                    f"progress {completed_chunks}/{total_chunks} "
                                    f"({progress_pct:.1f}%) mode=batched"
                                )
                                telemetry_events.append({
                                    "type": "chunk_complete",
                                    "chunk_index": i + rel_idx + 1,
                                    "chunks_completed": completed_chunks,
                                    "chunks_total": total_chunks,
                                    "batch_index": batch_index,
                                    "mode": "batched",
                                    "in_flight_chunks": len(batch_texts),
                                })
                            continue
                        except Exception as batch_error:
                            batch_fallbacks += 1
                            _status(
                                f"batch-fallback index={batch_index} reason={batch_error}"
                            )
                            telemetry_events.append({
                                "type": "batch_fallback",
                                "batch_index": batch_index,
                                "chunk_range": [batch_start_chunk, batch_end_chunk],
                                "batch_chunk_count": len(batch_texts),
                                "reason": str(batch_error),
                            })
                            print(
                                "Batched Qwen generation unavailable for this request; "
                                f"falling back to per-chunk calls ({batch_error})"
                            )
                    else:
                        telemetry_events.append({
                            "type": "batch_call",
                            "batch_index": batch_index,
                            "mode": "single",
                            "in_flight_chunks": 1,
                            "chunk_range": [batch_start_chunk, batch_end_chunk],
                            "reason": "batching_disabled_or_single_chunk_batch",
                        })

                    # Safe fallback: synthesize each chunk independently.
                    for rel_idx, chunk_text in enumerate(batch_texts):
                        chunk_index = i + rel_idx + 1
                        max_attempts = 1 + TTS_CHUNK_MAX_RETRIES
                        chunk_result = None
                        chunk_sr = None
                        last_chunk_error = None

                        for attempt in range(1, max_attempts + 1):
                            avg_chunk_s = _average_chunk_seconds()
                            t_chunk_start = time.perf_counter()
                            try:
                                chunk_wavs_obj, chunk_sr = self.model.generate_voice_clone(
                                    text=chunk_text,
                                    language=language,
                                    voice_clone_prompt=voice_prompt_items,
                                )
                                t_chunk_elapsed = time.perf_counter() - t_chunk_start
                                chunk_wavs = _normalize_wavs(chunk_wavs_obj)

                                if not chunk_wavs:
                                    raise RuntimeError("Empty wav result for chunk")

                                if _is_slow_chunk(t_chunk_elapsed, avg_chunk_s) and attempt < max_attempts:
                                    _status(
                                        f"chunk-retry index={chunk_index} attempt={attempt}/{max_attempts} "
                                        f"reason=slow elapsed={t_chunk_elapsed:.2f}s avg={avg_chunk_s}"
                                    )
                                    telemetry_events.append({
                                        "type": "chunk_retry",
                                        "chunk_index": chunk_index,
                                        "batch_index": batch_index,
                                        "attempt": attempt,
                                        "max_attempts": max_attempts,
                                        "reason": "slow",
                                        "elapsed_s": t_chunk_elapsed,
                                        "avg_chunk_s": avg_chunk_s,
                                    })
                                    continue

                                chunk_result = chunk_wavs[0]
                                chunk_duration_samples.append(t_chunk_elapsed)
                                break
                            except Exception as chunk_error:
                                last_chunk_error = chunk_error
                                _status(
                                    f"chunk-retry index={chunk_index} attempt={attempt}/{max_attempts} "
                                    f"reason=error error={chunk_error}"
                                )
                                telemetry_events.append({
                                    "type": "chunk_retry",
                                    "chunk_index": chunk_index,
                                    "batch_index": batch_index,
                                    "attempt": attempt,
                                    "max_attempts": max_attempts,
                                    "reason": "error",
                                    "error": str(chunk_error),
                                })
                                if attempt >= max_attempts:
                                    raise RuntimeError(
                                        f"Chunk {chunk_index} failed after {max_attempts} attempts"
                                    ) from chunk_error

                        if chunk_result is None:
                            raise RuntimeError(
                                f"Chunk {chunk_index} did not produce audio after retries"
                            ) from last_chunk_error

                        all_wavs.append(chunk_result)
                        sr = sr or chunk_sr
                        completed_chunks += 1
                        fallback_chunks += 1
                        progress_pct = (completed_chunks / total_chunks) * 100 if total_chunks else 100.0
                        _status(
                            f"progress {completed_chunks}/{total_chunks} "
                            f"({progress_pct:.1f}%) mode=single"
                        )
                        telemetry_events.append({
                            "type": "chunk_complete",
                            "chunk_index": chunk_index,
                            "chunks_completed": completed_chunks,
                            "chunks_total": total_chunks,
                            "batch_index": batch_index,
                            "mode": "single",
                            "in_flight_chunks": 1,
                        })

                telemetry_events.append({
                    "type": "chunk_summary",
                    "chunks_total": total_chunks,
                    "chunks_completed": completed_chunks,
                    "batched_requests": batched_requests,
                    "batch_fallbacks": batch_fallbacks,
                    "fallback_chunks": fallback_chunks,
                    "effective_max_concurrency": effective_max_concurrency,
                })
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer = time.perf_counter() - t_infer_start

            # Join chunk audio with a small gap to avoid hard boundaries between chunks.
            if not all_wavs:
                raise RuntimeError("TTS generation returned no audio chunks")
            if len(all_wavs) == 1:
                final_wav = all_wavs[0]
                # Still apply silence gap limiting and edge trim on single chunk
                final_wav = _limit_silence_gaps(final_wav, sr, TTS_SILENCE_GAP_LIMIT_MS)
                final_wav = _trim_chunk_edges(final_wav, sr)
            else:
                # Per-chunk: limit internal silence gaps, then trim edges
                all_wavs = [_limit_silence_gaps(w, sr, TTS_SILENCE_GAP_LIMIT_MS) for w in all_wavs]
                all_wavs = [_trim_chunk_edges(wav_chunk, sr) for wav_chunk in all_wavs]
                dtype = all_wavs[0].dtype
                def _gap(ms: int) -> np.ndarray:
                    samples = int((ms / 1000.0) * sr)
                    return np.zeros(samples, dtype=dtype) if samples > 0 else np.zeros(0, dtype=dtype)
                gap_for = {
                    _BOUNDARY_PARAGRAPH: _gap(TTS_GAP_PARAGRAPH_MS),
                    _BOUNDARY_SENTENCE:  _gap(TTS_GAP_SENTENCE_MS),
                    _BOUNDARY_CLAUSE:    _gap(TTS_GAP_CLAUSE_MS),
                }
                stitched_parts = []
                for idx, wav_chunk in enumerate(all_wavs):
                    stitched_parts.append(wav_chunk)
                    if idx < len(all_wavs) - 1:
                        # The boundary that follows chunk `idx` is `boundaries[idx]`
                        btype = boundaries[idx] if idx < len(boundaries) else _BOUNDARY_SENTENCE
                        g = gap_for.get(btype, gap_for[_BOUNDARY_SENTENCE])
                        if g.size > 0:
                            stitched_parts.append(g)
                final_wav = np.concatenate(stitched_parts)

            # Serialize directly to memory to avoid extra disk write/read.
            t_serialize_start = time.perf_counter()

            # Loudness normalization: target TTS_LUFS_TARGET LUFS
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(sr)
                loudness = meter.integrated_loudness(final_wav.astype(np.float64))
                if np.isfinite(loudness):
                    final_wav = pyln.normalize.loudness(
                        final_wav.astype(np.float64), loudness, TTS_LUFS_TARGET
                    ).astype(final_wav.dtype)
                    print(f"Loudness normalized: {loudness:.1f} LUFS → {TTS_LUFS_TARGET} LUFS")
            except Exception as e:
                print(f"Loudness normalization skipped: {e}")

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, final_wav, sr, format="WAV")
            audio_bytes = wav_buffer.getvalue()
            t_serialize = time.perf_counter() - t_serialize_start

            # Cleanup temp input file
            t_cleanup_start = time.perf_counter()
            if ref_audio_path and os.path.exists(ref_audio_path):
                os.unlink(ref_audio_path)
                ref_audio_path = None
            t_cleanup = time.perf_counter() - t_cleanup_start

            generation_time = time.time() - start_time
            t_total = time.perf_counter() - t0
            duration = len(final_wav) / sr
            rtf = t_infer / duration if duration > 0 else float("inf")
            x_realtime = duration / t_infer if t_infer > 0 else float("inf")

            print(f"Voice clone generated in {generation_time:.2f}s, duration: {duration:.2f}s")
            _status(f"complete generation_time={generation_time:.2f}s duration={duration:.2f}s")
            print(
                "TTS timing breakdown: "
                f"write_ref={t_write_ref:.3f}s, "
                f"inference={t_infer:.3f}s, "
                f"serialize={t_serialize:.3f}s, "
                f"cleanup={t_cleanup:.3f}s, "
                f"total={t_total:.3f}s"
            )
            print(
                "TTS performance: "
                f"rtf={rtf:.3f}, "
                f"x_realtime={x_realtime:.2f}x, "
                f"chunks={total_chunks}, "
                f"batched_requests={batched_requests}, "
                f"batch_fallbacks={batch_fallbacks}, "
                f"fallback_chunks={fallback_chunks}, "
                f"effective_max_concurrency={effective_max_concurrency}"
            )

            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "duration": duration,
                "sample_rate": sr,
                "prompt_bytes": new_prompt_bytes,
                "tts_telemetry": {
                    "chunks_total": total_chunks,
                    "planned_batches": total_batches,
                    "planned_batch_sizes": planned_batch_sizes,
                    "batch_size": batch_size,
                    "batching_enabled": TTS_ENABLE_BATCHING,
                    "configured_max_concurrency": configured_concurrency,
                    "effective_max_concurrency": effective_max_concurrency,
                    "batched_requests": batched_requests,
                    "batch_fallbacks": batch_fallbacks,
                    "fallback_chunks": fallback_chunks,
                    "events": telemetry_events,
                },
            }

        except Exception as e:
            import traceback
            trace = request_trace or "modal"
            print(f"Voice clone error: {e}")
            print(f"[tts:{trace}] error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            if ref_audio_path and os.path.exists(ref_audio_path):
                try:
                    os.unlink(ref_audio_path)
                except OSError:
                    pass


@app.cls(
    image=flux_image,
    gpu=IMAGE_GPU_TYPE,
    scaledown_window=IMAGE_CONTAINER_IDLE_TIMEOUT,
    timeout=600,  # 10 minutes max for image generation
    memory=IMAGE_MEMORY_MB,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("hf-token")],
)
class FluxImageGenerator:
    """FLUX.1-schnell image generation model."""

    @modal.enter()
    def load_model(self):
        """Load FLUX.1-schnell model once per container."""
        import torch
        from diffusers import FluxPipeline
        from huggingface_hub import login
        import time
        import os

        start_time = time.time()
        print(f"Loading FLUX.1-schnell model: {IMAGE_GENERATION_MODEL}...")

        # Authenticate with HuggingFace (required for gated models)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("Authenticated with HuggingFace")

        self.pipe = FluxPipeline.from_pretrained(
            IMAGE_GENERATION_MODEL,
            dtype=torch.bfloat16,
        )
        # Use sequential CPU offload for lower memory usage
        self.pipe.enable_sequential_cpu_offload()

        load_time = time.time() - start_time
        print(f"FLUX.1-schnell model loaded in {load_time:.2f}s")

    @modal.method()
    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels (512-1024)
            height: Image height in pixels (512-1024)
            num_inference_steps: Number of denoising steps (4 for schnell)
            guidance_scale: Classifier-free guidance scale (0.0 for schnell)

        Returns:
            Dict with image_bytes (PNG) and metadata
        """
        import io
        import time

        try:
            import torch
            start_time = time.time()
            print(f"Generating image: {prompt[:100]}...")

            # Clear GPU cache before generation
            torch.cuda.empty_cache()

            # Validate dimensions
            width = max(512, min(1024, width))
            height = max(512, min(1024, height))

            # Generate image (max_sequence_length=128 to save memory)
            image = self.pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=128,
            ).images[0]

            # Convert to PNG bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            generation_time = time.time() - start_time
            print(f"Image generated in {generation_time:.2f}s, size: {len(img_bytes)} bytes")

            return {
                "success": True,
                "image_bytes": img_bytes,
                "width": width,
                "height": height,
                "generation_time": generation_time,
            }

        except Exception as e:
            import traceback
            print(f"Image generation error: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
            }

@app.local_entrypoint()
def main():
    """Test Parakeet TDT model locally."""
    if len(sys.argv) < 2:
        print("Usage: modal run modal_app/app.py <audio_file_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Transcribing {audio_path} with streaming...")
    print("-" * 60)

    model = ParakeetSTTModel()
    segments = []

    for chunk in model.transcribe_stream.remote_gen(audio_bytes):
        data = json.loads(chunk)
        if data["type"] == "metadata":
            print(f"Duration: {data['duration']:.2f}s | Language: {data['language']}")
            print("-" * 60)
        elif data["type"] == "segment":
            print(f"[{data['start']:.2f}s - {data['end']:.2f}s] {data['text']}")
            segments.append(data['text'])
        elif data["type"] == "complete":
            print("-" * 60)
            print(f"Complete transcription:\n{' '.join(segments)}")
        elif data["type"] == "error":
            print(f"ERROR: {data['error']}")
