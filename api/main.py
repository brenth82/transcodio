"""FastAPI application for transcription service."""

import re
import sys
import logging
from pathlib import Path
from typing import Optional
import asyncio
import uuid
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from api.models import (
    TranscriptionResponse,
    ErrorResponse,
    HealthResponse,
    VoiceCloneResponse,
    ImageGenerationResponse,
    SavedVoiceListResponse,
    SaveVoiceResponse,
    SynthesizeResponse,
)
from api.streaming import transcription_event_stream
from utils.audio import validate_audio_file, AudioValidationError

logger = logging.getLogger("transcodio.api")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- API Key Authentication ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# UUID format regex for validation
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename for use in Content-Disposition headers."""
    if not filename:
        return "audio"
    # Extract just the basename (no path components)
    name = Path(filename).name
    # Remove control characters, quotes, semicolons, and newlines
    name = re.sub(r'[\x00-\x1f\x7f";\r\n\\]', '', name)
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name or "audio"

def _safe_content_type(filename: str) -> str:
    """Map filename extension to a known safe MIME type."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return config.SAFE_AUDIO_MIME_TYPES.get(ext, "application/octet-stream")

def _validate_uuid(value: str, name: str = "ID") -> None:
    """Validate that a string is a valid UUID format."""
    if not UUID_PATTERN.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name} format")

def _log_tts_telemetry(trace_id: str, telemetry: Optional[dict]) -> None:
    """Emit structured TTS chunk/batch telemetry into API logs."""
    def _emit(level: str, message: str, *args) -> None:
        formatted = message % args if args else message
        # Explicit console visibility even when logging config suppresses INFO.
        print(formatted, flush=True)
        if level == "warning":
            logger.warning(message, *args)
        else:
            logger.info(message, *args)

    if not telemetry:
        _emit("info", "[tts:%s] telemetry unavailable", trace_id)
        return

    _emit(
        "info",
        "[tts:%s] plan chunks_total=%s planned_batches=%s planned_batch_sizes=%s "
        "batch_size=%s batching_enabled=%s configured_max_concurrency=%s",
        trace_id,
        telemetry.get("chunks_total"),
        telemetry.get("planned_batches"),
        telemetry.get("planned_batch_sizes"),
        telemetry.get("batch_size"),
        telemetry.get("batching_enabled"),
        telemetry.get("configured_max_concurrency"),
    )

    for event in telemetry.get("events", []):
        event_type = event.get("type")
        if event_type == "batch_start":
            _emit(
                "info",
                "[tts:%s] batch start batch=%s/%s chunk_range=%s size=%s",
                trace_id,
                event.get("batch_index"),
                event.get("batches_total"),
                event.get("chunk_range"),
                event.get("batch_chunk_count"),
            )
        if event_type == "chunk_complete":
            _emit(
                "info",
                "[tts:%s] chunk complete chunk=%s/%s mode=%s in_flight=%s batch=%s",
                trace_id,
                event.get("chunks_completed"),
                event.get("chunks_total"),
                event.get("mode"),
                event.get("in_flight_chunks"),
                event.get("batch_index"),
            )
        elif event_type == "batch_fallback":
            _emit(
                "warning",
                "[tts:%s] batch fallback batch=%s chunk_range=%s size=%s reason=%s",
                trace_id,
                event.get("batch_index"),
                event.get("chunk_range"),
                event.get("batch_chunk_count"),
                event.get("reason"),
            )

    _emit(
        "info",
        "[tts:%s] summary batched_requests=%s batch_fallbacks=%s fallback_chunks=%s "
        "effective_max_concurrency=%s",
        trace_id,
        telemetry.get("batched_requests"),
        telemetry.get("batch_fallbacks"),
        telemetry.get("fallback_chunks"),
        telemetry.get("effective_max_concurrency"),
    )


_TTS_BATCH_START_RE = re.compile(r"batch-start index=(\d+)/(\d+) chunks=(\d+)-(\d+) size=(\d+)")
_TTS_PROGRESS_RE = re.compile(r"progress (\d+)/(\d+) \(([\d.]+)%\) mode=([a-z_]+)")
_TTS_CHUNK_PLAN_RE = re.compile(r"chunk-plan chunks=(\d+) batch_size=(\d+) planned_batches=(\d+)")


def _parse_tts_status_message(message: str) -> dict:
    """Parse Modal TTS status lines into structured progress for the browser UI."""
    payload = {"message": message}

    batch_match = _TTS_BATCH_START_RE.search(message)
    if batch_match:
        payload.update({
            "kind": "batch_start",
            "batch_index": int(batch_match.group(1)),
            "batch_total": int(batch_match.group(2)),
            "chunk_start": int(batch_match.group(3)),
            "chunk_end": int(batch_match.group(4)),
            "batch_size": int(batch_match.group(5)),
        })
        return payload

    progress_match = _TTS_PROGRESS_RE.search(message)
    if progress_match:
        payload.update({
            "kind": "chunk_progress",
            "chunk_current": int(progress_match.group(1)),
            "chunk_total": int(progress_match.group(2)),
            "progress_pct": float(progress_match.group(3)),
            "mode": progress_match.group(4),
        })
        return payload

    plan_match = _TTS_CHUNK_PLAN_RE.search(message)
    if plan_match:
        payload.update({
            "kind": "chunk_plan",
            "chunk_total": int(plan_match.group(1)),
            "batch_size": int(plan_match.group(2)),
            "batch_total": int(plan_match.group(3)),
        })
        return payload

    return payload

async def verify_api_key(request: Request):
    """Verify API key if configured. Skip for public routes."""
    # No auth required if API_KEY is not configured (dev mode)
    if not config.API_KEY:
        return
    # Public routes that don't require auth
    public_paths = {"/", "/health", "/docs", "/openapi.json"}
    if request.url.path in public_paths or request.url.path.startswith("/static"):
        return
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Security Headers Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        return response


# Simple in-memory cache for audio (session_id -> (audio_bytes, content_type, filename, expiry_time))
audio_cache = {}

# Simple in-memory cache for images (session_id -> (image_bytes, content_type, expiry_time))
image_cache = {}

def cleanup_expired_audio():
    """Remove expired audio from cache."""
    now = datetime.now()
    expired_keys = [k for k, v in audio_cache.items() if v[3] < now]
    for key in expired_keys:
        del audio_cache[key]

def cleanup_expired_images():
    """Remove expired images from cache."""
    now = datetime.now()
    expired_keys = [k for k, v in image_cache.items() if v[2] < now]
    for key in expired_keys:
        del image_cache[key]

# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

# Attach rate limiter state
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(
        content='{"detail": "Rate limit exceeded. Please try again later."}',
        status_code=429,
        media_type="application/json",
    )

# Add security headers middleware (outermost — runs on every response)
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        content="<h1>Transcodio Transcription Service</h1><p>Upload an audio file to /api/transcribe</p>",
        status_code=200,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
    )


@app.get("/api/audio/{session_id}")
async def get_audio(request: Request, session_id: str):
    """
    Retrieve original audio file by session ID.

    Args:
        session_id: Session ID returned in the transcription completion event

    Returns:
        Original audio file in its uploaded format

    Raises:
        HTTPException: If session ID is invalid or expired
    """
    await verify_api_key(request)
    _validate_uuid(session_id, "session ID")

    # Clean up expired entries first
    cleanup_expired_audio()

    if session_id not in audio_cache:
        raise HTTPException(
            status_code=404,
            detail="Audio session not found or expired"
        )

    audio_bytes, _content_type, filename, expiry = audio_cache[session_id]

    # Use safe MIME type derived from extension, not user-supplied content_type
    safe_filename = _sanitize_filename(filename)
    safe_mime = _safe_content_type(safe_filename)

    return Response(
        content=audio_bytes,
        media_type=safe_mime,
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "Accept-Ranges": "bytes",
        }
    )


@app.post("/api/transcribe", response_model=TranscriptionResponse)
@limiter.limit(config.RATE_LIMIT_TRANSCRIBE)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    """
    Transcribe an audio file (non-streaming).

    Args:
        file: Uploaded audio file

    Returns:
        Complete transcription with segments

    Raises:
        HTTPException: If validation fails or transcription errors
    """
    await verify_api_key(request)
    try:
        # Read file
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        # Validate and preprocess audio
        try:
            duration, preprocessed_bytes = validate_audio_file(
                filename=file.filename,
                file_size=file_size,
                audio_bytes=audio_bytes,
                content_type=file.content_type,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Import Modal and lookup function
        try:
            import modal

            # Lookup the deployed class and method
            STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
            model = STTModel()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail="Transcription service unavailable. Please try again later.",
            )

        # Call Modal transcription
        try:
            result = await model.transcribe.remote.aio(preprocessed_bytes)
            result["duration"] = duration
            return TranscriptionResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed. Please try again.",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )


@app.post("/api/transcribe/stream")
@limiter.limit(config.RATE_LIMIT_TRANSCRIBE)
async def transcribe_audio_stream(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    enable_diarization: bool = Form(default=False, description="Enable speaker diarization"),
):
    """
    Transcribe an audio file with streaming results via Server-Sent Events.

    Args:
        file: Uploaded audio file
        enable_diarization: Whether to identify speakers in the audio

    Returns:
        SSE stream of transcription segments

    Raises:
        HTTPException: If validation fails or transcription errors
    """
    await verify_api_key(request)
    try:
        # Read file
        audio_bytes = await file.read()
        file_size = len(audio_bytes)

        # Validate and preprocess audio
        try:
            duration, preprocessed_bytes = validate_audio_file(
                filename=file.filename,
                file_size=file_size,
                audio_bytes=audio_bytes,
                content_type=file.content_type,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Generate session ID and cache the ORIGINAL audio file
        session_id = str(uuid.uuid4())
        safe_filename = _sanitize_filename(file.filename)
        # Cache for 1 hour - store original audio for playback
        expiry_time = datetime.now() + timedelta(hours=1)
        audio_cache[session_id] = (
            audio_bytes,  # Original uploaded file
            _safe_content_type(safe_filename),  # Safe MIME type from extension
            safe_filename,  # Sanitized filename
            expiry_time
        )
        # Clean up expired entries
        cleanup_expired_audio()

        logger.info(
            "stream request accepted session_id=%s filename=%s size_bytes=%s duration_s=%.2f diarization=%s",
            session_id,
            safe_filename,
            file_size,
            duration,
            enable_diarization,
        )

        # Import Modal and lookup function
        try:
            import modal

            # Lookup the deployed class
            STTModel = modal.Cls.from_name(config.MODAL_APP_NAME, "ParakeetSTTModel")
            model = STTModel()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail="Transcription service unavailable. Please try again later.",
            )

        # Create async generator for streaming
        async def event_generator():
            try:
                import json

                progress_log_step_pct = 10
                progress_log_every_seconds = 8.0
                stream_started_at = time.perf_counter()
                last_progress_log_at = stream_started_at
                last_progress_bucket = -1
                yielded_segments = 0

                logger.info("stream started session_id=%s", session_id)

                # Accumulate segments for diarization
                segments_data = []
                full_text = ""

                async for segment_json in model.transcribe_stream.remote_gen.aio(preprocessed_bytes, duration):
                    # Parse and yield as SSE event
                    segment_data = json.loads(segment_json)
                    event_type = segment_data.get("type", "unknown")

                    if event_type == "metadata":
                        logger.info(
                            "stream metadata session_id=%s language=%s duration_s=%.2f",
                            session_id,
                            segment_data.get("language"),
                            float(segment_data.get("duration") or 0.0),
                        )
                        yield {
                            "event": "metadata",
                            "data": json.dumps({
                                "language": segment_data.get("language"),
                                "duration": segment_data.get("duration"),
                            })
                        }

                    elif event_type == "segment":
                        # Accumulate segment for diarization
                        segments_data.append({
                            "id": segment_data.get("id"),
                            "start": segment_data.get("start"),
                            "end": segment_data.get("end"),
                            "text": segment_data.get("text"),
                        })
                        yielded_segments += 1

                        # Throttled progress logging: emit at 10% buckets or every N seconds.
                        seg_end = float(segment_data.get("end") or 0.0)
                        total_duration = duration if duration > 0 else 1.0
                        progress_pct = max(0.0, min(100.0, (seg_end / total_duration) * 100.0))
                        progress_bucket = int(progress_pct // progress_log_step_pct)
                        now = time.perf_counter()
                        time_due = (now - last_progress_log_at) >= progress_log_every_seconds
                        bucket_due = progress_bucket > last_progress_bucket
                        if time_due or bucket_due:
                            logger.info(
                                "stream progress session_id=%s segments=%s progress=%.1f%%",
                                session_id,
                                yielded_segments,
                                progress_pct,
                            )
                            last_progress_log_at = now
                            last_progress_bucket = progress_bucket

                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "id": segment_data.get("id"),
                                "start": segment_data.get("start"),
                                "end": segment_data.get("end"),
                                "text": segment_data.get("text"),
                            })
                        }

                    elif event_type == "complete":
                        full_text = segment_data.get("text", "")
                        elapsed = time.perf_counter() - stream_started_at
                        logger.info(
                            "stream complete session_id=%s segments=%s elapsed_s=%.2f text_chars=%s",
                            session_id,
                            yielded_segments,
                            elapsed,
                            len(full_text),
                        )

                        # Run speaker diarization if enabled and there are segments
                        if enable_diarization and config.ENABLE_SPEAKER_DIARIZATION and segments_data:
                            try:
                                logger.info("diarization started session_id=%s", session_id)
                                # Import diarizer from Modal
                                Diarizer = modal.Cls.from_name(config.MODAL_APP_NAME, "SpeakerDiarizerModel")
                                diarizer = Diarizer()

                                # Run diarization
                                speaker_timeline = await diarizer.diarize.remote.aio(preprocessed_bytes, duration)

                                if speaker_timeline:
                                    # Import alignment function
                                    import sys
                                    from pathlib import Path
                                    sys.path.insert(0, str(Path(__file__).parent.parent / "modal_app"))
                                    from app import align_speakers_to_segments

                                    # Align speakers with segments
                                    segments_with_speakers = align_speakers_to_segments(
                                        segments_data,
                                        speaker_timeline
                                    )

                                    # Yield updated segments with speaker labels
                                    yield {
                                        "event": "speakers_ready",
                                        "data": json.dumps({"segments": segments_with_speakers})
                                    }

                                    logger.info(
                                        "diarization complete session_id=%s speaker_segments=%s",
                                        session_id,
                                        len(speaker_timeline),
                                    )
                                else:
                                    logger.info("diarization empty result session_id=%s", session_id)
                            except Exception as e:
                                logger.exception("diarization failed session_id=%s", session_id)
                                # Continue without speaker labels

                        # Yield completion event
                        yield {
                            "event": "complete",
                            "data": json.dumps({
                                "text": full_text,
                                "audio_session_id": session_id,
                            })
                        }

                    elif event_type == "error":
                        logger.warning(
                            "stream model error session_id=%s error=%s",
                            session_id,
                            segment_data.get("error"),
                        )
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "error": segment_data.get("error"),
                            })
                        }

            except Exception as e:
                logger.exception("stream event generator failed session_id=%s", session_id)
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": f"Transcription error: {str(e)}",
                    })
                }

        # Return SSE response
        return EventSourceResponse(event_generator())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )


@app.post("/api/voice-clone", response_model=VoiceCloneResponse)
@limiter.limit(config.RATE_LIMIT_VOICE_CLONE)
async def voice_clone(
    request: Request,
    ref_audio: UploadFile = File(..., description="Reference audio file (3-30 seconds)"),
    ref_text: str = Form(default="", description="Transcription of the reference audio (optional; omit to use speaker-embedding-only mode)"),
    target_text: str = Form(..., description="Text to synthesize with cloned voice"),
    language: str = Form(default="Spanish", description="Target language"),
    tts_model: str = Form(default=config.DEFAULT_TTS_MODEL, description="TTS model to use"),
):
    """
    Clone a voice and synthesize new text.

    Args:
        ref_audio: Reference audio file (3-30 seconds)
        ref_text: Transcription of the reference audio
        target_text: Text to synthesize with cloned voice
        language: Target language (Spanish, English, etc.)
        tts_model: TTS model to use (currently qwen)

    Returns:
        VoiceCloneResponse with audio_session_id for playback/download

    Raises:
        HTTPException: If validation fails or generation errors
    """
    await verify_api_key(request)
    # Check if voice cloning is enabled
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(
            status_code=503,
            detail="Voice cloning is disabled"
        )

    # Validate TTS model
    if tts_model not in config.TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid TTS model. Available: {', '.join(config.TTS_MODELS.keys())}"
        )

    try:
        request_trace = uuid.uuid4().hex[:8]
        t_request_start = time.perf_counter()
        logger.info("[voice-clone:%s] request started", request_trace)

        # Validate language
        if language not in config.VOICE_CLONE_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Supported: {', '.join(config.VOICE_CLONE_LANGUAGES)}"
            )

        # Validate target text length
        if len(target_text) > config.VOICE_CLONE_MAX_TARGET_TEXT:
            raise HTTPException(
                status_code=400,
                detail=f"Target text too long. Maximum {config.VOICE_CLONE_MAX_TARGET_TEXT} characters."
            )

        if len(target_text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Target text cannot be empty."
            )

        # Read reference audio
        t_read_start = time.perf_counter()
        logger.info("[voice-clone:%s] reading reference audio", request_trace)
        ref_audio_bytes = await ref_audio.read()
        file_size = len(ref_audio_bytes)
        t_read = time.perf_counter() - t_read_start

        # Validate file size (max 15MB for reference audio)
        if file_size > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Reference audio too large. Maximum 15MB."
            )

        # Preprocess reference audio (convert to 24kHz mono WAV)
        try:
            t_preprocess_start = time.perf_counter()
            logger.info("[voice-clone:%s] validating and preprocessing reference audio", request_trace)
            ref_duration, preprocessed_ref = validate_audio_file(
                filename=ref_audio.filename,
                file_size=file_size,
                audio_bytes=ref_audio_bytes,
                content_type=ref_audio.content_type,
                target_sample_rate=config.VOICE_CLONE_SAMPLE_RATE,
            )
            t_preprocess = time.perf_counter() - t_preprocess_start
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate reference audio duration
        if ref_duration < config.VOICE_CLONE_MIN_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too short. Minimum {config.VOICE_CLONE_MIN_REF_DURATION} seconds."
            )

        if ref_duration > config.VOICE_CLONE_MAX_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too long. Maximum {config.VOICE_CLONE_MAX_REF_DURATION} seconds."
            )

        # Import Modal and lookup TTS model based on selection
        try:
            import modal

            t_modal_lookup_start = time.perf_counter()
            logger.info("[voice-clone:%s] resolving Modal class %s", request_trace, "Qwen3TTSVoiceCloner")
            TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
            model = TTSModel()
            t_modal_lookup = time.perf_counter() - t_modal_lookup_start
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail="Voice cloning service unavailable. Please try again later.",
            )

        # Generate voice clone
        try:
            t_remote_start = time.perf_counter()
            logger.info("[voice-clone:%s] invoking Modal TTS (chars=%s, language=%s)", request_trace, len(target_text), language)
            result = await model.generate_voice_clone.remote.aio(
                preprocessed_ref,
                ref_text,
                target_text,
                language,
                request_trace,
            )
            t_remote = time.perf_counter() - t_remote_start

            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Voice cloning failed: {result.get('error', 'Unknown error')}"
                )

            _log_tts_telemetry(request_trace, result.get("tts_telemetry"))

            # Generate session ID and cache the generated audio
            t_cache_start = time.perf_counter()
            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=1)
            audio_cache[session_id] = (
                result["audio_bytes"],
                "audio/wav",
                "voice_clone.wav",
                expiry_time
            )

            # Clean up expired entries
            cleanup_expired_audio()
            t_cache = time.perf_counter() - t_cache_start

            t_total = time.perf_counter() - t_request_start
            logger.info(
                "[voice-clone:%s] timing read=%.3fs preprocess=%.3fs modal_lookup=%.3fs "
                "remote_tts=%.3fs cache=%.3fs total=%.3fs",
                request_trace,
                t_read,
                t_preprocess,
                t_modal_lookup,
                t_remote,
                t_cache,
                t_total,
            )

            return VoiceCloneResponse(
                success=True,
                audio_session_id=session_id,
                duration=result.get("duration"),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Voice cloning failed. Please try again.",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )


@app.get("/api/voices", response_model=SavedVoiceListResponse)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
async def list_voices(request: Request):
    """
    List all saved voices.

    Returns:
        SavedVoiceListResponse with list of saved voices
    """
    await verify_api_key(request)
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    try:
        import modal

        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        voices = await storage.list_voices.remote.aio()
        return SavedVoiceListResponse(voices=voices)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Voice storage service unavailable. Please try again later."
        )


@app.post("/api/voices", response_model=SaveVoiceResponse)
@limiter.limit(config.RATE_LIMIT_DEFAULT)
async def save_voice(
    request: Request,
    name: str = Form(..., description="Name for the voice"),
    ref_audio: UploadFile = File(..., description="Reference audio file (3-60 seconds)"),
    ref_text: str = Form(..., description="Exact transcription of the reference audio"),
    language: str = Form(default="Spanish", description="Voice language"),
):
    """
    Save a new voice for later use.

    Args:
        name: User-friendly name for the voice
        ref_audio: Reference audio file (3-60 seconds)
        ref_text: Exact transcription of the reference audio
        language: Language of the voice

    Returns:
        SaveVoiceResponse with voice_id
    """
    await verify_api_key(request)
    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    # Validate name
    if not name or len(name.strip()) == 0:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty")
    if len(name) > 50:
        raise HTTPException(status_code=400, detail="Voice name too long (max 50 characters)")

    # Validate language
    if language not in config.VOICE_CLONE_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported: {', '.join(config.VOICE_CLONE_LANGUAGES)}"
        )

    try:
        # Read and validate audio
        ref_audio_bytes = await ref_audio.read()
        file_size = len(ref_audio_bytes)

        if file_size > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Reference audio too large (max 15MB)")

        # Preprocess audio
        try:
            ref_duration, preprocessed_ref = validate_audio_file(
                filename=ref_audio.filename,
                file_size=file_size,
                audio_bytes=ref_audio_bytes,
                content_type=ref_audio.content_type,
                target_sample_rate=config.VOICE_CLONE_SAMPLE_RATE,
            )
        except AudioValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate duration
        if ref_duration < config.VOICE_CLONE_MIN_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too short (min {config.VOICE_CLONE_MIN_REF_DURATION}s)"
            )
        if ref_duration > config.VOICE_CLONE_MAX_REF_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio too long (max {config.VOICE_CLONE_MAX_REF_DURATION}s)"
            )

        # Compute voice prompt on Modal GPU
        import modal
        logger.info("[save-voice] computing voice prompt on GPU")
        TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
        tts_model = TTSModel()
        prompt_bytes = await tts_model.compute_voice_prompt.remote.aio(
            preprocessed_ref,
            ref_text.strip(),
            language,
        )
        logger.info("[save-voice] voice prompt computed, storing voice metadata")

        # Save voice with prompt (no audio stored)
        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()

        voice_id = str(uuid.uuid4())
        result = await storage.save_voice.remote.aio(
            voice_id=voice_id,
            name=name.strip(),
            ref_text=ref_text.strip(),
            language=language,
            prompt_bytes=prompt_bytes,
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to save voice"))

        return SaveVoiceResponse(success=True, voice_id=voice_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.delete("/api/voices/{voice_id}")
@limiter.limit(config.RATE_LIMIT_DEFAULT)
async def delete_voice(request: Request, voice_id: str):
    """
    Delete a saved voice.

    Args:
        voice_id: ID of the voice to delete

    Returns:
        Success status
    """
    await verify_api_key(request)
    _validate_uuid(voice_id, "voice ID")

    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    try:
        import modal

        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        result = await storage.delete_voice.remote.aio(voice_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail="Voice not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.post("/api/synthesize", response_model=SynthesizeResponse)
@limiter.limit(config.RATE_LIMIT_VOICE_CLONE)
async def synthesize_with_voice(
    request: Request,
    voice_id: str = Form(..., description="ID of the saved voice to use"),
    target_text: str = Form(..., description="Text to synthesize"),
):
    """
    Synthesize text using a saved voice.

    Args:
        voice_id: ID of the saved voice
        target_text: Text to synthesize (max 500 characters)

    Returns:
        SynthesizeResponse with audio_session_id for playback/download
    """
    await verify_api_key(request)
    _validate_uuid(voice_id, "voice ID")

    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    # Validate target text
    if not target_text or len(target_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Target text cannot be empty")
    if len(target_text) > config.VOICE_CLONE_MAX_TARGET_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"Target text too long (max {config.VOICE_CLONE_MAX_TARGET_TEXT} characters)"
        )

    try:
        import modal
        request_trace = uuid.uuid4().hex[:8]
        t_request_start = time.perf_counter()
        print(f"[synthesize:{request_trace}] request started", flush=True)
        logger.info("[synthesize:%s] request started", request_trace)

        # Get saved voice (metadata + audio + cached voice prompt if available)
        t_storage_start = time.perf_counter()
        logger.info("[synthesize:%s] loading saved voice metadata and prompt", request_trace)
        VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
        storage = VoiceStorage()
        voice_data = await storage.get_voice_with_prompt.remote.aio(voice_id)
        t_storage = time.perf_counter() - t_storage_start

        if not voice_data.get("success"):
            raise HTTPException(status_code=404, detail=voice_data.get("error", "Voice not found"))

        metadata = voice_data["metadata"]
        prompt_bytes = voice_data.get("prompt_bytes")
        ref_audio_bytes = voice_data.get("audio_bytes")

        logger.info(
            "[synthesize:%s] loaded voice: name=%s language=%s",
            request_trace,
            metadata.get("name"),
            metadata["language"],
        )

        # Legacy fallback: for voices saved before prompt caching, compute once
        # from stored ref audio, then persist prompt.pt for subsequent calls.
        if prompt_bytes is None:
            if not ref_audio_bytes:
                raise HTTPException(status_code=404, detail="Voice prompt not found")

            logger.info("[synthesize:%s] legacy voice detected; computing prompt from ref audio", request_trace)
            TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
            prompt_model = TTSModel()
            prompt_bytes = await prompt_model.compute_voice_prompt.remote.aio(
                ref_audio_bytes,
                metadata.get("ref_text", ""),
                metadata["language"],
            )
            save_prompt_result = await storage.save_voice_prompt.remote.aio(voice_id, prompt_bytes)
            if not save_prompt_result.get("success"):
                logger.warning(
                    "[synthesize:%s] failed to persist migrated prompt for voice_id=%s: %s",
                    request_trace,
                    voice_id,
                    save_prompt_result.get("error", "unknown"),
                )
            else:
                logger.info("[synthesize:%s] migrated legacy voice prompt for voice_id=%s", request_trace, voice_id)

        # Generate audio with TTS
        t_modal_lookup_start = time.perf_counter()
        logger.info("[synthesize:%s] resolving Modal class %s", request_trace, "Qwen3TTSVoiceCloner")
        TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
        model = TTSModel()
        t_modal_lookup = time.perf_counter() - t_modal_lookup_start

        t_remote_start = time.perf_counter()
        print(
            f"[synthesize:{request_trace}] invoking Modal TTS chars={len(target_text.strip())} "
            f"language={metadata['language']} voice_id={voice_id}",
            flush=True,
        )
        logger.info(
            "[synthesize:%s] invoking Modal TTS (chars=%s, language=%s, voice_id=%s)",
            request_trace,
            len(target_text.strip()),
            metadata["language"],
            voice_id,
        )
        import json

        result = None
        stream_had_progress = False
        stream_error_message = None
        try:
            async for synth_event in model.generate_voice_clone_stream.remote_gen.aio(
                None,  # ref_audio_bytes not needed; using cached prompt
                metadata.get("ref_text", ""),
                target_text.strip(),
                metadata["language"],
                request_trace,
                prompt_bytes,  # use cached prompt directly
            ):
                # Modal generators may surface payloads as dicts, JSON strings, or bytes.
                if isinstance(synth_event, (bytes, bytearray)):
                    synth_event = synth_event.decode("utf-8", errors="replace")
                if isinstance(synth_event, str):
                    try:
                        synth_event = json.loads(synth_event)
                    except json.JSONDecodeError:
                        # Treat non-JSON status lines as progress messages.
                        synth_event = {"type": "status", "message": synth_event}
                if not isinstance(synth_event, dict):
                    print(
                        f"[synthesize:{request_trace}] unexpected stream event type="
                        f"{type(synth_event)} value={synth_event!r}",
                        flush=True,
                    )
                    continue

                event_type = synth_event.get("type")
                if event_type == "status":
                    message = synth_event.get("message", "")
                    if message:
                        stream_had_progress = True
                        print(message, flush=True)
                        logger.info("[synthesize:%s] %s", request_trace, message)
                elif event_type == "error":
                    error_msg = synth_event.get("error", "Unknown synthesis error")
                    stream_error_message = error_msg
                    logger.warning(
                        "[synthesize:%s] stream emitted error event: %s",
                        request_trace,
                        error_msg,
                    )
                    # Break and fall back to non-stream path below.
                    break
                elif event_type == "complete":
                    result = synth_event.get("result")
        except HTTPException:
            raise
        except Exception as stream_error:
            # Backward compatibility: if Modal deploy is older and doesn't yet have
            # generate_voice_clone_stream, fall back to non-stream invocation.
            print(
                f"[synthesize:{request_trace}] stream path failed ({stream_error}); "
                "falling back to non-stream call",
                flush=True,
            )
            logger.warning(
                "[synthesize:%s] stream path failed, falling back to non-stream: %s",
                request_trace,
                stream_error,
            )
        # If stream did not complete successfully, fall back to non-stream.
        if result is None:
            if stream_error_message:
                print(
                    f"[synthesize:{request_trace}] stream returned error event: {stream_error_message}; "
                    "falling back to non-stream call",
                    flush=True,
                )
            elif stream_had_progress:
                print(
                    f"[synthesize:{request_trace}] stream ended without complete event; "
                    "falling back to non-stream call",
                    flush=True,
                )
            result = await model.generate_voice_clone.remote.aio(
                None,
                metadata.get("ref_text", ""),
                target_text.strip(),
                metadata["language"],
                request_trace,
                prompt_bytes,
            )

        t_remote = time.perf_counter() - t_remote_start

        if result is None:
            raise HTTPException(status_code=500, detail="Synthesis failed: no result returned")

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {result.get('error', 'Unknown error')}"
            )

        _log_tts_telemetry(request_trace, result.get("tts_telemetry"))

        # Cache generated audio
        t_cache_start = time.perf_counter()
        session_id = str(uuid.uuid4())
        expiry_time = datetime.now() + timedelta(hours=1)
        audio_cache[session_id] = (
            result["audio_bytes"],
            "audio/wav",
            "synthesized.wav",
            expiry_time,
        )
        cleanup_expired_audio()
        t_cache = time.perf_counter() - t_cache_start

        t_total = time.perf_counter() - t_request_start
        logger.info(
            "[synthesize:%s] timing storage=%.3fs modal_lookup=%.3fs remote_tts=%.3fs "
            "cache=%.3fs total=%.3fs",
            request_trace,
            t_storage,
            t_modal_lookup,
            t_remote,
            t_cache,
            t_total,
        )

        return SynthesizeResponse(
            success=True,
            audio_session_id=session_id,
            duration=result.get("duration"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[synthesize] unexpected server error")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.post("/api/synthesize/stream")
@limiter.limit(config.RATE_LIMIT_VOICE_CLONE)
async def synthesize_with_voice_stream(
    request: Request,
    voice_id: str = Form(..., description="ID of the saved voice to use"),
    target_text: str = Form(..., description="Text to synthesize"),
):
    """Stream synth progress to the browser while generating audio."""
    await verify_api_key(request)
    _validate_uuid(voice_id, "voice ID")

    if not config.ENABLE_VOICE_CLONING:
        raise HTTPException(status_code=503, detail="Voice cloning is disabled")

    if not target_text or len(target_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Target text cannot be empty")
    if len(target_text) > config.VOICE_CLONE_MAX_TARGET_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"Target text too long (max {config.VOICE_CLONE_MAX_TARGET_TEXT} characters)"
        )

    import modal
    import json

    request_trace = uuid.uuid4().hex[:8]
    VoiceStorage = modal.Cls.from_name(config.MODAL_APP_NAME, "VoiceStorage")
    storage = VoiceStorage()
    voice_data = await storage.get_voice_with_prompt.remote.aio(voice_id)
    if not voice_data.get("success"):
        raise HTTPException(status_code=404, detail=voice_data.get("error", "Voice not found"))

    metadata = voice_data["metadata"]
    prompt_bytes = voice_data.get("prompt_bytes")
    ref_audio_bytes = voice_data.get("audio_bytes")

    if prompt_bytes is None:
        if not ref_audio_bytes:
            raise HTTPException(status_code=404, detail="Voice prompt not found")
        TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
        prompt_model = TTSModel()
        prompt_bytes = await prompt_model.compute_voice_prompt.remote.aio(
            ref_audio_bytes,
            metadata.get("ref_text", ""),
            metadata["language"],
        )
        await storage.save_voice_prompt.remote.aio(voice_id, prompt_bytes)

    async def event_generator():
        TTSModel = modal.Cls.from_name(config.MODAL_APP_NAME, "Qwen3TTSVoiceCloner")
        model = TTSModel()
        result = None

        try:
            async for synth_event in model.generate_voice_clone_stream.remote_gen.aio(
                None,
                metadata.get("ref_text", ""),
                target_text.strip(),
                metadata["language"],
                request_trace,
                prompt_bytes,
            ):
                if isinstance(synth_event, (bytes, bytearray)):
                    synth_event = synth_event.decode("utf-8", errors="replace")
                if isinstance(synth_event, str):
                    try:
                        synth_event = json.loads(synth_event)
                    except json.JSONDecodeError:
                        synth_event = {"type": "status", "message": synth_event}

                if not isinstance(synth_event, dict):
                    continue

                event_type = synth_event.get("type")
                if event_type == "status":
                    yield {
                        "event": "progress",
                        "data": json.dumps(_parse_tts_status_message(synth_event.get("message", ""))),
                    }
                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": synth_event.get("error", "Unknown synthesis error")}),
                    }
                    return
                elif event_type == "complete":
                    result = synth_event.get("result")

            if result is None:
                result = await model.generate_voice_clone.remote.aio(
                    None,
                    metadata.get("ref_text", ""),
                    target_text.strip(),
                    metadata["language"],
                    request_trace,
                    prompt_bytes,
                )

            if not result or not result.get("success"):
                yield {
                    "event": "error",
                    "data": json.dumps({"error": (result or {}).get("error", "Unknown synthesis error")}),
                }
                return

            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=1)
            audio_cache[session_id] = (
                result["audio_bytes"],
                "audio/wav",
                "synthesized.wav",
                expiry_time,
            )
            cleanup_expired_audio()

            yield {
                "event": "complete",
                "data": json.dumps({
                    "audio_session_id": session_id,
                    "duration": result.get("duration"),
                }),
            }
        except Exception:
            logger.exception("[synthesize-stream:%s] unexpected server error", request_trace)
            yield {
                "event": "error",
                "data": json.dumps({"error": "An unexpected error occurred."}),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/image/{session_id}")
async def get_image(request: Request, session_id: str):
    """
    Retrieve generated image by session ID.

    Args:
        session_id: Session ID returned in the image generation response

    Returns:
        Generated image as PNG

    Raises:
        HTTPException: If session ID is invalid or expired
    """
    await verify_api_key(request)
    _validate_uuid(session_id, "session ID")

    # Clean up expired entries first
    cleanup_expired_images()

    if session_id not in image_cache:
        raise HTTPException(
            status_code=404,
            detail="Image session not found or expired"
        )

    image_bytes, content_type, expiry = image_cache[session_id]

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": 'inline; filename="generated-image.png"',
        }
    )


@app.post("/api/generate-image", response_model=ImageGenerationResponse)
@limiter.limit(config.RATE_LIMIT_IMAGE)
async def generate_image(
    request: Request,
    prompt: str = Form(..., description="Text prompt describing the image to generate"),
    width: int = Form(default=768, description="Image width (512-1024)"),
    height: int = Form(default=768, description="Image height (512-1024)"),
):
    """
    Generate an image from a text prompt using FLUX.1-schnell.

    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (512-1024)
        height: Image height in pixels (512-1024)

    Returns:
        ImageGenerationResponse with image_session_id for retrieval

    Raises:
        HTTPException: If validation fails or generation errors
    """
    await verify_api_key(request)
    # Check if image generation is enabled
    if not config.ENABLE_IMAGE_GENERATION:
        raise HTTPException(
            status_code=503,
            detail="Image generation is disabled"
        )

    try:
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )

        if len(prompt) > config.IMAGE_MAX_PROMPT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long. Maximum {config.IMAGE_MAX_PROMPT_LENGTH} characters."
            )

        # Validate dimensions
        width = max(512, min(1024, width))
        height = max(512, min(1024, height))

        # Import Modal and lookup function
        try:
            import modal

            ImageGenerator = modal.Cls.from_name(config.MODAL_APP_NAME, "FluxImageGenerator")
            generator = ImageGenerator()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail="Image generation service unavailable. Please try again later.",
            )

        # Generate image
        try:
            result = await generator.generate_image.remote.aio(
                prompt=prompt.strip(),
                width=width,
                height=height,
                num_inference_steps=config.IMAGE_NUM_INFERENCE_STEPS,
                guidance_scale=config.IMAGE_GUIDANCE_SCALE,
            )

            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {result.get('error', 'Unknown error')}"
                )

            # Generate session ID and cache the generated image
            session_id = str(uuid.uuid4())
            expiry_time = datetime.now() + timedelta(hours=config.IMAGE_CACHE_EXPIRY_HOURS)
            image_cache[session_id] = (
                result["image_bytes"],
                "image/png",
                expiry_time
            )

            # Clean up expired entries
            cleanup_expired_images()

            return ImageGenerationResponse(
                success=True,
                image_session_id=session_id,
                width=result.get("width"),
                height=result.get("height"),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Image generation failed. Please try again.",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.DEBUG,
    )
