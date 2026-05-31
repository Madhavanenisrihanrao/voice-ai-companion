"""
app.py
------
Web interface for the Emotion-Aware AI Companion.

Serves a modern Three.js-powered front-end with real-time chat,
optional webcam emotion detection, and browser-native TTS/STT
via the Web Speech API.

Usage:
    python app.py                  # Start with camera
    python app.py --no-camera      # Start without camera
    python app.py --port 5000      # Custom port
"""

import os

# eventlet must be monkey-patched before all other imports when running on Render
if os.environ.get("RENDER"):
    import eventlet
    eventlet.monkey_patch()

import argparse
import base64
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, send_file
from flask_socketio import SocketIO, emit
from io import BytesIO
from gtts import gTTS

from emotion_detector import EmotionDetector
from groq_chat import GroqChat
from sentiment_analyzer import SentimentAnalyzer
from session_logger import SessionLogger
from emotion_analytics import EmotionAnalytics
from language_detector import detect_language
from logger_setup import get_logger
import settings

log = get_logger(__name__)

# ── Flask application ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
_async_mode = "eventlet" if os.environ.get("RENDER") else "threading"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_async_mode)

# ── Shared camera state ──────────────────────────────────────────────
_emotion_detector: EmotionDetector | None = None
_camera_lock = threading.Lock()
_current_emotion = "neutral"
_current_emotion_confidence = 1.0
_use_camera = False

# Per-client sessions  {sid: {chat, sentiment, session, analytics}}
_clients: dict[str, dict] = {}

# Lazy FER detector for browser-frame emotion analysis
_fer_detector = None
_fer_lock = threading.Lock()

# Lazy Vision analyzer for multimodal/object recognition
_vision_analyzer = None
_vision_lock = threading.Lock()

def _get_fer():
    global _fer_detector
    if _fer_detector is None:
        with _fer_lock:
            if _fer_detector is None:
                try:
                    from fer import FER
                except (ImportError, ModuleNotFoundError):
                    from fer.fer import FER
                _fer_detector = FER(mtcnn=True)
    return _fer_detector

def _get_vision():
    global _vision_analyzer
    if _vision_analyzer is None:
        with _vision_lock:
            if _vision_analyzer is None:
                from vision_analyzer import VisionAnalyzer
                _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer


# ── Camera helpers ────────────────────────────────────────────────────

def _init_camera(enabled: bool, cam_index: int = 0) -> None:
    global _emotion_detector, _use_camera
    _use_camera = enabled
    if enabled:
        try:
            _emotion_detector = EmotionDetector(camera_index=cam_index)
            _emotion_detector.start_camera()
        except RuntimeError as exc:
            log.warning("Camera unavailable: %s", exc)
            _use_camera = False


def _emotion_loop() -> None:
    """Background thread: continuously analyse emotions from the webcam."""
    global _current_emotion, _current_emotion_confidence
    while True:
        if not _use_camera or _emotion_detector is None:
            time.sleep(1)
            continue
        try:
            with _camera_lock:
                frame = _emotion_detector.capture_frame()
            emotion, conf = _emotion_detector.detect_emotion(frame=frame)
            _current_emotion = emotion
            _current_emotion_confidence = float(conf)
            socketio.emit("emotion_update", {
                "emotion": _current_emotion,
                "confidence": _current_emotion_confidence
            })
        except Exception:
            pass
        time.sleep(settings.EMOTION_ANALYSIS_INTERVAL)


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", use_camera=_use_camera)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the webcam with an emotion label overlay."""
    if not _use_camera or _emotion_detector is None:
        return Response(status=204)

    def generate():
        while True:
            try:
                with _camera_lock:
                    frame = _emotion_detector.capture_frame()
                cv2.putText(
                    frame, _current_emotion.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            except Exception:
                pass
            time.sleep(0.033)  # ~30 fps

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/tts")
def tts_endpoint():
    """Generate speech audio from text using gTTS (supports many languages)."""
    text = request.args.get("text", "").strip()
    lang = request.args.get("lang", "en").strip()
    if not text:
        return Response(status=400)
    # gTTS uses ISO 639-1 codes; fall back to English on error
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg")
    except Exception as exc:
        log.warning("gTTS error (lang=%s): %s — falling back to en", lang, exc)
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return send_file(buf, mimetype="audio/mpeg")
        except Exception:
            return Response(status=500)


# ── Socket.IO events ─────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.sid
    try:
        chat = GroqChat()
    except RuntimeError as exc:
        emit("error", {"message": str(exc)})
        return

    _clients[sid] = {
        "chat": chat,
        "sentiment": SentimentAnalyzer(),
        "session": SessionLogger(),
        "analytics": EmotionAnalytics(),
        "lang": "en",
        "processing": False,  # True when AI is busy responding
    }

    try:
        opening = chat.get_opening_message(_current_emotion, _current_emotion_confidence)
    except Exception:
        opening = "Hello! I'm your AI companion. How are you feeling today?"

    emit("ai_response", {
        "text": opening,
        "emotion": _current_emotion,
        "confidence": _current_emotion_confidence,
        "greeting": True,
        "lang": "en",
    })
    log.info("Client connected: %s", sid)


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    ctx = _clients.pop(sid, None)
    if ctx:
        ctx["analytics"].print_report()
        ctx["session"].save()
    log.info("Client disconnected: %s", sid)


@socketio.on("browser_frame")
def on_browser_frame(data):
    """Receive a camera frame from the browser and run emotion detection."""
    global _current_emotion, _current_emotion_confidence
    image_data = data.get("image", "")
    if not image_data or "," not in image_data:
        return
    try:
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        detector = _get_fer()
        result = detector.top_emotion(frame)
        if result and result[0]:
            emotion, conf = result
            _current_emotion = emotion
            _current_emotion_confidence = float(conf)
            emit("emotion_update", {
                "emotion": emotion,
                "confidence": float(conf)
            })
    except Exception as exc:
        log.warning("browser_frame error: %s", exc)


@socketio.on("vision_frame")
def on_vision_frame(data):
    """Receive a high-quality frame for proactive vision analysis."""
    sid = request.sid
    ctx = _clients.get(sid)
    if not ctx:
        return

    # Skip proactive observation if AI is busy processing
    if ctx.get("processing", False):
        return

    image_data = data.get("image", "")
    if not image_data:
        return

    try:
        vision = _get_vision()
        lang = ctx.get("lang", "en")
        
        should_speak, message = vision.proactive_observe(
            image_data,
            emotion=_current_emotion,
            lang=lang,
        )
        
        if should_speak and message:
            emit("ai_proactive", {
                "text": message,
                "emotion": _current_emotion,
                "confidence": _current_emotion_confidence,
                "lang": lang,
            })
            log.info("Proactive observation: %s", message[:50])
    except Exception as exc:
        log.warning("vision_frame error: %s", exc)


@socketio.on("analyze_image")
def on_analyze_image(data):
    """Analyze an image when user asks about something they're showing."""
    sid = request.sid
    ctx = _clients.get(sid)
    if not ctx:
        return

    image_data = data.get("image", "")
    question = data.get("question", "").strip()
    if not image_data:
        return

    ctx["processing"] = True  # Mark as busy
    try:
        vision = _get_vision()
        lang = ctx.get("lang", "en")
        
        response = vision.analyze_image(
            image_data,
            user_question=question,
            emotion=_current_emotion,
            lang=lang,
        )
        
        if response:
            emit("ai_response", {
                "text": response,
                "emotion": _current_emotion,
                "confidence": _current_emotion_confidence,
                "lang": lang,
                "vision": True,
            })
            # Also add to chat history
            ctx["chat"]._conversation_history.append({
                "role": "assistant",
                "content": response,
            })
    except Exception as exc:
        log.warning("analyze_image error: %s", exc)
        emit("ai_response", {
            "text": "I couldn't analyze the image. Please try again.",
            "emotion": _current_emotion,
            "confidence": _current_emotion_confidence,
            "lang": "en",
        })
    finally:
        ctx["processing"] = False  # Done


@socketio.on("user_message")
def on_user_message(data):
    sid = request.sid
    ctx = _clients.get(sid)
    if not ctx:
        return

    text = (data.get("text") or "").strip()
    if not text:
        return

    ctx["processing"] = True  # Mark as busy
    log.info("User [%s]: %s", sid[:8], text)

    # Detect language
    lang = detect_language(text)
    ctx["lang"] = lang

    # Sentiment analysis
    sentiment_result = ctx["sentiment"].analyze(text)
    combined = ctx["sentiment"].combined_mood(
        _current_emotion, text,
        emotion_weight=settings.EMOTION_WEIGHT,
        sentiment_weight=settings.SENTIMENT_WEIGHT,
    )
    ctx["analytics"].record(combined)

    # Check for exit intent
    lower = text.lower()
    if any(kw in lower for kw in settings.EXIT_KEYWORDS):
        farewell = "It was great talking to you! Take care and see you next time!"
        ctx["session"].log_turn(
            user_text=text, ai_reply=farewell,
            facial_emotion=_current_emotion,
            text_sentiment=sentiment_result,
            combined_mood=combined,
        )
        ctx["analytics"].print_report()
        ctx["session"].save()
        emit("ai_response", {
            "text": farewell,
            "emotion": _current_emotion,
            "confidence": _current_emotion_confidence,
            "farewell": True,
            "lang": ctx.get("lang", "en"),
        })
        ctx["processing"] = False
        return

    # LLM response — pass both facial emotion, confidence, and combined mood
    reply = ctx["chat"].get_response(
        text,
        facial_emotion=_current_emotion,
        facial_emotion_confidence=_current_emotion_confidence,
        combined_mood=combined,
        lang=lang,
    )

    ctx["session"].log_turn(
        user_text=text,
        ai_reply=reply,
        facial_emotion=_current_emotion,
        text_sentiment=sentiment_result,
        combined_mood=combined,
    )

    emit("ai_response", {
        "text": reply,
        "emotion": _current_emotion,
        "confidence": _current_emotion_confidence,
        "mood": combined,
        "lang": lang,
    })
    ctx["processing"] = False  # Done responding


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Companion — Web Server")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cam-index", type=int, default=0)
    args = parser.parse_args()

    _init_camera(not args.no_camera, args.cam_index)

    if _use_camera:
        threading.Thread(target=_emotion_loop, daemon=True).start()

    print(f"\n  AI Companion web server running at http://{args.host}:{args.port}\n")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
