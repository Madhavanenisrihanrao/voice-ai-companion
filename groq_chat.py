"""
groq_chat.py
------------
Conversational AI powered by the Groq API (free tier).

Sends conversation history and the user's current emotion to a
large-language model (LLaMA 3) and returns an empathetic response.
"""

import os
# Suppress TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from groq import Groq
import settings
from logger_setup import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# System prompt that shapes the AI companion's personality
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a warm, empathetic AI companion. "
    "Your goal is to make the user feel heard, supported, and uplifted. "
    "You are friendly, kind, and genuinely interested in the user's well-being. "
    "Keep your responses concise (2-4 sentences) and conversational — "
    "as if you are a caring friend speaking naturally. "
    "Never be preachy or give unsolicited long advice. "
    "Adapt your tone to the user's current emotional state. "
    "IMPORTANT: You are multilingual. Always reply in THE SAME LANGUAGE "
    "the user is speaking. If they write in Telugu, reply in Telugu. "
    "If they write in Hindi, reply in Hindi. Match their language exactly."
)

# Human-readable language names for the system message
_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
    "it": "Italian",
}


class GroqChat:
    """Conversational AI wrapper around the Groq SDK."""

    def __init__(
        self,
        model: str = settings.GROQ_MODEL,
        max_history: int = settings.MAX_CONVERSATION_HISTORY,
    ):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set.\n"
                "1. Get a free API key at https://console.groq.com\n"
                "2. Create a .env file with: GROQ_API_KEY=your_key_here"
            )

        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_history = max_history
        self._conversation_history: list[dict] = []

        log.info("Initialised with model: %s", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_response(
        self,
        user_message: str,
        facial_emotion: str = "neutral",
        facial_emotion_confidence: float = 1.0,
        combined_mood: str = "neutral",
        lang: str = "en",
    ) -> str:
        """Send a user message (with emotion context) and return the AI's reply.

        Args:
            user_message:   The text the user said.
            facial_emotion: The user's current facial expression from the camera.
            facial_emotion_confidence: Confidence score of the facial expression.
            combined_mood:  Combined score of facial + text sentiment.
            lang:           ISO 639-1 language code detected from the user's text.

        Returns:
            The AI companion's reply as a string.
        """
        lang_name = _LANG_NAMES.get(lang, "English")
        conf_percent = round(facial_emotion_confidence * 100)

        # Build dynamic uncertainty instructions for the LLM based on confidence brackets
        if facial_emotion_confidence > 0.70:
            emotion_instructions = (
                f"- The user's FACIAL EXPRESSION is detected with HIGH confidence: "
                f"{facial_emotion.capitalize()} ({conf_percent}% confidence).\n"
                f"- IMPORTANT: You MUST refer to their emotion scientifically and warm-neutrally. "
                f"Do not assume or claim to know their feelings (e.g. do NOT say 'You look like you're having a great day!'). "
                f"Instead, state it as: 'I detected a {facial_emotion} facial expression ({conf_percent}% confidence). How are you feeling today?' "
                f"Or adapt it similarly to be objective and scientific, yet friendly and supportive."
            )
        elif facial_emotion_confidence >= 0.40:
            emotion_instructions = (
                f"- The user's FACIAL EXPRESSION is detected with MEDIUM confidence: "
                f"Possible facial expression: {facial_emotion.capitalize()} ({conf_percent}% confidence).\n"
                f"- IMPORTANT: Since the confidence is low/medium (40%-70%), do NOT assume they feel this way. "
                f"Instead, refer to it only as a possible expression and ask neutral check-in questions. "
                f"For example, you could say: 'Possible facial expression: {facial_emotion.capitalize()} ({conf_percent}% confidence). How is your day going?' "
                f"Do not act certain or assume emotion."
            )
        else:
            emotion_instructions = (
                f"- The user's FACIAL EXPRESSION is UNCERTAIN (confidence {conf_percent}% is under 40%).\n"
                f"- IMPORTANT: Because it is completely uncertain, DO NOT assume any emotion, DO NOT refer to any facial cues or expressions. "
                f"Be warm, friendly, and completely neutral in your tone."
            )

        # Build the system message with emotion + language context
        system_message = (
            f"{SYSTEM_PROMPT}\n\n"
            f"EMOTION CONTEXT:\n"
            f"{emotion_instructions}\n"
            f"- The user's overall mood (face + text combined) is: {combined_mood}\n\n"
            f"LANGUAGE: The user is writing in {lang_name}. "
            f"You MUST reply in {lang_name}."
        )

        # Append the user message to history
        self._conversation_history.append({"role": "user", "content": user_message})

        # Trim history if it exceeds the limit
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        # Assemble the full message list for the API call
        messages = [{"role": "system", "content": system_message}] + self._conversation_history

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            log.error("API error: %s", exc)
            reply = (
                "I'm sorry, I'm having a little trouble thinking right now. "
                "Could you say that again?"
            )

        # Store the assistant reply in history
        self._conversation_history.append({"role": "assistant", "content": reply})

        log.info("AI response: %s", reply)
        return reply

    def get_opening_message(self, emotion: str, confidence: float = 1.0) -> str:
        """Generate a context-aware opening message based on the detected emotion and confidence.

        This is used when the AI companion starts the conversation.

        Args:
            emotion: The detected facial emotion.
            confidence: The confidence of the detected facial emotion.

        Returns:
            An empathetic opening message.
        """
        conf_percent = round(confidence * 100)
        if confidence > 0.70:
            prompt = (
                f"Greet the user warmly and scientifically state the detected facial expression. "
                f"You MUST say something like: 'I detected a {emotion} facial expression ({conf_percent}% confidence). How are you feeling today?' "
                f"Keep it to 1-2 sentences."
            )
        elif confidence >= 0.40:
            prompt = (
                f"Greet the user warmly and mention that you detected a possible facial expression of '{emotion}' with {conf_percent}% confidence. "
                f"Ask a warm, neutral check-in question, without assuming their feelings. Keep it to 1-2 sentences."
            )
        else:
            prompt = (
                f"Greet the user warmly and neutrally. Since the facial expression is uncertain (confidence under 40%), "
                f"do not mention any facial expressions or emotions. Just greet them and ask how they are doing. Keep it to 1-2 sentences."
            )

        return self.get_response(
            prompt,
            facial_emotion=emotion,
            facial_emotion_confidence=confidence
        )

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._conversation_history.clear()
        log.info("Conversation history cleared.")
