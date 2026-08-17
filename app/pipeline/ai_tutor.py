"""
ai_tutor.py
Gemini-powered Raga Learning Coach.
Uses the free gemini-flash-latest model to analyse a practice session
result and return personalised, educational feedback.
"""
import os
import logging
import json

logger = logging.getLogger(__name__)


def _build_prompt(analysis: dict, target_raga: str | None, borgeet_mode: bool) -> str:
    """Build a structured prompt from the pipeline result."""
    top_match = analysis.get("raga_candidates", [{}])[0]
    detected_raga = top_match.get("raga", "unknown")
    confidence = round(top_match.get("score", 0) * 100, 1)
    tonic = analysis.get("tonic_nearest_note", "unknown")
    dominant_notes = [n["note"] for n in analysis.get("dominant_notes", [])[:5]]
    histogram = analysis.get("swara_histogram", [])
    SWARA_LABELS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
    
    if isinstance(histogram, list) and len(histogram) == 12:
        note_pairs = [(SWARA_LABELS[i], val * 100) for i, val in enumerate(histogram)]
    elif isinstance(histogram, dict):
        note_pairs = list(histogram.items())
    else:
        note_pairs = []

    # Build a sorted list of (note, pct) for the top notes
    sorted_notes = sorted(note_pairs, key=lambda x: x[1], reverse=True)[:7]
    note_summary = ", ".join(f"{n}={round(v, 1)}%" for n, v in sorted_notes)

    tradition = "Borgeet/Sattriya" if borgeet_mode else "Hindustani Classical"

    if target_raga:
        task = f"""The student was practising **{target_raga}** ({tradition} tradition). 
The system detected **{detected_raga}** instead (confidence: {confidence}%).
Tell them:
1. What they did correctly (notes in common between {target_raga} and what they sang).
2. Exactly which notes they sang that do NOT belong to {target_raga}.
3. Concrete, actionable tips to fix their intonation or note selection.
4. A short (2-sentence) encouraging closing remark."""
    else:
        task = f"""The student recorded a free-form practice session without specifying a target raga.
The system identified **{detected_raga}** ({tradition} tradition) with {confidence}% confidence.
Tell them:
1. Which raga their singing most resembles and why.
2. Their strongest notes (vadi/samvadi analysis).
3. Notes that appeared but do not fit well into the detected raga.
4. A suggested exercise or alap phrase they can practise next.
5. A short encouraging closing remark."""

    prompt = f"""You are SWARAAG-Guru, an expert in {tradition} vocal music with deep knowledge 
of Borgeet (Assamese devotional music by Srimanta Sankardev) and Hindustani classical ragas.
A student just recorded a practice session. Here is the acoustic analysis from the SWARAAG pipeline:

**Detected tonic (Sa):** {tonic}
**Top raga match:** {detected_raga} ({confidence}% confidence)
**Dominant swaras sung:** {', '.join(dominant_notes)}
**Full swara distribution:** {note_summary}

{task}

OUTPUT FORMAT:
You must return a valid JSON object with exactly two keys:
1. "markdown_feedback": A detailed, warm, encouraging response formatted with markdown headers (##). Keep under 350 words. Use swara notation (Sa, Re, Ga).
2. "spoken_summary": A 1 to 2 sentence conversational summary of the analysis that will be read aloud by a text-to-speech engine. Example: "You sang Sa and Pa perfectly, but try adding a little Komal Ga to capture the true feeling of Dhanashri."

IMPORTANT: Output ONLY valid JSON. Do not wrap it in markdown code blocks."""
    return prompt


def generate_tutor_feedback(analysis: dict,
                             target_raga: str | None = None,
                             borgeet_mode: bool = True,
                             api_key: str | None = None) -> dict:
    """
    Call Gemini to generate tutor feedback.
    Returns a dict: { "feedback": str, "model": str, "error": str | None }
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {
            "feedback": None,
            "model": None,
            "error": "GEMINI_API_KEY not set. Add it to your .env or environment variables."
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-flash-latest")
        prompt = _build_prompt(analysis, target_raga, borgeet_mode)
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Try to parse JSON
        if raw_text.startswith("`json"):
            raw_text = raw_text.removeprefix("`json").removesuffix("`").strip()
            
        try:
            import json
            parsed = json.loads(raw_text)
            feedback_text = parsed.get("markdown_feedback", raw_text)
            spoken_summary = parsed.get("spoken_summary", "")
        except Exception as e:
            logger.warning(f"Failed to parse Gemini JSON: {e}")
            feedback_text = raw_text
            spoken_summary = ""

        audio_b64 = None
        if spoken_summary:
            try:
                import edge_tts
                import asyncio
                import base64
                
                # Fix TTS pronunciation of Swaras
                import re
                phonetic_text = spoken_summary
                replacements = {
                    r"\bSa\b": "Saa",
                    r"\bRe\b": "Ray",
                    r"\bGa\b": "Gaa",
                    r"\bMa\b": "Maa",
                    r"\bPa\b": "Paa",
                    r"\bDha\b": "Dhaa",
                    r"\bNi\b": "Nee"
                }
                for pattern, phon in replacements.items():
                    phonetic_text = re.sub(pattern, phon, phonetic_text)

                async def generate_tts():
                    communicate = edge_tts.Communicate(phonetic_text, "en-IN-NeerjaNeural")
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    return audio_data
                
                # We need a new event loop since this might be called in a sync thread by FastAPI
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                if loop.is_running():
                    # This happens if FastAPI uses an async def route but calls sync def function
                    import nest_asyncio
                    nest_asyncio.apply()
                    audio_bytes = loop.run_until_complete(generate_tts())
                else:
                    audio_bytes = loop.run_until_complete(generate_tts())
                    
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                logger.info(f"Generated TTS audio ({len(audio_b64)} bytes base64)")
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")

        logger.info("Gemini tutor feedback generated (%d chars)", len(feedback_text))
        return {
            "feedback": feedback_text,
            "audio_base64": audio_b64,
            "model": "gemini-flash-latest",
            "error": None,
        }
    except Exception as e:
        logger.error("Gemini tutor failed: %s", e)
        return {
            "feedback": None,
            "model": None,
            "error": str(e),
        }
