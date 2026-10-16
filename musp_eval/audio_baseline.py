"""How the audio-input models were queried on their own (the gray rows and the "Audio input" bars).

Each question is one request: the system prompt below, the recording of its piece, and the question block.
Recordings are sent inline as 16 kHz mono 64 kbit/s MP3 (the longest, about 12.5 minutes, is about 6 MB).
Keys are read from the environment: DASHSCOPE_API_KEY (Qwen on Alibaba Cloud Model Studio) and GEMINI_API_KEY.
"""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

from .questions import question_block

SYSTEM_AUDIO = (
    "You are an expert musician listening to a recording of a classical music performance. "
    "Answer the question about this recording. Timestamps refer to the recording as given. "
    "Think as needed, then end your reply with one line of the form\n"
    "Final answer: <answer>\n"
    "using exactly the requested answer format."
)
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def encode_mp3(wav: Path, out: Path) -> Path:
    """16 kHz mono 64 kbit/s MP3, the format every audio-input model received."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
                    "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", str(out)], check=True)
    return out


def ask_qwen(model: str, mp3: Path, row, max_tokens: int = 8192, temperature: float = 0.0) -> str:
    """Qwen omni models through DashScope's OpenAI-compatible endpoint (omni models require streaming)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=DASHSCOPE_URL)
    audio = base64.b64encode(mp3.read_bytes()).decode()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_AUDIO},
                  {"role": "user", "content": [
                      {"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio}", "format": "mp3"}},
                      {"type": "text", "text": question_block(row)}]}],
        modalities=["text"], max_tokens=max_tokens, temperature=temperature, stream=True)
    return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)


def ask_gemini(model: str, mp3: Path, row, max_tokens: int = 32768) -> str:
    """Gemini through the google-genai SDK: default thinking, provider-default temperature."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=mp3.read_bytes(), mime_type="audio/mp3"), question_block(row)],
        config=types.GenerateContentConfig(system_instruction=SYSTEM_AUDIO, max_output_tokens=max_tokens))
    return r.text or ""
