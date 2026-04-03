# stt_service.py

import requests

class STTService:
    def __init__(self, url="http://localhost:9000"):
        self.url = url

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            files = {
                "file": ("audio.wav", f, "audio/wav"),
            }

            data = {
                "language": "en-US",
            }

            response = requests.post(
                f"{self.url}/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=30
            )

        if response.status_code != 200:
            raise Exception(f"STT error: {response.text}")

        return response.json()["text"]