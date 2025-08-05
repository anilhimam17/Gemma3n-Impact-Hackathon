from faster_whisper import WhisperModel
from pathlib import Path


class AudioTranscription:
    """Class provides the API for Audio Transcription using Whisper."""

    def __init__(self) -> None:
        self.model = WhisperModel("base", device="auto", compute_type="int8")

    def transcribe(self, file_path: list):
        segments, _ = self.model.transcribe(file_path[0])
        transcript = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            } 
            for segment in segments
        ]
        return transcript
    
    def check_audio(self, files: list) -> bool:
        for file in files:
            if Path(file).suffix == ".wav":
                return True
        return False