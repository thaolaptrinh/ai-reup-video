import logging
from typing import List, Dict
from faster_whisper import WhisperModel
import torch

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, device: str = "cuda"):
        """Initialize the Whisper transcriber.
        
        Args:
            device: The device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.model = WhisperModel(
            "large-v3",
            device=device,
            compute_type="float16" if device == "cuda" else "float32"
        )
        logger.info(f"Initialized Whisper model on {device}")

    def transcribe(self, video_path: str) -> List[Dict]:
        """Transcribe audio from video file to Chinese text with timestamps.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of dictionaries containing transcribed segments with timestamps
        """
        try:
            # Transcribe with word-level timestamps
            segments, _ = self.model.transcribe(
                video_path,
                language="zh",
                task="transcribe",
                word_timestamps=True,
                beam_size=5
            )
            
            # Convert segments to list of dictionaries
            transcription = []
            for segment in segments:
                transcription.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end
                        }
                        for word in segment.words
                    ] if segment.words else []
                })
            
            logger.info(f"Successfully transcribed {len(transcription)} segments")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing video {video_path}: {str(e)}")
            raise 