import logging
import os
import tempfile
from typing import List, Dict
from TTS.api import TTS
import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

class TTSSynthesizer:
    def __init__(self, device: str = "cuda"):
        """Initialize the TTS synthesizer.
        
        Args:
            device: The device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        
        # Initialize Coqui TTS with Vietnamese model
        self.tts = TTS(
            model_name="tts_models/vi/vivos/vits",
            progress_bar=False
        ).to(device)
        
        logger.info(f"Initialized TTS synthesizer on {device}")

    def synthesize(self, segments: List[Dict]) -> str:
        """Synthesize Vietnamese speech from translated segments.
        
        Args:
            segments: List of dictionaries containing Vietnamese text with timestamps
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Create temporary directory for audio files
            temp_dir = tempfile.mkdtemp()
            audio_segments = []
            
            # Synthesize each segment
            for i, segment in enumerate(segments):
                # Generate audio for segment
                audio = self.tts.tts(
                    text=segment['text'],
                    speaker_wav=None,  # Use default voice
                    language="vi"
                )
                
                # Save segment to temporary file
                segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
                sf.write(segment_path, audio, self.tts.synthesizer.output_sample_rate)
                audio_segments.append(segment_path)
            
            # Combine all segments into one audio file
            final_audio_path = os.path.join(temp_dir, "final_audio.wav")
            self._combine_audio_segments(audio_segments, final_audio_path)
            
            logger.info(f"Successfully synthesized audio: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            raise

    def _combine_audio_segments(self, segment_paths: List[str], output_path: str):
        """Combine multiple audio segments into one file.
        
        Args:
            segment_paths: List of paths to audio segment files
            output_path: Path to save the combined audio file
        """
        try:
            # Read and concatenate all segments
            combined_audio = []
            for path in segment_paths:
                audio, sample_rate = sf.read(path)
                combined_audio.append(audio)
            
            # Concatenate and save
            final_audio = np.concatenate(combined_audio)
            sf.write(output_path, final_audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Error combining audio segments: {str(e)}")
            raise 