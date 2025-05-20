#!/usr/bin/env python3

import os
import logging
from pathlib import Path
from typing import List
import torch
from tqdm import tqdm

from src.asr.transcriber import WhisperTranscriber
from src.translation.translator import NLLBTranslator
from src.tts.synthesizer import TTSSynthesizer
from src.video.processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoProcessingPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.transcriber = WhisperTranscriber(device=self.device)
        self.translator = NLLBTranslator(device=self.device)
        self.synthesizer = TTSSynthesizer(device=self.device)
        self.video_processor = VideoProcessor()

    def process_video(self, input_path: str, output_path: str):
        """Process a single video file through the pipeline."""
        try:
            logger.info(f"Processing video: {input_path}")
            
            # 1. Extract audio and transcribe
            logger.info("Transcribing Chinese audio...")
            transcription = self.transcriber.transcribe(input_path)
            
            # 2. Translate to Vietnamese
            logger.info("Translating to Vietnamese...")
            translation = self.translator.translate(transcription)
            
            # 3. Generate Vietnamese speech
            logger.info("Generating Vietnamese speech...")
            audio_path = self.synthesizer.synthesize(translation)
            
            # 4. Process video with copyright protection
            logger.info("Applying video modifications...")
            self.video_processor.process(
                input_path=input_path,
                output_path=output_path,
                audio_path=audio_path,
                subtitles=translation
            )
            
            logger.info(f"Successfully processed video: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing video {input_path}: {str(e)}")
            raise

    def process_directory(self, input_dir: str, output_dir: str):
        """Process all videos in the input directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        video_files = [
            f for f in input_path.glob('**/*')
            if f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_file in tqdm(video_files, desc="Processing videos"):
            relative_path = video_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.mp4')
            
            # Create output subdirectories if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.process_video(str(video_file), str(output_file))

def main():
    # Get input and output directories
    input_dir = os.getenv('INPUT_DIR', 'input')
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    
    # Initialize and run pipeline
    pipeline = VideoProcessingPipeline()
    pipeline.process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main() 