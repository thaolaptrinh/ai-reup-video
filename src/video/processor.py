import logging
import os
import tempfile
from typing import List, Dict
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import ffmpeg

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor."""
        self.temp_dir = tempfile.mkdtemp()
        logger.info("Initialized video processor")

    def process(self, input_path: str, output_path: str, audio_path: str, subtitles: List[Dict]):
        """Process video with copyright protection and subtitles.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save processed video
            audio_path: Path to generated audio file
            subtitles: List of subtitle segments with timestamps
        """
        try:
            # Load video
            video = VideoFileClip(input_path)
            
            # Apply copyright protection modifications
            modified_video = self._apply_copyright_protection(video)
            
            # Add subtitles
            video_with_subs = self._add_subtitles(modified_video, subtitles)
            
            # Replace audio
            final_video = self._replace_audio(video_with_subs, audio_path)
            
            # Write final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                threads=4,
                preset='medium'
            )
            
            # Clean up
            video.close()
            final_video.close()
            
            logger.info(f"Successfully processed video: {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def _apply_copyright_protection(self, video: VideoFileClip) -> VideoFileClip:
        """Apply copyright protection modifications to video.
        
        Args:
            video: Input video clip
            
        Returns:
            Modified video clip
        """
        try:
            # Apply minor modifications to avoid detection
            def modify_frame(frame):
                # Add slight noise
                noise = np.random.normal(0, 1, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)
                
                # Slight color shift
                frame = cv2.convertScaleAbs(frame, alpha=1.02, beta=1)
                
                return frame
            
            # Apply modifications
            modified_video = video.fl_image(modify_frame)
            
            # Slightly change speed
            modified_video = modified_video.speedx(1.02)
            
            return modified_video
            
        except Exception as e:
            logger.error(f"Error applying copyright protection: {str(e)}")
            raise

    def _add_subtitles(self, video: VideoFileClip, subtitles: List[Dict]) -> VideoFileClip:
        """Add subtitles to video.
        
        Args:
            video: Input video clip
            subtitles: List of subtitle segments
            
        Returns:
            Video clip with subtitles
        """
        try:
            # Create subtitle clips
            subtitle_clips = []
            
            for segment in subtitles:
                # Create text clip
                txt_clip = TextClip(
                    segment['text'],
                    fontsize=24,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial',
                    method='caption',
                    size=(video.w, None)
                )
                
                # Position at bottom of video
                txt_clip = txt_clip.set_position(('center', 'bottom'))
                
                # Set timing
                txt_clip = txt_clip.set_start(segment['start']).set_end(segment['end'])
                
                subtitle_clips.append(txt_clip)
            
            # Combine video with subtitles
            video_with_subs = CompositeVideoClip([video] + subtitle_clips)
            
            return video_with_subs
            
        except Exception as e:
            logger.error(f"Error adding subtitles: {str(e)}")
            raise

    def _replace_audio(self, video: VideoFileClip, audio_path: str) -> VideoFileClip:
        """Replace video audio with new audio.
        
        Args:
            video: Input video clip
            audio_path: Path to new audio file
            
        Returns:
            Video clip with new audio
        """
        try:
            # Load new audio
            new_audio = AudioFileClip(audio_path)
            
            # Match audio duration to video
            if new_audio.duration > video.duration:
                new_audio = new_audio.subclip(0, video.duration)
            else:
                # Loop audio if shorter than video
                n_loops = int(np.ceil(video.duration / new_audio.duration))
                new_audio = new_audio.loop(n_loops)
                new_audio = new_audio.subclip(0, video.duration)
            
            # Set new audio
            video = video.set_audio(new_audio)
            
            return video
            
        except Exception as e:
            logger.error(f"Error replacing audio: {str(e)}")
            raise 