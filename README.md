# AI Video Processing Tool

A powerful command-line tool for batch processing videos with AI-powered features:
- Chinese to Vietnamese translation with subtitles
- Voice synthesis in Vietnamese
- Copyright protection modifications
- High-performance GPU acceleration

## Features

- 🎥 Batch video processing from input directory
- 🔄 Automatic Chinese to Vietnamese translation
- 🎙️ High-quality Vietnamese voice synthesis
- 🛡️ Copyright protection modifications
- 🚀 GPU-accelerated processing
- 🐳 Docker containerization

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker and NVIDIA Container Toolkit
- At least 16GB RAM
- 50GB free disk space

## Quick Start

1. Place your Chinese videos in the `input/` directory
2. Build and run the Docker container:

```bash
# Build the container
docker build -t ai-video-processor .

# Run the container
docker run --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output ai-video-processor
```

## Directory Structure

```
.
├── input/          # Place your input videos here
├── output/         # Processed videos will be saved here
├── src/            # Source code
│   ├── asr/        # Speech recognition modules
│   ├── translation/# Translation modules
│   ├── tts/        # Text-to-speech modules
│   └── video/      # Video processing modules
├── Dockerfile      # Docker configuration
└── requirements.txt# Python dependencies
```

## Processing Pipeline

1. Video files are detected in the input directory
2. Chinese audio is transcribed using Whisper
3. Text is translated to Vietnamese
4. Vietnamese text is converted to speech
5. Copyright protection modifications are applied
6. Final video is rendered with subtitles

## Performance

- Processing speed depends on GPU capabilities
- Typical processing time: 2-3x video duration
- Supports batch processing of multiple videos

## License

This project uses open-source models and is released under the MIT License. 