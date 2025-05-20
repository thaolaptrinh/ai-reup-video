import logging
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

class NLLBTranslator:
    def __init__(self, device: str = "cuda"):
        """Initialize the NLLB translator.
        
        Args:
            device: The device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        self.model_name = "facebook/nllb-200-distilled-600M"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        
        # Set source and target languages
        self.src_lang = "zho_Hans"  # Chinese (Simplified)
        self.tgt_lang = "vie_Latn"  # Vietnamese
        
        logger.info(f"Initialized NLLB translator on {device}")

    def translate(self, segments: List[Dict]) -> List[Dict]:
        """Translate Chinese segments to Vietnamese while preserving timestamps.
        
        Args:
            segments: List of dictionaries containing Chinese text with timestamps
            
        Returns:
            List of dictionaries containing Vietnamese text with timestamps
        """
        try:
            translated_segments = []
            
            for segment in segments:
                # Prepare input text
                input_text = segment['text']
                
                # Tokenize with language codes
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    src_lang=self.src_lang
                ).to(self.device)
                
                # Generate translation
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                    max_length=512,
                    num_beams=5
                )
                
                # Decode translation
                translated_text = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]
                
                # Create new segment with translated text
                translated_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': translated_text,
                    'words': [
                        {
                            'word': word['word'],  # Keep original word timing
                            'start': word['start'],
                            'end': word['end']
                        }
                        for word in segment['words']
                    ] if segment['words'] else []
                }
                
                translated_segments.append(translated_segment)
            
            logger.info(f"Successfully translated {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            logger.error(f"Error translating segments: {str(e)}")
            raise 