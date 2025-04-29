import os
import pytesseract
from PIL import Image
import numpy as np
import cv2
from typing import Optional, List, Dict, Any
import logging
from pdf2image import convert_from_path
import re

class OCRProcessor:
    def __init__(self, language: str = "eng+urd+mar+hin"):
        self.language = language
        self.tesseract_path = os.getenv('TESSERACT_PATH')
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Language-specific configurations
        self.language_configs = {
            'urd': {
                'psm': 6,  # Assume uniform block of text
                'oem': 1,  # LSTM only
                'config': '--psm 6 --oem 1 -c tessedit_char_whitelist=ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھیےآأإؤئى'
            },
            'mar': {
                'psm': 6,
                'oem': 1,
                'config': '--psm 6 --oem 1'
            },
            'eng': {
                'psm': 3,  # Fully automatic page segmentation
                'oem': 3,  # Default OCR engine mode
                'config': '--psm 3 --oem 3'
            },
            'hin': {
                'psm': 3,
                'oem': 3,
                'config': '--psm 3 --oem 3'
            }
        }

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img)
        
        return Image.fromarray(img)

    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text for each language
            texts = []
            for lang in self.language.split('+'):
                if lang in self.language_configs:
                    config = self.language_configs[lang]
                    text = pytesseract.image_to_string(
                        processed_image,
                        lang=lang,
                        config=config['config']
                    )
                    texts.append(text)
            
            # Combine texts and clean up
            combined_text = ' '.join(texts)
            return self._clean_text(combined_text)
            
        except Exception as e:
            self.logger.error(f"Error in OCR processing: {str(e)}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Unicode characters
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        
        # Remove special characters while preserving language-specific characters
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0900-\u097F]', ' ', text)
        
        return text.strip()

    def process_pdf(self, pdf_path: str) -> str:
        """Process PDF file and extract text"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each page
            texts = []
            for image in images:
                text = self.extract_text(image)
                if text:
                    texts.append(text)
            
            return '\n\n'.join(texts)
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            return ""

    def process_image(self, image_path: str) -> str:
        """Process image file and extract text"""
        try:
            image = Image.open(image_path)
            return self.extract_text(image)
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return "" 