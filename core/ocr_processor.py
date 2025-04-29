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
        try:
            # Convert to numpy array
            img = np.array(image)
            
            # Skip if image is too small or invalid
            if img.size == 0 or img.ndim < 2:
                return image
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Resize if image is too small - helps with low-resolution embedded images
            if img.shape[0] < 1000 or img.shape[1] < 1000:
                # Calculate scale factor based on image size
                target_height = 1500
                scale_factor = target_height / img.shape[0]
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Create multiple processing variants and choose the best result
            variants = []
            
            # Variant 1: Standard processing with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img1 = clahe.apply(img)
            
            # Use adaptive thresholding with params optimized for text
            threshold1 = cv2.adaptiveThreshold(
                img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9
            )
            
            # Apply morphological operations to enhance text
            kernel = np.ones((1, 1), np.uint8)
            threshold1 = cv2.morphologyEx(threshold1, cv2.MORPH_CLOSE, kernel)
            variants.append(threshold1)
            
            # Variant 2: More aggressive processing for faded or light text
            img2 = img.copy()
            # Increase contrast using histogram equalization
            img2 = cv2.equalizeHist(img2)
            
            # Use more aggressive thresholding for faint text
            threshold2 = cv2.adaptiveThreshold(
                img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15
            )
            
            # More aggressive morphological operations
            kernel2 = np.ones((2, 2), np.uint8)
            threshold2 = cv2.morphologyEx(threshold2, cv2.MORPH_CLOSE, kernel2)
            
            # Dilate slightly to make text more prominent
            threshold2 = cv2.dilate(threshold2, np.ones((1, 1), np.uint8), iterations=1)
            variants.append(threshold2)
            
            # Variant 3: Processing optimized for structured content (tables, indexes)
            img3 = img.copy()
            # Use bilateral filtering to preserve edges in tables
            img3 = cv2.bilateralFilter(img3, 9, 75, 75)
            
            # Apply Otsu's thresholding which works well on bimodal images like tables
            _, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use appropriate morphological operations for structured text
            kernel3 = np.ones((1, 1), np.uint8)
            threshold3 = cv2.morphologyEx(threshold3, cv2.MORPH_OPEN, kernel3)
            variants.append(threshold3)
            
            # Apply denoising to all variants
            for i in range(len(variants)):
                variants[i] = cv2.fastNlMeansDenoising(variants[i], None, 10, 7, 21)
            
            # Select the best variant based on text clarity (count of connected components)
            best_variant = threshold1  # Default to first variant
            max_components = 0
            
            for variant in variants:
                # Count text-like connected components
                contours, _ = cv2.findContours(255-variant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_like = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    # Filter by size and aspect ratio to find text-like components
                    if w > 5 and h > 5 and aspect_ratio < 10 and aspect_ratio > 0.1:
                        text_like += 1
                        
                if text_like > max_components:
                    max_components = text_like
                    best_variant = variant
            
            return Image.fromarray(best_variant)
            
        except Exception as e:
            # Log the error but return original image if processing fails
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            return image

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