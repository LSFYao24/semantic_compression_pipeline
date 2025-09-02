import cv2
import numpy as np
import time
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

# BLIP-2 imports
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
    print("✅ BLIP-2 transformers available")
except ImportError as e:
    print(f"❌ BLIP-2 not available: {e}")
    BLIP2_AVAILABLE = False


class PromptManager:
    """
    Manages different prompt strategies for VLM compression
    Based on complexity levels and desired detail levels
    """

    def __init__(self):
        self.prompt_templates = {
            'short': {
                'system': "You are an expert at describing images concisely.",
                'user': "Describe this image in one clear sentence."
            },
            'medium': {
                'system': "You are an expert at describing images with moderate detail.",
                'user': "Describe this image in 2-3 sentences. Include the main objects, their colors, and the setting or background."
            },
            'detailed': {
                'system': "You are an expert at describing images thoroughly for reconstruction.",
                'user': "Describe this image in detail including all visible objects, their colors, positions, spatial relationships, lighting conditions, and background elements. Make the description comprehensive and suitable for text-to-image generation."
            }
        }

        # Adaptive prompts based on image complexity
        self.adaptive_prompts = {
            'low': 'short',
            'medium': 'medium', 
            'high': 'detailed'
        }

    def get_prompt(self, detail_level: str = 'medium',
                   adaptive: bool = False,
                   complexity_level: str = None) -> Dict[str, str]:
        """
        Get prompt based on detail level or image complexity
        
        Args:
            detail_level: 'short', 'medium', or 'detailed'
            adaptive: Whether to use adaptive prompting based on complexity
            complexity_level: 'low', 'medium', or 'high' (for adaptive mode)
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        if adaptive and complexity_level:
            detail_level = self.adaptive_prompts.get(complexity_level, 'medium')
        
        return self.prompt_templates.get(detail_level, self.prompt_templates['medium'])


class VLMClient(ABC):
    """Abstract base class for VLM API clients"""
    
    @abstractmethod
    def describe_image(self, image: np.ndarray, prompt: Dict[str, str]) -> Dict:
        """Generate text description from image"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is properly configured"""
        pass


class BLIP2Client(VLMClient):
    """BLIP-2 client optimized for Google Colab"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load BLIP-2 model and processor"""
        if not BLIP2_AVAILABLE:
            print("Warning: BLIP-2 dependencies not available")
            return
        
        try:
            print(f"Loading BLIP-2 model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            # Load processor
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            
            # Load model with memory optimization for Colab
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto"           # Automatic device mapping
            )
            
            print("✅ BLIP-2 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading BLIP-2 model: {e}")
            import traceback
            traceback.print_exc()
            self.processor = None
            self.model = None
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        return Image.fromarray(image_rgb)
    
    def _adapt_prompt_for_blip2(self, base_prompt: str) -> str:
        """
        Adapt general prompts to BLIP-2 Q&A format
        BLIP-2 works better with question-answer format
        """
        base_lower = base_prompt.lower()
        
        if "one clear sentence" in base_lower:
            return "Question: What do you see in this image in one sentence? Answer:"
        elif "2-3 sentences" in base_lower:
            return "Question: Describe this image in detail including objects, colors, and setting. Answer:"
        elif "detail including all visible objects" in base_lower:
            return "Question: Provide a comprehensive detailed description of this image including all objects, colors, positions, lighting, and background elements suitable for image generation. Answer:"
        else:
            return f"Question: {base_prompt} Answer:"
    
    def _get_generation_params(self, detail_level: str) -> dict:
        """Get generation parameters based on detail level"""
        params = {
            'short': {
                'max_new_tokens': 50,
                'min_length': 10,
                'num_beams': 3,
                'do_sample': False,
                'repetition_penalty': 1.1
            },
            'medium': {
                'max_new_tokens': 100,
                'min_length': 30,
                'num_beams': 5,
                'do_sample': False,
                'repetition_penalty': 1.1
            },
            'detailed': {
                'max_new_tokens': 200,
                'min_length': 60,
                'num_beams': 5,
                'do_sample': False,
                'repetition_penalty': 1.2
            }
        }
        return params.get(detail_level, params['medium'])
    
    def describe_image(self, image: np.ndarray, prompt: Dict[str, str]) -> Dict:
        """Generate description using BLIP-2"""
        if not self.is_available():
            return {
                'success': False,
                'error': "BLIP-2 model not available. Please check model loading.",
                'model': self.model_name,
                'prompt_used': prompt
            }
        
        try:
            # Convert to PIL Image
            pil_image = self._numpy_to_pil(image)
            
            # Determine detail level from prompt
            detail_level = 'medium'  # default
            user_prompt_lower = prompt['user'].lower()
            if 'one' in user_prompt_lower and 'sentence' in user_prompt_lower:
                detail_level = 'short'
            elif 'detail' in user_prompt_lower or 'comprehensive' in user_prompt_lower:
                detail_level = 'detailed'
            
            # Adapt prompt for BLIP-2
            blip2_prompt = self._adapt_prompt_for_blip2(prompt['user'])
            
            # Get generation parameters
            gen_params = self._get_generation_params(detail_level)
            
            # Prepare inputs
            inputs = self.processor(pil_image, blip2_prompt, return_tensors="pt").to(self.device)
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            # Decode generated text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the output (remove the prompt part)
            if "Answer:" in generated_text:
                description = generated_text.split("Answer:")[-1].strip()
            else:
                description = generated_text.strip()
            
            # Remove any remaining question artifacts
            description = description.replace("Question:", "").strip()
            
            # Estimate token count
            estimated_tokens = len(description.split()) * 1.3
            
            return {
                'success': True,
                'description': description,
                'model': self.model_name,
                'prompt_used': prompt,
                'blip2_prompt': blip2_prompt,
                'generation_params': gen_params,
                'tokens_used': int(estimated_tokens),
                'device_used': self.device,
                'detail_level': detail_level
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': self.model_name,
                'prompt_used': prompt
            }
    
    def is_available(self) -> bool:
        """Check if BLIP-2 model is loaded and available"""
        return BLIP2_AVAILABLE and self.processor is not None and self.model is not None


class MockVLMClient(VLMClient):
    """Mock VLM client for testing without real models"""
    
    def __init__(self):
        self.model_name = "mock-vlm"
    
    def describe_image(self, image: np.ndarray, prompt: Dict[str, str]) -> Dict:
        """Generate mock description for testing"""
        height, width = image.shape[:2]
        mean_brightness = np.mean(image)
        
        # Generate different descriptions based on prompt level
        if 'one' in prompt['user'].lower() and 'sentence' in prompt['user'].lower():
            description = f"A {width}x{height} image with moderate brightness and various visual elements."
        elif '2-3 sentences' in prompt['user'].lower():
            description = f"This is a {width}x{height} pixel image with an average brightness of {mean_brightness:.1f}. The image contains various visual elements distributed across the frame. The overall composition appears to have a balanced visual structure."
        else:  # detailed
            description = f"This is a detailed {width}x{height} pixel digital image with an average brightness level of {mean_brightness:.1f} out of 255. The image features multiple visual elements with varying textures and spatial relationships throughout the composition. The lighting appears even with good contrast, and the overall visual structure suggests a complex scene with multiple objects or regions of interest distributed across the frame."
        
        return {
            'success': True,
            'description': description,
            'model': self.model_name,
            'prompt_used': prompt,
            'tokens_used': len(description.split()) * 1.3
        }
    
    def is_available(self) -> bool:
        """Mock client is always available"""
        return True


class VLMCompressor:
    """
    Main VLM compression orchestrator
    Handles image-to-text compression using various VLM models
    """
    
    def __init__(self, use_blip2: bool = True, use_mock: bool = True):
        self.prompt_manager = PromptManager()
        self.clients = {}
        
        # Add BLIP-2 as primary model for Colab
        if use_blip2:
            print("Initializing BLIP-2 client...")
            self.clients['blip2'] = BLIP2Client()
        
        # Always add mock client for testing
        if use_mock:
            self.clients['mock'] = MockVLMClient()
        
        # Set default client
        self.default_client = self._get_best_available_client()
        print(f"Default client set to: {self.default_client}")
    
    def _get_best_available_client(self) -> str:
        """Determine the best available client"""
        priority = ['blip2', 'mock']
        for client_name in priority:
            if client_name in self.clients and self.clients[client_name].is_available():
                return client_name
        return 'mock'  # Fallback
    
    def list_available_models(self) -> List[str]:
        """List all available VLM models"""
        available = []
        for name, client in self.clients.items():
            if client.is_available():
                available.append(name)
        return available
    
    def compress_image_to_text(self, image: np.ndarray,
                               model: str = None,
                               detail_level: str = 'medium',
                               adaptive: bool = False,
                               complexity_level: str = None) -> Dict:
        """
        Compress image to text description using specified VLM
        
        Args:
            image: Input image as numpy array
            model: VLM model to use ('blip2', 'mock')
            detail_level: Prompt detail level ('short', 'medium', 'detailed')
            adaptive: Whether to adapt prompt based on image complexity
            complexity_level: Image complexity level for adaptive prompting
            
        Returns:
            Compression result dictionary
        """
        # Use default client if not specified
        if model is None:
            model = self.default_client
        
        # Check if model is available
        if model not in self.clients:
            raise ValueError(f"Model '{model}' not available. Available models: {self.list_available_models()}")
        
        if not self.clients[model].is_available():
            raise RuntimeError(f"Model '{model}' is not properly configured")
        
        # Get appropriate prompt
        prompt = self.prompt_manager.get_prompt(
            detail_level=detail_level,
            adaptive=adaptive,
            complexity_level=complexity_level
        )
        
        # Record compression start time
        start_time = time.time()
        
        # Generate description
        result = self.clients[model].describe_image(image, prompt)
        
        # Calculate compression metrics
        compression_time = time.time() - start_time
        
        if result['success']:
            description = result['description']
            
            # Calculate compression ratio
            image_size_bytes = image.nbytes
            text_size_bytes = len(description.encode('utf-8'))
            compression_ratio = image_size_bytes / text_size_bytes if text_size_bytes > 0 else float('inf')
            
            # Add compression metrics
            result.update({
                'compression_time': compression_time,
                'original_size_bytes': image_size_bytes,
                'compressed_size_bytes': text_size_bytes,
                'compression_ratio': compression_ratio,
                'compression_efficiency': f"{compression_ratio:.2f}:1"
            })
        
        return result
    
    def batch_compress(self, images: List[np.ndarray],
                       model: str = None,
                       detail_level: str = 'medium',
                       adaptive: bool = True) -> List[Dict]:
        """
        Compress multiple images with optional adaptive prompting
        
        Args:
            images: List of images as numpy arrays
            model: VLM model to use
            detail_level: Base detail level for prompts
            adaptive: Whether to use adaptive prompting
            
        Returns:
            List of compression results
        """
        results = []
        
        # Import complexity analyzer if adaptive mode is enabled
        if adaptive:
            try:
                from complexity_analyzer import ComplexityAnalyzer
                analyzer = ComplexityAnalyzer()
            except ImportError:
                print("Warning: ComplexityAnalyzer not available, disabling adaptive mode")
                adaptive = False
        
        for i, image in enumerate(images):
            print(f"Processing image {i + 1}/{len(images)}...")
            
            # Determine complexity level if adaptive
            complexity_level = None
            if adaptive:
                complexity_result = analyzer.analyze_image(image, verbose=False)
                complexity_level = complexity_result['complexity_level']
            
            # Compress image
            result = self.compress_image_to_text(
                image=image,
                model=model,
                detail_level=detail_level,
                adaptive=adaptive,
                complexity_level=complexity_level
            )
            
            # Add image index and complexity info
            result['image_index'] = i
            if complexity_level:
                result['detected_complexity'] = complexity_level
            
            results.append(result)
        
        return results


# Usage example and testing
if __name__ == "__main__":
    print("VLM Compressor initialized for Colab environment")
    print("Supported models: BLIP-2 (Salesforce/blip2-opt-2.7b)")
    
    # Test initialization
    compressor = VLMCompressor(use_blip2=True, use_mock=True)
    print(f"Available models: {compressor.list_available_models()}")
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test compression
    result = compressor.compress_image_to_text(
        image=test_image,
        detail_level='medium'
    )
    
    if result['success']:
        print(f"✅ Test successful!")
        print(f"Description: {result['description']}")
        print(f"Model used: {result['model']}")
    else:
        print(f"❌ Test failed: {result['error']}")
