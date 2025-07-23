import cv2
import numpy as np
from PIL import Image
import os
from typing import Union, Tuple, Optional


class ImageLoader:
    """Image loading and preprocessing class"""

    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image file

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array in BGR format
        """
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} does not exist")
            return None

        # Check file format
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.supported_formats:
            print(f"Error: Unsupported file format {ext}")
            return None

        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image {image_path}")
                return None

            print(f"Successfully loaded image: {image_path}")
            print(f"Image shape: {image.shape}")
            return image

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def load_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Load image from URL

        Args:
            url: Image URL

        Returns:
            Image as numpy array
        """
        try:
            import requests
            from io import BytesIO

            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            print(f"Successfully loaded image from URL: {url}")
            print(f"Image shape: {image_cv.shape}")
            return image_cv

        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None

    def preprocess_image(self, image: np.ndarray,
                         resize: Optional[Tuple[int, int]] = None,
                         normalize: bool = False) -> np.ndarray:
        """
        Image preprocessing

        Args:
            image: Input image
            resize: Resize dimensions (width, height)
            normalize: Whether to normalize to [0,1]

        Returns:
            Preprocessed image
        """
        processed_image = image.copy()

        # Resize
        if resize:
            processed_image = cv2.resize(processed_image, resize)
            print(f"Image resized to: {resize}")

        # Normalize
        if normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
            print("Image normalized to [0,1] range")

        return processed_image

    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get basic image information

        Args:
            image: Input image

        Returns:
            Dictionary containing image information
        """
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1

        info = {
            'width': width,
            'height': height,
            'channels': channels,
            'total_pixels': width * height,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024)
        }

        return info

    def display_image_info(self, image: np.ndarray, title: str = "Image Info"):
        """Display basic image information"""
        info = self.get_image_info(image)

        print(f"\n=== {title} ===")
        print(f"Width: {info['width']} pixels")
        print(f"Height: {info['height']} pixels")
        print(f"Channels: {info['channels']}")
        print(f"Total pixels: {info['total_pixels']:,}")
        print(f"Data type: {info['dtype']}")
        print(f"File size: {info['size_mb']:.2f} MB")


# Usage example
if __name__ == "__main__":
    # Create image loader instance
    loader = ImageLoader()

    # Example usage
    print("Image loader initialized")
    print("Supported formats:", loader.supported_formats)

    # Example for loading local image
    # image = loader.load_image("path/to/your/image.jpg")
    # if image is not None:
    #     loader.display_image_info(image)
    #     processed_image = loader.preprocess_image(image, resize=(224, 224))