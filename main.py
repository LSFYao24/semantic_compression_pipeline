#!/usr/bin/env python3
"""
Semantic Compression Pipeline
Main entry point for the semantic compression analysis project
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/descriptions', 
        'data/reconstructed',
        'data/results',
        'configs',
        'notebooks'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def test_image_loader():
    """Test the image loader module"""
    try:
        from image_loader import ImageLoader
        
        loader = ImageLoader()
        print("✓ ImageLoader imported successfully")
        print(f"Supported formats: {loader.supported_formats}")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import ImageLoader: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Semantic Compression Pipeline')
    parser.add_argument('--setup', action='store_true', 
                       help='Setup project directories')
    parser.add_argument('--test', action='store_true',
                       help='Run basic tests')
    parser.add_argument('--image', type=str,
                       help='Test with a specific image file')
    
    args = parser.parse_args()
    
    print("=== Semantic Compression Pipeline ===")
    
    if args.setup:
        print("Setting up project directories...")
        setup_directories()
        print("Setup complete!")
        return
    
    if args.test:
        print("Running basic tests...")
        success = test_image_loader()
        
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed!")
        return
    
    if args.image:
        print(f"Testing with image: {args.image}")
        
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} does not exist")
            return
        
        try:
            from image_loader import ImageLoader
            
            loader = ImageLoader()
            image = loader.load_image(args.image)
            
            if image is not None:
                loader.display_image_info(image, "Test Image")
                print("✓ Image loaded successfully!")
            else:
                print("✗ Failed to load image")
                
        except ImportError as e:
            print(f"Error importing modules: {e}")
        
        return
    
    # Default behavior
    print("Usage:")
    print("  python main.py --setup          # Setup project directories")
    print("  python main.py --test           # Run basic tests") 
    print("  python main.py --image <path>   # Test with specific image")
    print("\nFor more options, use: python main.py --help")

if __name__ == "__main__":
    main()
