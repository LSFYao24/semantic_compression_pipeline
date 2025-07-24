import sys
sys.path.append('src')
from image_loader import ImageLoader
from complexity_analyzer import ComplexityAnalyzer

# Initialize modules
loader = ImageLoader()
analyzer = ComplexityAnalyzer()

# Test with real image (you need to provide a path)
image_path = "/mnt/e/USC/Summer Project/semantic_compression_pipeline/test_pics/test_pic_1.jpg"  # Change this to actual image path
image = loader.load_image(image_path)

if image is not None:
    results = analyzer.analyze_image(image)
    print(f"Image: {image_path}")
    print(f"Complexity: {results['complexity_level']}")
else:
    print("Please provide a valid image path")
