import sys
sys.path.append('src')
import numpy as np
import cv2
from image_loader import ImageLoader
from complexity_analyzer import ComplexityAnalyzer

def create_solid_color_image(width=200, height=200, color=(100, 150, 200)):
    """
    Create a solid color image (lowest complexity)
    
    Args:
        width: Image width
        height: Image height  
        color: BGR color tuple
        
    Returns:
        Solid color image as numpy array
    """
    # Create image filled with single color
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def create_simple_geometric_image(width=200, height=200):
    """
    Create a simple geometric shape image (low complexity)
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Simple geometric image as numpy array
    """
    # Create white background
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    
    # Draw a simple blue rectangle in center
    start_x, start_y = width//4, height//4
    end_x, end_y = 3*width//4, 3*height//4
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), -1)
    
    return image

def test_complexity_progression():
    """Test different complexity levels"""
    
    # Initialize modules
    loader = ImageLoader()
    analyzer = ComplexityAnalyzer()
    
    print("="*60)
    print("TESTING COMPLEXITY PROGRESSION")
    print("="*60)
    
    # Test 1: Pure solid color (should be LOW complexity)
    print("\n1. SOLID COLOR IMAGE (Expected: LOW complexity)")
    print("-" * 50)
    solid_image = create_solid_color_image(color=(0, 128, 255))  # Orange color
    results1 = analyzer.analyze_image(solid_image, verbose=True)
    
    # Test 2: Simple geometric shape (should be LOW-MEDIUM complexity)  
    print("\n2. SIMPLE GEOMETRIC SHAPE (Expected: LOW-MEDIUM complexity)")
    print("-" * 50)
    geometric_image = create_simple_geometric_image()
    results2 = analyzer.analyze_image(geometric_image, verbose=True)
    
    # Test 3: Random noise (should be MEDIUM-HIGH complexity)
    print("\n3. RANDOM NOISE (Expected: MEDIUM-HIGH complexity)")
    print("-" * 50)
    noise_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    results3 = analyzer.analyze_image(noise_image, verbose=True)
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("="*60)
    
    test_cases = [
        ("Solid Color", results1),
        ("Simple Geometric", results2), 
        ("Random Noise", results3)
    ]
    
    print(f"{'Image Type':<20} {'Score':<8} {'Level':<10} {'Edge%':<8} {'Entropy%':<10} {'ColorVar%'}")
    print("-" * 70)
    
    for name, results in test_cases:
        score = results['overall_score']
        level = results['complexity_level']
        edge_pct = results['metrics_breakdown']['edge_density_pct']
        entropy_pct = results['metrics_breakdown']['entropy_pct']
        color_pct = results['metrics_breakdown']['color_variance_pct']
        
        print(f"{name:<20} {score:<8.3f} {level:<10} {edge_pct:<8.1f} {entropy_pct:<10.1f} {color_pct:<8.1f}")
    
    # Save test images (optional)
    print(f"\nSaving test images to data/raw/...")
    cv2.imwrite('data/raw/test_solid_color.png', solid_image)
    cv2.imwrite('data/raw/test_geometric.png', geometric_image) 
    cv2.imwrite('data/raw/test_noise.png', noise_image)
    print("Test images saved!")

if __name__ == "__main__":
    test_complexity_progression()
