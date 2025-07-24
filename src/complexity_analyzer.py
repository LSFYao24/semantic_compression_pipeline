import cv2
import numpy as np
from typing import Dict, Tuple
import math

class ComplexityAnalyzer:
    """
    Image complexity analyzer implementing three key metrics:
    1. Edge Density - proportion of edge pixels
    2. Entropy - information content measure
    3. Color Variance - degree of color variation
    """
    
    def __init__(self):
        self.complexity_thresholds = {
            'low': (0.0, 0.33),
            'medium': (0.33, 0.66), 
            'high': (0.66, 1.0)
        }
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Edge density ratio (0.0 to 1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / total_pixels
        
        return edge_density
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate image entropy measuring information content
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized entropy value (0.0 to 1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Normalize histogram to get probabilities
        hist = hist / hist.sum()
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize entropy (max entropy for 8-bit image is log2(256) = 8)
        normalized_entropy = entropy / 8.0
        
        return normalized_entropy
    
    def calculate_color_variance(self, image: np.ndarray) -> float:
        """
        Calculate color variance across the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized color variance (0.0 to 1.0)
        """
        if len(image.shape) == 2:
            # Grayscale image
            variance = np.var(image)
            # Normalize by maximum possible variance for 8-bit image
            max_variance = (255.0 ** 2) / 4  # Maximum variance occurs when half pixels are 0, half are 255
            normalized_variance = min(variance / max_variance, 1.0)
        else:
            # Color image - calculate variance for each channel
            variances = []
            for channel in range(image.shape[2]):
                channel_var = np.var(image[:, :, channel])
                variances.append(channel_var)
            
            # Use mean variance across channels
            mean_variance = np.mean(variances)
            max_variance = (255.0 ** 2) / 4
            normalized_variance = min(mean_variance / max_variance, 1.0)
        
        return normalized_variance
    
    def calculate_complexity_score(self, image: np.ndarray) -> Dict:
        """
        Calculate overall complexity score and individual metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing all metrics and overall score
        """
        # Calculate individual metrics
        edge_density = self.calculate_edge_density(image)
        entropy = self.calculate_entropy(image)
        color_variance = self.calculate_color_variance(image)
        
        # Calculate weighted sum (equal weights as per formula)
        overall_score = (entropy + edge_density + color_variance) / 3.0
        
        # Determine complexity level
        complexity_level = self.classify_complexity(overall_score)
        
        results = {
            'edge_density': edge_density,
            'entropy': entropy,
            'color_variance': color_variance,
            'overall_score': overall_score,
            'complexity_level': complexity_level,
            'metrics_breakdown': {
                'edge_density_pct': edge_density * 100,
                'entropy_pct': entropy * 100,
                'color_variance_pct': color_variance * 100,
                'overall_score_pct': overall_score * 100
            }
        }
        
        return results
    
    def classify_complexity(self, score: float) -> str:
        """
        Classify complexity level based on score
        
        Args:
            score: Overall complexity score (0.0 to 1.0)
            
        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        for level, (min_val, max_val) in self.complexity_thresholds.items():
            if min_val <= score < max_val:
                return level
        
        # Handle edge case where score is exactly 1.0
        if score >= self.complexity_thresholds['high'][1]:
            return 'high'
        
        return 'low'  # Default fallback
    
    def analyze_image(self, image: np.ndarray, verbose: bool = True) -> Dict:
        """
        Complete analysis of image complexity with optional detailed output
        
        Args:
            image: Input image as numpy array
            verbose: Whether to print detailed results
            
        Returns:
            Complete analysis results
        """
        results = self.calculate_complexity_score(image)
        
        if verbose:
            self.print_analysis_results(results)
        
        return results
    
    def print_analysis_results(self, results: Dict):
        """Print formatted analysis results"""
        print("\n" + "="*50)
        print("IMAGE COMPLEXITY ANALYSIS RESULTS")
        print("="*50)
        
        print(f"Overall Complexity Score: {results['overall_score']:.3f}")
        print(f"Complexity Level: {results['complexity_level'].upper()}")
        
        print("\nDetailed Metrics:")
        print(f"  Edge Density:    {results['edge_density']:.3f} ({results['metrics_breakdown']['edge_density_pct']:.1f}%)")
        print(f"  Entropy:         {results['entropy']:.3f} ({results['metrics_breakdown']['entropy_pct']:.1f}%)")
        print(f"  Color Variance:  {results['color_variance']:.3f} ({results['metrics_breakdown']['color_variance_pct']:.1f}%)")
        
        print(f"\nThresholds:")
        print(f"  Low:    0.000 - 0.330")
        print(f"  Medium: 0.330 - 0.660") 
        print(f"  High:   0.660 - 1.000")
        
        print("="*50)

# Usage example and testing
if __name__ == "__main__":
    # Example usage
    analyzer = ComplexityAnalyzer()
    
    print("Complexity Analyzer initialized successfully!")
    print("Supported analysis metrics:")
    print("  - Edge Density (structural complexity)")
    print("  - Entropy (information content)")
    print("  - Color Variance (color diversity)")
    
    # Test with a simple synthetic image
    print("\nCreating test image...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    results = analyzer.analyze_image(test_image)
    
    print(f"\nTest completed! Complexity level: {results['complexity_level']}")
