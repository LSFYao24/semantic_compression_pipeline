import re
import nltk
from typing import Dict
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextComplexityAnalyzer:
    """
    Text complexity analyzer for VLM-generated descriptions
    Provides individual metrics and overall complexity classification
    """
    
    def __init__(self):
        self.complexity_thresholds = {
            'low': (0.0, 0.33),
            'medium': (0.33, 0.66),
            'high': (0.66, 1.0)
        }
    
    def calculate_token_count(self, text: str) -> int:
        """Calculate total number of tokens"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return len(tokens)
    
    def calculate_sentence_count(self, text: str) -> int:
        """Calculate number of sentences"""
        try:
            sentences = nltk.sent_tokenize(text)
            return max(1, len([s for s in sentences if len(s.strip()) > 2]))
        except:
            # Fallback method
            return max(1, text.count('.') + text.count('!') + text.count('?'))
    
    def calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average words per sentence"""
        word_count = len(text.split())
        sentence_count = self.calculate_sentence_count(text)
        return word_count / sentence_count if sentence_count > 0 else 0.0
    
    def calculate_type_token_ratio(self, text: str) -> float:
        """Calculate lexical diversity (unique words / total words)"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)
    
    def calculate_overall_complexity_score(self, text: str) -> float:
        """
        Calculate overall complexity score (0-1) based on multiple metrics
        Used for classification into low/medium/high
        """
        token_count = self.calculate_token_count(text)
        sentence_count = self.calculate_sentence_count(text)
        avg_sentence_length = self.calculate_avg_sentence_length(text)
        type_token_ratio = self.calculate_type_token_ratio(text)
        
        # Normalize individual metrics to 0-1 scale
        # These thresholds can be adjusted based on your data
        normalized_token = min(token_count / 50.0, 1.0)  # Normalize by expected max ~50 tokens
        normalized_sentence = min(sentence_count / 5.0, 1.0)  # Normalize by expected max ~5 sentences
        normalized_avg_length = min(avg_sentence_length / 15.0, 1.0)  # Normalize by expected max ~15 words/sentence
        normalized_ttr = type_token_ratio  # Already 0-1
        
        # Simple average of normalized metrics
        overall_score = (normalized_token + normalized_sentence + 
                        normalized_avg_length + normalized_ttr) / 4.0
        
        return overall_score
    
    def classify_complexity(self, score: float) -> str:
        """Classify text complexity level based on overall score"""
        for level, (min_val, max_val) in self.complexity_thresholds.items():
            if min_val <= score < max_val:
                return level
        
        # Handle edge case where score is exactly 1.0
        if score >= self.complexity_thresholds['high'][1]:
            return 'high'
        
        return 'low'  # Default fallback
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text complexity - returns metrics and overall classification
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with complexity metrics and classification
        """
        # Calculate individual metrics
        token_count = self.calculate_token_count(text)
        sentence_count = self.calculate_sentence_count(text)
        avg_sentence_length = self.calculate_avg_sentence_length(text)
        type_token_ratio = self.calculate_type_token_ratio(text)
        
        # Calculate overall score and classification
        overall_score = self.calculate_overall_complexity_score(text)
        complexity_level = self.classify_complexity(overall_score)
        
        return {
            # Core metrics from your table
            'token_count': token_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'type_token_ratio': round(type_token_ratio, 3),
            
            # Additional basic info
            'char_count': len(text),
            'word_count': len(text.split()),
            
            # Overall assessment
            'overall_score': round(overall_score, 3),
            'complexity_level': complexity_level
        }
    
    def batch_analyze(self, texts: list) -> list:
        """Analyze multiple texts and return results"""
        results = []
        for i, text in enumerate(texts):
            result = self.analyze_text(text)
            result['text_index'] = i
            result['original_text'] = text
            results.append(result)
        return results
    
    def print_analysis(self, text: str):
        """Print formatted analysis results for single text"""
        result = self.analyze_text(text)
        
        print(f"\n--- Text Complexity Analysis ---")
        print(f"Text: \"{text}\"")
        print(f"Complexity Level: {result['complexity_level'].upper()}")
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Metrics:")
        print(f"  Token Count: {result['token_count']}")
        print(f"  Sentence Count: {result['sentence_count']}")
        print(f"  Avg Sentence Length: {result['avg_sentence_length']} words")
        print(f"  Type-Token Ratio: {result['type_token_ratio']:.3f}")


# Simple usage and testing
if __name__ == "__main__":
    analyzer = TextComplexityAnalyzer()
    
    # Test with VLM-like outputs of different complexities
    test_texts = [
        "A red circle.",  # Should be LOW
        "The image shows a red circle and green square on white background.",  # Should be MEDIUM  
        "This detailed image contains multiple geometric shapes with vibrant colors, complex spatial relationships, and varied textures distributed across the frame."  # Should be HIGH
    ]
    
    print("Testing Text Complexity Analyzer...")
    
    for text in test_texts:
        analyzer.print_analysis(text)
    
    print("\n✅ Text Complexity Analyzer ready for use!")
