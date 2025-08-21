# Semantic Compression Pipeline

A research project exploring the relationships between image complexity, VLM-based text compression, text complexity, and image reconstruction quality.

## Overview

This project investigates semantic compression by:
1. Analyzing original image complexity
2. Compressing images to text descriptions using VLMs
3. Analyzing text complexity 
4. Reconstructing images from text using diffusion models
5. Evaluating reconstruction quality and compression relationships

## Project Structure

```
semantic_compression_pipeline/
├── src/                    # Source code modules
├── data/                   # Data storage
├── configs/                # Configuration files
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt        # Python dependencies
└── main.py                # Main entry point
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic_compression_pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Setup project directories:
```bash
python main.py --setup
```

## Quick Start

1. Test the installation:
```bash
python main.py --test
```

2. Test with a sample image:
```bash
python main.py --image path/to/your/image.jpg
```

## Research Questions

- How does image complexity affect VLM description quality?
- What's the relationship between prompt detail and reconstruction quality?
- How does text complexity correlate with reconstruction fidelity?
- Which VLM + diffusion model combinations provide optimal compression-quality tradeoffs?

## Development Status

- [x] Project setup and basic infrastructure
- [x] Image loading and preprocessing
- [x] Image complexity analysis
- [x] VLM integration for text compression
- [ ] Text complexity analysis
- [ ] Image reconstruction pipeline
- [ ] Similarity evaluation metrics
- [ ] Results analysis and visualization

## Contributing

This is a research project. Please feel free to contribute or suggest improvements.

## License

[Add your license information here]
