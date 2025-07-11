# VGG16 Partitioning Demo

This project demonstrates how to partition a VGG16 neural network for distributed inference between client and server.

## Files

- `vgg_partition_demo.py` - Main demo showing VGG16 model partitioning
- `test_environment.py` - Environment setup verification script

## Features

- Split VGG16 model at different layers
- Measure client vs server processing times
- Calculate data transmission requirements
- Compare different partitioning strategies

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install tensorflow numpy
```

## Usage

1. Test your environment:
```bash
python test_environment.py
```

2. Run the VGG16 partitioning demo:
```bash
python vgg_partition_demo.py
```

## Output

The demo will show:
- Model loading and partitioning
- Processing times for client and server portions
- Data transmission requirements
- Performance comparison across different cut points
