# my_ai_project
My AI learning projects for internship
# Housing Price Prediction: Dropout A/B Test

This project compares two neural network models for Boston housing price prediction:
- **Model A**: No dropout
- **Model B**: With dropout layers

Both models use:
- Handcrafted features (`RM * LSTAT`, `1/DIS`, etc.)
- Huber loss for robust regression
- StandardScaler for input/output normalization
- Early stopping to prevent overfitting

## 📦 Requirements

- Python ≥ 3.8
- See `requirements.txt`

## 🚀 Quick Start

1. Place `housing.csv` in the project root (format: space-separated, no header)
2. Install dependencies:
   ```bash
