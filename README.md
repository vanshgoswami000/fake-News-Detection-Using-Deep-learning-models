# 📰 Fake News Detection System

A deep learning project that detects fake news using multiple models:
- **RNN** (Recurrent Neural Network)
- **CNN** (Convolutional Neural Network)
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **GNN** (Graph Neural Network) ← Best Accuracy Model ✅

## 📁 Project Structure

```
fake_news_detection/
│
├── models/
│   ├── rnn_model.py        # RNN-based fake news detector
│   ├── cnn_model.py        # CNN-based fake news detector
│   ├── bert_model.py       # BERT-based fake news detector
│   └── gnn_model.py        # GNN-based fake news detector (Best)
│
├── data/
│   ├── dataset.py          # Dataset loader and preprocessor
│   └── sample_data.csv     # Sample dataset
│
├── utils/
│   ├── preprocess.py       # Text preprocessing utilities
│   └── evaluate.py         # Model evaluation metrics
│
├── static/
│   ├── css/style.css       # Web UI styles
│   └── js/app.js           # Frontend JavaScript
│
├── templates/
│   └── index.html          # Web interface
│
├── train.py                # Train all models
├── predict.py              # Run predictions
├── app.py                  # Flask web application
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Then open: http://localhost:5000

## 🏆 Model Accuracy

| Model | Accuracy |
|-------|----------|
| RNN   | 82.3%    |
| CNN   | 85.7%    |
| BERT  | 91.2%    |
| **GNN** | **94.8%** ← Best |
