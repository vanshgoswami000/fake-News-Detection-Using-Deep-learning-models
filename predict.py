"""
predict.py — Run predictions using all four models
             GNN gives best accuracy (~94.8%)

Usage:
    python predict.py --text "Breaking: Scientists discover miracle cure"
    python predict.py --model gnn --text "Senate passes budget bill"
    python predict.py --interactive
"""

import os
import argparse
import torch
import torch.nn.functional as F

from data.dataset import load_data
from utils.preprocess import build_vocab, text_to_indices, build_graph_features
from models.rnn_model import build_rnn
from models.cnn_model import build_cnn
from models.gnn_model import build_gnn, build_adj, gnn_predict_single

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "saved_models"

LABELS = {0: "✅ REAL", 1: "🚨 FAKE"}


def load_model(name, vocab_size):
    """Load a trained model from disk."""
    path = f"{SAVE_DIR}/{name}_model.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model at '{path}'. Run: python train.py --model {name}"
        )
    if name == "rnn":
        model = build_rnn(vocab_size)
    elif name == "cnn":
        model = build_cnn(vocab_size)
    else:  # gnn
        model = build_gnn(vocab_size)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_rnn_cnn(model, text, vocab) -> dict:
    """Predict with RNN or CNN model."""
    x = torch.tensor([text_to_indices(text, vocab)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
    return {
        "label": "FAKE" if pred == 1 else "REAL",
        "fake_prob": round(float(probs[1]) * 100, 2),
        "real_prob": round(float(probs[0]) * 100, 2),
        "confidence": round(float(probs[pred]) * 100, 2),
    }


def predict_gnn(model, text, vocab) -> dict:
    """Predict with GNN model (BEST)."""
    graph = build_graph_features(text, vocab)
    return gnn_predict_single(model, graph, vocab, DEVICE)


def predict_all(text, vocab):
    """Run prediction with all available trained models."""
    print(f"\n{'='*55}")
    print(f"  News: \"{text[:70]}{'...' if len(text)>70 else ''}\"")
    print(f"{'='*55}")

    results = {}
    for name in ["rnn", "cnn", "gnn"]:
        try:
            model = load_model(name, len(vocab))
            if name == "gnn":
                result = predict_gnn(model, text, vocab)
            else:
                result = predict_rnn_cnn(model, text, vocab)
            results[name] = result
            star = "  ← BEST MODEL" if name == "gnn" else ""
            label_icon = "🚨 FAKE" if result["label"] == "FAKE" else "✅ REAL"
            print(
                f"  {name.upper():<6}: {label_icon:<12} "
                f"(confidence: {result['confidence']:.1f}%){star}"
            )
        except FileNotFoundError as e:
            print(f"  {name.upper():<6}: {e}")

    print(f"{'='*55}\n")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("--text", type=str, help="News text to classify")
    parser.add_argument("--model", choices=["rnn", "cnn", "gnn", "all"],
                        default="all")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive prediction mode")
    args = parser.parse_args()

    # Build vocab from training data
    texts, _ = load_data()
    vocab = build_vocab(texts)

    if args.interactive:
        print("\n🔍 Fake News Detector — Interactive Mode")
        print("  Type a news headline and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("  Enter news text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                predict_all(text, vocab)
        return

    if args.text:
        if args.model == "all":
            predict_all(args.text, vocab)
        else:
            model = load_model(args.model, len(vocab))
            if args.model == "gnn":
                result = predict_gnn(model, args.text, vocab)
            else:
                result = predict_rnn_cnn(model, args.text, vocab)
            label_icon = "🚨 FAKE" if result["label"] == "FAKE" else "✅ REAL"
            print(f"\n  Result: {label_icon}")
            print(f"  Fake Probability : {result['fake_prob']:.1f}%")
            print(f"  Real Probability : {result['real_prob']:.1f}%")
            print(f"  Confidence       : {result['confidence']:.1f}%\n")
    else:
        print("Usage: python predict.py --text 'Your news headline here'")
        print("       python predict.py --interactive")


if __name__ == "__main__":
    main()
