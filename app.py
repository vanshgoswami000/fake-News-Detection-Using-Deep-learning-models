"""
app.py — Flask web application for Fake News Detection
         Visit: http://localhost:5000
"""

import os
import json
from flask import Flask, render_template, request, jsonify
import torch

from data.dataset import load_data
from utils.preprocess import build_vocab, text_to_indices, build_graph_features
from models.rnn_model import build_rnn
from models.cnn_model import build_cnn
from models.gnn_model import build_gnn, build_adj, gnn_predict_single
import torch.nn.functional as F

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "saved_models"

# ── Load vocab and models at startup ─────────────────────────────────────────

texts, _ = load_data()
VOCAB = build_vocab(texts)
VOCAB_SIZE = len(VOCAB)

MODELS = {}

def _try_load(name, builder):
    path = f"{SAVE_DIR}/{name}_model.pt"
    if os.path.exists(path):
        m = builder(VOCAB_SIZE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.to(DEVICE).eval()
        print(f"  ✓ Loaded {name.upper()} from {path}")
        return m
    print(f"  ✗ {name.upper()} not found — run: python train.py")
    return None

MODELS["rnn"]  = _try_load("rnn", build_rnn)
#MODELS["cnn"]  = _try_load("cnn", build_cnn)
#MODELS["gnn"]  = _try_load("gnn", build_gnn)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = {}

    # RNN
    if MODELS["rnn"]:
        x = torch.tensor([text_to_indices(text, VOCAB)],
                         dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(MODELS["rnn"](x), dim=1)[0]
        pred = int(probs.argmax())
        results["rnn"] = {
            "label": "FAKE" if pred == 1 else "REAL",
            "fake_prob": round(float(probs[1]) * 100, 1),
            "real_prob": round(float(probs[0]) * 100, 1),
            "accuracy": 82.3,
        }

    # CNN
   # if MODELS["cnn"]:
    #    x = torch.tensor([text_to_indices(text, VOCAB)],
     #                    dtype=torch.long).to(DEVICE)
      #  with torch.no_grad():
       #     probs = F.softmax(MODELS["cnn"](x), dim=1)[0]
        #pred = int(probs.argmax())
        #results["cnn"] = {
         #   "label": "FAKE" if pred == 1 else "REAL",
          #  "fake_prob": round(float(probs[1]) * 100, 1),
           # "real_prob": round(float(probs[0]) * 100, 1),
            #"accuracy": 85.7,
        #}

    # GNN (BEST)
    #if MODELS["gnn"]:
     #   graph = build_graph_features(text, VOCAB)
      #  res = gnn_predict_single(MODELS["gnn"], graph, VOCAB, DEVICE)
       # results["gnn"] = {**res, "accuracy": 94.8}

    if not results:
        # Demo mode — return simulated predictions
        results = {
            "rnn":  {"label": "FAKE", "fake_prob": 78.2, "real_prob": 21.8, "accuracy": 82.3},
            "cnn":  {"label": "FAKE", "fake_prob": 82.5, "real_prob": 17.5, "accuracy": 85.7},
            "gnn":  {"label": "FAKE", "fake_prob": 91.3, "real_prob": 8.7,  "accuracy": 94.8},
            "demo": True,
        }

    return jsonify(results)


@app.route("/benchmarks")
def benchmarks():
    return jsonify({
        "rnn":  {"accuracy": 82.3, "f1": 82.2},
        "cnn":  {"accuracy": 85.7, "f1": 85.5},
        "bert": {"accuracy": 91.2, "f1": 91.2},
        "gnn":  {"accuracy": 94.8, "f1": 94.7, "best": True},
    })


if __name__ == "__main__":
    print("\n🔍 Fake News Detection Server")
    print(f"   Device : {DEVICE}")
    print(f"   Vocab  : {VOCAB_SIZE} tokens")
    print("   URL    : http://localhost:5000\n")
    app.run(debug=True, port=5000)
