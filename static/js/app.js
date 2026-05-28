// static/js/app.js

async function analyze() {
  const text = document.getElementById("newsInput").value.trim();
  if (!text) {
    alert("Please enter some news text to analyze.");
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  btn.textContent = "Analyzing…";
  btn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    alert("Error connecting to server. Make sure app.py is running.");
  } finally {
    btn.textContent = "Analyze News";
    btn.disabled = false;
  }
}

function renderResults(data) {
  const section = document.getElementById("results");
  const cards   = document.getElementById("modelCards");
  const verdict = document.getElementById("verdict");

  section.style.display = "block";
  cards.innerHTML = "";

  const modelMeta = {
    rnn:  { name: "RNN",  extra: "" },
    cnn:  { name: "CNN",  extra: "" },
    gnn:  { name: "GNN",  extra: '<span class="star"> ★ BEST</span>' },
  };

  let gnnLabel = null;

  for (const [key, meta] of Object.entries(modelMeta)) {
    if (!data[key]) continue;
    const m = data[key];
    if (key === "gnn") gnnLabel = m.label;

    const card = document.createElement("div");
    card.className = "model-card" + (key === "gnn" ? " best" : "");

    const labelClass = m.label === "FAKE" ? "fake" : "real";
    const icon = m.label === "FAKE" ? "🚨" : "✅";

    card.innerHTML = `
      <div class="model-name">${meta.name}${meta.extra}</div>
      <div class="verdict-label ${labelClass}">${icon} ${m.label}</div>
      <div class="prob-bar-wrap">
        <div class="prob-label"><span>Fake</span><span>${m.fake_prob}%</span></div>
        <div class="prob-bar"><div class="prob-fill fake-fill" style="width:${m.fake_prob}%"></div></div>
      </div>
      <div class="prob-bar-wrap">
        <div class="prob-label"><span>Real</span><span>${m.real_prob}%</span></div>
        <div class="prob-bar"><div class="prob-fill real-fill" style="width:${m.real_prob}%"></div></div>
      </div>
      <div style="font-size:.75rem;color:#64748b;margin-top:10px">Model accuracy: ${m.accuracy}%</div>
    `;
    cards.appendChild(card);
  }

  // Final verdict based on GNN (best model)
  if (gnnLabel) {
    const isFake = gnnLabel === "FAKE";
    verdict.className = "verdict " + (isFake ? "fake-verdict" : "real-verdict");
    verdict.innerHTML = isFake
      ? `<span class="verdict-icon">🚨</span><strong>VERDICT: FAKE NEWS</strong><br>
         <span style="color:#94a3b8;font-size:.9rem;font-weight:400">
           The GNN model (94.8% accuracy) classifies this as fake news.
         </span>`
      : `<span class="verdict-icon">✅</span><strong>VERDICT: REAL NEWS</strong><br>
         <span style="color:#94a3b8;font-size:.9rem;font-weight:400">
           The GNN model (94.8% accuracy) classifies this as real news.
         </span>`;
  }

  if (data.demo) {
    verdict.innerHTML += `<br><em style="color:#f59e0b;font-size:.8rem">
      ⚠ Demo mode: train models first → python train.py</em>`;
  }

  section.scrollIntoView({ behavior: "smooth" });
}

// Allow Enter key shortcut (Ctrl+Enter)
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") analyze();
});
