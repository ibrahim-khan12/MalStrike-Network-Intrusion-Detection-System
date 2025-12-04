let model = null;
let csvData = null;          // { headers: [...], rows: [...], featureIndices: [...], labelIndex: number }

/* ---------- Model loading ---------- */

async function loadModel() {
  const status = document.getElementById("status");
  try {
    const res = await fetch("melm_model.json");
    if (!res.ok) throw new Error("HTTP " + res.status);
    model = await res.json();

    document.getElementById("predictBtn").disabled = false;
    status.textContent = `Model loaded. Expecting ${model.input_dim} features.`;
  } catch (err) {
    console.error(err);
    status.textContent = "Failed to load model. Check melm_model.json path.";
  }
}

/* ---------- Utilities ---------- */

function parseFeatures(text) {
  const tokens = text
    .split(/[\s,;]+/)
    .map(t => t.trim())
    .filter(Boolean);

  if (!tokens.length) {
    throw new Error("No feature values found.");
  }

  const vals = tokens.map(Number);
  if (vals.some(v => Number.isNaN(v))) {
    throw new Error("Some feature values are not valid numbers.");
  }
  return vals;
}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function predict(features) {
  if (!model) {
    throw new Error("Model not loaded.");
  }
  if (features.length !== model.input_dim) {
    throw new Error(`Expected ${model.input_dim} features, got ${features.length}.`);
  }

  // Standardize
  const x = features.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i]);

  // Hidden layer (morphological dilation)
  const H = new Array(model.hidden_dim);
  for (let j = 0; j < model.hidden_dim; j++) {
    const wj = model.W[j];
    let maxVal = -Infinity;
    for (let i = 0; i < model.input_dim; i++) {
      const val = x[i] + wj[i];
      if (val > maxVal) maxVal = val;
    }
    H[j] = maxVal + model.b[j];
  }

  // Output layer: scores = H @ beta
  const scores = new Array(model.class_names.length).fill(0);
  for (let c = 0; c < model.class_names.length; c++) {
    let sum = 0;
    for (let j = 0; j < model.hidden_dim; j++) {
      sum += H[j] * model.beta[j][c];
    }
    scores[c] = sum;
  }

  const probs = softmax(scores);
  const best = probs.indexOf(Math.max(...probs));
  const label = model.class_names[best]; // numeric (0 or 1)

  return { label, probs };
}

/* ---------- UI: manual predict ---------- */

document.getElementById("predictBtn").addEventListener("click", () => {
  const input = document.getElementById("featureInput").value.trim();
  try {
    const feats = parseFeatures(input);
    const result = predict(feats);
    showResult(result);
    document.getElementById("status").textContent = "Prediction complete.";
  } catch (err) {
    alert(err.message);
  }
});

function showResult(result) {
  const card = document.getElementById("resultCard");
  const labelEl = document.getElementById("predictionLabel");
  const barsContainer = document.getElementById("probBars");

  const isMalware = Number(result.label) === 1;

  labelEl.textContent = isMalware ? "⚠️ Classified as MALWARE" : "✅ Classified as BENIGN";
  labelEl.style.color = isMalware ? "var(--malware)" : "var(--benign)";

  barsContainer.innerHTML = "";

  result.probs.forEach((p, idx) => {
    const cls = model.class_names[idx];
    const isM = Number(cls) === 1;

    const row = document.createElement("div");
    row.className = "probability-bar-row";

    const label = document.createElement("div");
    label.className = "probability-bar-label";
    label.textContent = `Class ${cls} (${isM ? "Malware" : "Benign"})`;

    const barOuter = document.createElement("div");
    barOuter.className = "probability-bar";

    const barInner = document.createElement("div");
    barInner.className = "prob-fill";
    barInner.style.background = isM ? "var(--malware)" : "var(--benign)";
    requestAnimationFrame(() => {
      barInner.style.width = (p * 100).toFixed(1) + "%";
    });

    barOuter.appendChild(barInner);

    const text = document.createElement("div");
    text.className = "prob-text";
    text.textContent = `${(p * 100).toFixed(2)} %`;

    row.appendChild(label);
    row.appendChild(barOuter);
    row.appendChild(text);

    barsContainer.appendChild(row);
  });

  card.style.display = "block";
}

/* ---------- CSV upload + auto-columns ---------- */

document.getElementById("csvFile").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result;
      csvData = parseCSV(text);
      populateRowPicker();
    } catch (err) {
      alert("Failed to parse CSV: " + err.message);
    }
  };
  reader.readAsText(file);
});

function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (!lines.length) throw new Error("CSV appears to be empty.");

  const headerLine = lines[0];
  const headers = headerLine.split(",").map(h => h.trim());

  const labelName = document.getElementById("labelCol").value.trim() || "Label";
  const labelIndex = headers.findIndex(h => h.toLowerCase() === labelName.toLowerCase());
  if (labelIndex === -1) {
    throw new Error(`Label column '${labelName}' not found in header.`);
  }

  const featureIndices = headers
    .map((h, idx) => ({ h, idx }))
    .filter(obj => obj.idx !== labelIndex)
    .map(obj => obj.idx);

  const rows = lines.slice(1).map(line => line.split(",").map(v => v.trim()));

  const info = document.getElementById("csvInfo");
  info.textContent = `Loaded CSV with ${rows.length} rows and ${headers.length} columns. `
    + `Using '${headers[labelIndex]}' as label, and ${featureIndices.length} feature columns.`;

  return { headers, rows, featureIndices, labelIndex };
}

function populateRowPicker() {
  if (!csvData) return;

  const rowPicker = document.getElementById("rowPicker");
  const select = document.getElementById("rowSelect");
  select.innerHTML = "";

  const maxOptions = Math.min(csvData.rows.length, 1000);
  for (let i = 0; i < maxOptions; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `Row ${i}`;
    select.appendChild(opt);
  }

  rowPicker.style.display = "flex";
}

document.getElementById("loadRowBtn").addEventListener("click", () => {
  if (!csvData) {
    alert("No CSV loaded.");
    return;
  }
  const idx = Number(document.getElementById("rowSelect").value);
  const row = csvData.rows[idx];
  const featureValues = csvData.featureIndices.map(i => row[i]);

  document.getElementById("featureInput").value = featureValues.join(", ");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ---------- Theme toggle ---------- */

document.getElementById("themeToggle").addEventListener("change", (e) => {
  document.body.className = e.target.checked ? "light" : "dark";
});

/* ---------- Init ---------- */
window.addEventListener("load", loadModel);
