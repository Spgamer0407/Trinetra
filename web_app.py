import io
import json
import mimetypes
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

import config
from forensic_fingerprint import verify_integrity

HOST = "127.0.0.1"
PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TRINETRA - AI Digital Authenticity & Forensic Traceability</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 28, 45, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-amber: #f59e0b;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 15% 20%, #151c33 0%, var(--bg) 60%);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      color: white;
      font-size: 18px;
      box-shadow: 0 0 16px rgba(59, 130, 246, 0.4);
    }
    .brand-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.5px;
      background: linear-gradient(90deg, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .badge-vtu {
      font-size: 11px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 20px;
      color: #94a3b8;
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px;
      width: 100%;
      flex: 1;
    }
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 10px 18px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .tab-btn:hover { color: var(--text-main); }
    .tab-btn.active {
      color: var(--accent-cyan);
      border-bottom: 2px solid var(--accent-cyan);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; animation: fadeIn 0.3s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 860px) { .grid-2 { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 24px;
      backdrop-filter: blur(8px);
    }
    .card-title {
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #e2e8f0;
    }
    .dropzone {
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 10px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      background: rgba(0, 0, 0, 0.2);
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.05);
    }
    .dropzone-icon {
      font-size: 36px;
      margin-bottom: 8px;
      color: var(--text-muted);
    }
    .dropzone-label {
      font-size: 14px;
      color: var(--text-muted);
    }
    .dropzone-browse {
      color: var(--accent-cyan);
      font-weight: 600;
      text-decoration: underline;
      cursor: pointer;
    }
    input[type="file"] { display: none; }

    .form-group {
      margin-top: 16px;
    }
    .form-label {
      font-size: 13px;
      color: var(--text-muted);
      display: block;
      margin-bottom: 6px;
    }
    .range-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--accent-cyan);
    }
    .range-val {
      font-family: var(--font-mono);
      font-size: 13px;
      width: 44px;
    }
    .checkbox-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-main);
      cursor: pointer;
      user-select: none;
      margin-top: 12px;
    }
    .btn-primary {
      width: 100%;
      margin-top: 20px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px 18px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, box-shadow 0.2s;
      box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3);
    }
    .btn-primary:hover { opacity: 0.92; }
    .btn-primary:disabled {
      opacity: 0.4;
      cursor: not-allowed;
      box-shadow: none;
    }

    .verdict-box {
      border-radius: 10px;
      padding: 20px;
      text-align: center;
      margin-bottom: 20px;
      display: none;
    }
    .verdict-box.real {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.4);
      display: block;
    }
    .verdict-box.fake {
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.4);
      display: block;
    }
    .verdict-box.unclassified {
      background: rgba(156, 163, 175, 0.12);
      border: 1px solid rgba(156, 163, 175, 0.4);
      display: block;
    }
    .verdict-badge {
      display: inline-block;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .verdict-box.real .verdict-badge { color: var(--accent-green); }
    .verdict-box.fake .verdict-badge { color: var(--accent-red); }
    .verdict-box.unclassified .verdict-badge { color: var(--text-muted); }
    .verdict-sub {
      font-size: 13px;
      color: var(--text-muted);
    }

    .metrics-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    .metric-card {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .metric-num {
      font-size: 18px;
      font-weight: 700;
      color: #f1f5f9;
      font-family: var(--font-mono);
    }
    .metric-lbl {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-top: 2px;
    }

    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }
    .gallery-item {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
    }
    .gallery-item img {
      width: 100%;
      height: 160px;
      object-fit: contain;
      border-radius: 6px;
      background: #000;
    }
    .gallery-label {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 6px;
      font-weight: 500;
    }

    .hash-group {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px;
      margin-top: 12px;
      font-family: var(--font-mono);
      font-size: 12px;
      word-break: break-all;
    }
    .hash-label {
      font-size: 11px;
      color: var(--accent-cyan);
      font-family: var(--font-sans);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .spinner {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }
    th {
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
    }
    td code {
      font-family: var(--font-mono);
      font-size: 11px;
      background: rgba(255, 255, 255, 0.05);
      padding: 2px 6px;
      border-radius: 4px;
    }
    .status-tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
    }
    .status-tag.real { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
    .status-tag.fake { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }

    footer {
      border-top: 1px solid var(--card-border);
      padding: 20px;
      text-align: center;
      font-size: 12px;
      color: var(--text-muted);
      background: rgba(11, 15, 25, 0.95);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">T</div>
      <div>
        <div class="brand-title">TRINETRA</div>
        <div class="brand-sub">Digital Authenticity & Forensic Traceability</div>
      </div>
    </div>
    <div class="badge-vtu">Sahyadri College of Eng. & Mgmt (VTU) | 2025-26</div>
  </header>

  <main>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('detect')">🔍 Fake Image Detection</button>
      <button class="tab-btn" onclick="switchTab('verify')">🛡️ Forensic Integrity Verification</button>
      <button class="tab-btn" onclick="switchTab('ledger')">📜 Forensic Ledger</button>
    </div>

    <!-- TAB 1: DETECTION -->
    <div id="tab-detect" class="tab-panel active">
      <div class="grid-2">
        <!-- Input Column -->
        <div class="card">
          <div class="card-title">Upload Target Image</div>
          <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
            <div class="dropzone-icon">🖼️</div>
            <div class="dropzone-label">Drag & drop image here, or <span class="dropzone-browse">browse</span></div>
            <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Supported formats: JPG, PNG, WEBP (Max 20MB)</div>
          </div>
          <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(this.files[0])">
          <div id="filePreviewName" style="font-size: 12px; color: var(--accent-cyan); margin-top: 8px; font-family: var(--font-mono);"></div>

          <div class="form-group">
            <label class="form-label">Classification Threshold: <span id="threshVal" class="range-val">0.50</span></label>
            <div class="range-wrap">
              <input type="range" id="threshInput" min="0.1" max="0.9" step="0.05" value="0.5" oninput="document.getElementById('threshVal').innerText = parseFloat(this.value).toFixed(2)">
            </div>
          </div>

          <label class="checkbox-wrap">
            <input type="checkbox" id="explainInput" checked> Enable Grad-CAM Visual Explainability (Heatmap)
          </label>
          <label class="checkbox-wrap">
            <input type="checkbox" id="registerInput"> Register Verified Authentic into Forensic Ledger
          </label>

          <button class="btn-primary" id="analyzeBtn" onclick="runAnalysis()">Analyze Authenticity</button>
        </div>

        <!-- Output Column -->
        <div class="card" id="resultCard">
          <div class="card-title">Authenticity Assessment</div>
          <div id="placeholderMsg" style="text-align: center; padding: 60px 20px; color: var(--text-muted); font-size: 13px;">
            Upload an image and click <b>Analyze Authenticity</b> to inspect facial manipulation artifacts and cryptographic fingerprints.
          </div>

          <div id="analysisContent" style="display: none;">
            <div id="verdictBox" class="verdict-box">
              <div class="verdict-badge" id="verdictBadge">REAL / AUTHENTIC</div>
              <div class="verdict-sub" id="verdictSub">Analyzed using DenseNet-201 and YuNet Facial Alignment</div>
            </div>

            <div class="metrics-row" style="grid-template-columns: repeat(4, 1fr);">
              <div class="metric-card" style="border-top: 3px solid var(--accent-green);">
                <div class="metric-num" id="metricRealProb" style="color: var(--accent-green);">--%</div>
                <div class="metric-lbl">Real Probability</div>
              </div>
              <div class="metric-card" style="border-top: 3px solid var(--accent-red);">
                <div class="metric-num" id="metricFakeProb" style="color: var(--accent-red);">--%</div>
                <div class="metric-lbl">Fake Probability</div>
              </div>
              <div class="metric-card" style="border-top: 3px solid var(--accent-cyan);">
                <div class="metric-num" id="metricConf" style="color: var(--accent-cyan);">--%</div>
                <div class="metric-lbl">Confidence</div>
              </div>
              <div class="metric-card">
                <div class="metric-num" id="metricFaces">0</div>
                <div class="metric-lbl">Faces Detected</div>
              </div>
            </div>

            <!-- Probability Distribution Split Bar -->
            <div style="margin: 14px 0 16px; background: rgba(0,0,0,0.25); border: 1px solid var(--card-border); border-radius: 8px; padding: 10px 14px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 6px; font-weight: 600;">
                <span style="color: var(--accent-green);">🟢 Authentic (Real): <span id="barRealText">50%</span></span>
                <span id="barThreshText" style="color: var(--text-muted); font-size: 11px; font-family: var(--font-mono);">Decision Threshold: 50%</span>
                <span style="color: var(--accent-red);">🔴 Manipulated (Fake): <span id="barFakeText">50%</span></span>
              </div>
              <div style="height: 10px; background: rgba(255,255,255,0.08); border-radius: 5px; overflow: hidden; display: flex;">
                <div id="barReal" style="height: 100%; background: linear-gradient(90deg, #059669, #10b981); width: 50%; transition: width 0.4s ease;"></div>
                <div id="barFake" style="height: 100%; background: linear-gradient(90deg, #f87171, #ef4444); width: 50%; transition: width 0.4s ease;"></div>
              </div>
            </div>

            <div id="explanationCard" style="margin-top: 16px; padding: 14px 18px; background: rgba(6, 182, 212, 0.08); border-left: 4px solid var(--accent-cyan); border-radius: 8px;">
              <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--accent-cyan); font-weight: 700; margin-bottom: 4px;">🔬 Forensic Localization Insight (XAI)</div>
              <div id="explanationText" style="font-size: 13.5px; line-height: 1.5; color: #e2e8f0;">--</div>
            </div>

            <div class="card-title" style="margin-top: 20px;">Visual Evidence & Explainability</div>
            <div class="gallery" id="galleryBox"></div>

            <div class="card-title" style="margin-top: 20px;">Forensic Fingerprint</div>
            <div class="hash-group">
              <div class="hash-label">Cryptographic SHA-256</div>
              <div id="hashSha256">--</div>
            </div>
            <div class="hash-group">
              <div class="hash-label">Perceptual Hash (pHash)</div>
              <div id="hashPhash">--</div>
            </div>
            <div id="ledgerBadge" style="margin-top: 12px; font-size: 12px;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: VERIFICATION -->
    <div id="tab-verify" class="tab-panel">
      <div class="card" style="max-width: 800px; margin: 0 auto;">
        <div class="card-title">Digital Integrity & Traceability Verification</div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 20px;">
          Check whether a candidate image is authentic, has been registered in the TRINETRA forensic ledger, or has been tampered with.
        </p>

        <div class="dropzone" id="verifyDropzone" onclick="document.getElementById('verifyFileInput').click()">
          <div class="dropzone-icon">🛡️</div>
          <div class="dropzone-label">Select image to verify against the ledger</div>
        </div>
        <input type="file" id="verifyFileInput" accept="image/*" onchange="runVerification(this.files[0])">

        <div id="verifyResult" style="margin-top: 24px; display: none;"></div>
      </div>
    </div>

    <!-- TAB 3: LEDGER -->
    <div id="tab-ledger" class="tab-panel">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <div class="card-title" style="margin: 0;">Immutable Forensic Integrity Ledger</div>
          <button class="tab-btn" onclick="loadLedger()" style="border: 1px solid var(--card-border); border-radius: 6px;">🔄 Refresh</button>
        </div>
        <div style="overflow-x: auto;">
          <table id="ledgerTable">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Timestamp (UTC)</th>
                <th>Verdict</th>
                <th>SHA-256 Digest</th>
                <th>pHash</th>
              </tr>
            </thead>
            <tbody id="ledgerBody">
              <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Loading ledger records...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </main>

  <footer>
    TRINETRA Mini Project &copy; 2025-26 | Department of Computer Science & Engineering | Sahyadri College of Engineering & Management
  </footer>

  <script>
    let selectedFile = null;

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + name).classList.add('active');
      if (name === 'ledger') loadLedger();
    }

    function handleFileSelect(file) {
      if (!file) return;
      selectedFile = file;
      document.getElementById('filePreviewName').innerText = 'Selected: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    }

    // Drag and drop setup
    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });

    async function runAnalysis() {
      if (!selectedFile) {
        alert('Please select an image file first!');
        return;
      }
      const btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Analyzing Image...';

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('threshold', document.getElementById('threshInput').value);
      formData.append('explain', document.getElementById('explainInput').checked ? 'true' : 'false');
      formData.append('register', document.getElementById('registerInput').checked ? 'true' : 'false');

      try {
        const resp = await fetch('/api/analyze-image', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        document.getElementById('placeholderMsg').style.display = 'none';
        document.getElementById('analysisContent').style.display = 'block';

        const fakeProb = data.fake_probability !== null ? data.fake_probability : 0.5;
        const realProb = data.real_probability !== null ? data.real_probability : (1.0 - fakeProb);
        const conf = data.confidence !== null ? data.confidence : Math.max(realProb, fakeProb);

        const vbox = document.getElementById('verdictBox');
        vbox.className = 'verdict-box ' + (data.final_label === 'REAL' ? 'real' : (data.final_label === 'FAKE' ? 'fake' : 'unclassified'));
        document.getElementById('verdictBadge').innerText = data.verdict;
        document.getElementById('verdictSub').innerText = data.final_label === 'REAL'
          ? `Verified authentic. Real: ${(realProb * 100).toFixed(1)}% vs Fake: ${(fakeProb * 100).toFixed(1)}% (below threshold).`
          : (data.final_label === 'FAKE'
            ? `Flagged as manipulated. Fake: ${(fakeProb * 100).toFixed(1)}% vs Real: ${(realProb * 100).toFixed(1)}% (exceeds threshold).`
            : 'No face detected in image.');

        document.getElementById('metricRealProb').innerText = (realProb * 100).toFixed(1) + '%';
        document.getElementById('metricFakeProb').innerText = (fakeProb * 100).toFixed(1) + '%';
        document.getElementById('metricConf').innerText = (conf * 100).toFixed(1) + '%';
        document.getElementById('metricFaces').innerText = data.faces_detected;

        // Populate Probability Split Bar
        const realPct = (realProb * 100).toFixed(1);
        const fakePct = (fakeProb * 100).toFixed(1);
        document.getElementById('barReal').style.width = realPct + '%';
        document.getElementById('barFake').style.width = fakePct + '%';
        document.getElementById('barRealText').innerText = realPct + '%';
        document.getElementById('barFakeText').innerText = fakePct + '%';
        const threshPct = (parseFloat(document.getElementById('threshInput').value) * 100).toFixed(0);
        document.getElementById('barThreshText').innerText = 'Decision Threshold: ' + threshPct + '%';

        // Populate Forensic Localization Insight
        document.getElementById('explanationText').innerText = data.forensic_explanation || 'No localization anomalies recorded.';

        // Populate Gallery
        const gallery = document.getElementById('galleryBox');
        gallery.innerHTML = '';
        if (data.annotated_image_url) {
          gallery.innerHTML += `<div class="gallery-item"><img src="${data.annotated_image_url}"> <div class="gallery-label">Detection Overview</div></div>`;
        }
        if (data.faces && data.faces.length > 0) {
          data.faces.forEach(f => {
            if (f.crop_url) gallery.innerHTML += `<div class="gallery-item"><img src="${f.crop_url}"> <div class="gallery-label">Face #${f.rank} Crop (${f.label})</div></div>`;
            if (f.gradcam_url) gallery.innerHTML += `<div class="gallery-item"><img src="${f.gradcam_url}"> <div class="gallery-label">Face #${f.rank} Grad-CAM</div></div>`;
          });
        }

        // Fingerprint
        document.getElementById('hashSha256').innerText = data.sha256;
        document.getElementById('hashPhash').innerText = data.phash;
        document.getElementById('ledgerBadge').innerHTML = data.registered_in_ledger
          ? '<span style="color: var(--accent-green); font-weight: 600;">✓ Registered into Forensic Ledger</span>'
          : '<span style="color: var(--text-muted);">Not registered in ledger</span>';

      } catch (err) {
        alert('Analysis error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = 'Analyze Authenticity';
      }
    }

    async function runVerification(file) {
      if (!file) return;
      const resultDiv = document.getElementById('verifyResult');
      resultDiv.style.display = 'block';
      resultDiv.innerHTML = '<div style="text-align: center; color: var(--accent-cyan);"><span class="spinner"></span> Checking ledger records...</div>';

      const formData = new FormData();
      formData.append('image', file);

      try {
        const resp = await fetch('/api/verify', { method: 'POST', body: formData });
        const res = await resp.json();

        let color = '#94a3b8';
        let icon = 'ℹ️';
        if (res.status === 'VERIFIED_AUTHENTIC_MATCH' || res.status === 'VERIFIED_EXACT_MATCH') {
          color = 'var(--accent-green)';
          icon = '🛡️';
        } else if (res.status === 'VERIFIED_AUTHENTIC_PERCEPTUAL' || res.status === 'VERIFIED_PERCEPTUAL_MATCH') {
          color = 'var(--accent-cyan)';
          icon = '🔍';
        } else if (res.status === 'KNOWN_MANIPULATED_MEDIA' || res.status === 'KNOWN_MANIPULATED_SIMILAR') {
          color = 'var(--accent-red)';
          icon = '🚨';
        } else if (res.status === 'SUSPICIOUS_SIMILARITY') {
          color = 'var(--accent-amber)';
          icon = '⚠️';
        } else if (res.status === 'NOT_REGISTERED' || res.status === 'NOT_FOUND') {
          color = '#94a3b8';
          icon = '❓';
        }

        resultDiv.innerHTML = `
          <div style="background: rgba(0,0,0,0.3); border: 1px solid ${color}; border-radius: 10px; padding: 20px;">
            <div style="font-size: 18px; font-weight: 700; color: ${color}; margin-bottom: 8px;">${icon} ${res.status.replace(/_/g, ' ')}</div>
            <div style="font-size: 13px; line-height: 1.6; margin-bottom: 16px;">${res.message}</div>
            <div class="hash-group">
              <div class="hash-label">Query File SHA-256</div>
              <div>${res.query_fingerprint.sha256}</div>
            </div>
            <div class="hash-group">
              <div class="hash-label">Query File pHash</div>
              <div>${res.query_fingerprint.phash}</div>
            </div>
            ${res.matched_record ? `
              <div style="margin-top: 14px; font-size: 12px; color: var(--text-muted);">
                Matched Ledger Record: <b>${res.matched_record.file_name}</b> (Registered: ${res.matched_record.timestamp})
                ${res.hamming_distance !== null ? ` | Hamming Distance: <b>${res.hamming_distance}</b>` : ''}
              </div>
            ` : ''}
          </div>
        `;
      } catch (err) {
        resultDiv.innerHTML = `<div style="color: var(--accent-red);">Error: ${err.message}</div>`;
      }
    }

    async function loadLedger() {
      const tbody = document.getElementById('ledgerBody');
      tbody.innerHTML = '<tr><td colspan="5" style="text-align: center;"><span class="spinner"></span> Loading ledger records...</td></tr>';
      try {
        const resp = await fetch('/api/ledger');
        const data = await resp.json();
        if (!data || data.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No records in ledger yet.</td></tr>';
          return;
        }
        tbody.innerHTML = data.slice().reverse().map(item => `
          <tr>
            <td><b>${item.file_name || 'N/A'}</b></td>
            <td>${item.timestamp ? item.timestamp.split('T')[0] + ' ' + item.timestamp.split('T')[1].slice(0,8) : 'N/A'}</td>
            <td><span class="status-tag ${(item.verdict || 'REAL').toLowerCase()}">${item.verdict || 'REAL'}</span></td>
            <td><code>${(item.sha256 || '').slice(0, 16)}...</code></td>
            <td><code>${item.phash || 'N/A'}</code></td>
          </tr>
        `).join('');
      } catch (err) {
        tbody.innerHTML = `<tr><td colspan="5" style="color: var(--accent-red); text-align: center;">Error loading ledger: ${err.message}</td></tr>`;
      }
    }
  </script>
</body>
</html>
"""


def parse_multipart(rfile, content_type: str, content_length: int):
    """Parse multipart/form-data body into fields and files without external libraries."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary=") :].strip('"')
            break
    if not boundary:
        raise ValueError("No boundary found in Content-Type")

    body = rfile.read(content_length)
    boundary_bytes = ("--" + boundary).encode("latin1")
    raw_parts = body.split(boundary_bytes)

    fields = {}
    files = {}

    for raw_part in raw_parts:
        if not raw_part or raw_part == b"--\r\n" or raw_part == b"--":
            continue
        # Split headers and data
        if b"\r\n\r\n" not in raw_part:
            continue
        header_data, content = raw_part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]

        headers_str = header_data.decode("latin1", errors="replace")
        name_match = re.search(r'name="([^"]+)"', headers_str)
        if not name_match:
            continue
        field_name = name_match.group(1)

        filename_match = re.search(r'filename="([^"]+)"', headers_str)
        if filename_match:
            filename = filename_match.group(1)
            files[field_name] = {"filename": filename, "data": content}
        else:
            fields[field_name] = content.decode("utf-8", errors="replace")

    return fields, files


class TrinetraHTTPHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/ledger":
            ledger_path = config.FORENSIC_LEDGER
            if ledger_path.exists():
                try:
                    with ledger_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = []
            else:
                data = []
            self.send_json(200, data)
            return

        # Serve static generated output files (annotated images, crops, gradcam)
        if path.startswith("/outputs/"):
            rel_path = path.lstrip("/")
            file_path = config.ROOT_DIR / rel_path
            if file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                if mime is None:
                    mime = "application/octet-stream"
                with file_path.open("rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(content)
                return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if path == "/api/analyze-image":
            try:
                fields, files = parse_multipart(self.rfile, content_type, content_length)
                if "image" not in files:
                    self.send_json(400, {"error": "No image file provided in request."})
                    return

                uploaded = files["image"]
                filename = uploaded["filename"] or "uploaded_image.jpg"
                temp_upload_dir = config.ROOT_DIR / "outputs" / "uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)
                target_path = temp_upload_dir / filename

                with target_path.open("wb") as f:
                    f.write(uploaded["data"])

                threshold = float(fields.get("threshold", 0.5))
                explain = fields.get("explain", "true").lower() == "true"
                register_ledger = fields.get("register", "false").lower() == "true"

                from process_image import process_image

                result = process_image(
                    image_path=target_path,
                    threshold=threshold,
                    explain=explain,
                    register_ledger=register_ledger,
                )

                # Format URLs for browser display
                annotated_url = None
                if result.annotated_image_path:
                    annotated_url = "/" + result.annotated_image_path.relative_to(config.ROOT_DIR).as_posix()

                faces_data = []
                for f in result.faces:
                    crop_url = "/" + f.crop_path.relative_to(config.ROOT_DIR).as_posix()
                    cam_url = "/" + f.gradcam_path.relative_to(config.ROOT_DIR).as_posix() if f.gradcam_path else None
                    faces_data.append({
                        "rank": f.rank,
                        "label": f.label,
                        "confidence": f.confidence,
                        "fake_probability": f.fake_probability,
                        "explanation": f.explanation,
                        "crop_url": crop_url,
                        "gradcam_url": cam_url,
                    })

                response_payload = {
                    "file_name": result.image_path.name,
                    "verdict": result.verdict,
                    "final_label": result.final_label,
                    "confidence": result.confidence,
                    "fake_probability": result.fake_probability,
                    "real_probability": result.real_probability,
                    "forensic_explanation": result.forensic_explanation,
                    "faces_detected": len(result.faces),
                    "sha256": result.fingerprint["sha256"],
                    "phash": result.fingerprint["phash"],
                    "registered_in_ledger": result.registered_in_ledger,
                    "annotated_image_url": annotated_url,
                    "faces": faces_data,
                }
                self.send_json(200, response_payload)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        if path == "/api/verify":
            try:
                fields, files = parse_multipart(self.rfile, content_type, content_length)
                if "image" not in files:
                    self.send_json(400, {"error": "No image file provided for verification."})
                    return

                uploaded = files["image"]
                temp_upload_dir = config.ROOT_DIR / "outputs" / "uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)
                target_path = temp_upload_dir / (uploaded["filename"] or "verify_image.jpg")

                with target_path.open("wb") as f:
                    f.write(uploaded["data"])

                verif_res = verify_integrity(target_path)
                self.send_json(200, verif_res)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
            return

        self.send_response(404)
        self.end_headers()


def run_server(port: int = PORT):
    server = HTTPServer((HOST, port), TrinetraHTTPHandler)
    print(f"\n=======================================================", flush=True)
    print(f"  TRINETRA Web Interface Started Successfully!", flush=True)
    print(f"  URL: http://{HOST}:{port}", flush=True)
    print(f"  Press Ctrl+C to stop the server.", flush=True)
    print(f"=======================================================\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TRINETRA server.", flush=True)
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port)
