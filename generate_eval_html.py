import os
import sys
import glob
import json
from pathlib import Path

def get_samples_for_model(model_dir):
    img_dir = Path(model_dir) / "images"
    if not img_dir.exists():
        return []

    real_A_files = sorted(list(img_dir.glob("*_real_A.png")))
    samples = []
    for f in real_A_files:
        sid = f.name.replace("_real_A.png", "")
        if (img_dir / f"{sid}_fake_B.png").exists() and \
           (img_dir / f"{sid}_real_B.png").exists() and \
           (img_dir / f"{sid}_fake_A.png").exists():
            samples.append(sid)
    return samples

def format_model_title(rel_path):
    titles = {
        "dual_encoder_cyclegan/test_latest": "Dual-Encoder CycleGAN (Latest — 71 Pairs)",
        "cyclegan_ssim/test_latest": "Paired SSIM λ=5.0 (Latest — 71 Pairs)",
        "cyclegan_paired_l1_10/test_200": "Paired L1 λ=10.0 (Epoch 200 — 71 Pairs)",
        "cyclegan_paired_l1_p5/test_200": "Paired L1 λ=0.5 (Epoch 200 — 71 Pairs)",
        "cyclegan_unaligned/test_latest": "Unpaired CycleGAN (Latest — 71 Pairs)",
        "cyclegan_paired_ssim/test_200": "Paired SSIM (Epoch 200 — 50 Samples)",
        "cyclegan_unaligned/test_200": "Unpaired CycleGAN (Epoch 200 — 50 Samples)",
        "cyclegan_unaligned/test_300": "Unpaired CycleGAN (Epoch 300 — 50 Samples)",
        "fundus_cyclegan_topodlri/test_100": "Fundus Topo DLRI (Epoch 100 — 50 Samples)",
        "fundus_cyclegan_topodlri/test_200": "Fundus Topo DLRI (Epoch 200 — 50 Samples)"
    }
    if rel_path in titles:
        return titles[rel_path]
    parts = rel_path.split('/')
    m_name = parts[0].replace('_', ' ').title()
    epoch = parts[1] if len(parts) > 1 else ''
    return f"{m_name} ({epoch})"

def build_html_page(samples, current_rel_path, model_title, all_models_dict, is_master=False):
    prefix_to_root = "" if is_master else "../../"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Retinal Evaluation — {model_title}</title>
<style>
:root {{
    --bg-primary: #0b1120;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-real: #10b981;
    --accent-fake: #a855f7;
    --accent-gt: #0ea5e9;
    --accent-rev: #f59e0b;
    --img-size: 540px;
}}

body.light-mode {{
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --border-color: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #64748b;
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    padding-top: 76px;
    padding-bottom: 60px;
    transition: background-color 0.2s, color 0.2s;
}}

/* Sticky Navbar */
header.top-nav {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 68px;
    background-color: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    z-index: 1000;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}}

.nav-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nav-left h1 {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}}

select.select-model {{
    background: var(--bg-primary);
    color: #a855f7;
    border: 1.5px solid rgba(168, 85, 247, 0.4);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    outline: none;
}}

.nav-center {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nav-btn {{
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
}}

.nav-btn:hover {{
    background: #3b82f6;
    color: #fff;
    border-color: #3b82f6;
}}

.counter-badge {{
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 500;
    white-space: nowrap;
}}

.counter-badge b {{
    color: var(--text-primary);
}}

select.select-jump {{
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    outline: none;
    max-width: 220px;
    cursor: pointer;
}}

.nav-right {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.size-controls {{
    display: flex;
    align-items: center;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    overflow: hidden;
}}

.size-btn {{
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 6px 11px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
}}

.size-btn.active {{
    background: #3b82f6;
    color: #ffffff;
}}

.theme-toggle-btn {{
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 15px;
}}

/* Main Container */
main.container {{
    max-width: 1480px;
    margin: 0 auto;
    padding: 16px 20px;
}}

/* Model Overview Hub Grid */
.models-hub-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 24px;
}}

.models-hub-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}}

.models-hub-header h2 {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}}

.models-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 12px;
}}

.model-entry {{
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 10px;
    transition: border-color 0.15s;
}}

.model-entry:hover {{
    border-color: #3b82f6;
}}

.model-entry.current {{
    border-color: #a855f7;
    background: rgba(168, 85, 247, 0.08);
}}

.model-entry-title {{
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
}}

.model-entry-meta {{
    font-size: 11.5px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
}}

.model-entry-actions {{
    display: flex;
    gap: 8px;
}}

.btn-model-link {{
    flex: 1;
    text-align: center;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.15s;
}}

.btn-primary-eval {{
    background: #3b82f6;
    color: #fff;
    border: 1px solid #3b82f6;
}}

.btn-primary-eval:hover {{
    background: #2563eb;
}}

.btn-secondary-grid {{
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}}

.btn-secondary-grid:hover {{
    background: var(--bg-secondary);
    color: var(--text-primary);
}}

/* Evaluation Guide Banner */
.info-banner {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    gap: 16px;
    flex-wrap: wrap;
}}

body.light-mode .info-banner {{
    background: #f1f5f9;
}}

.legend-tags {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
}}

.dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
}}

/* Card Container */
.sample-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-bottom: 36px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.15s, box-shadow 0.15s;
}}

.card-header {{
    background: rgba(0,0,0,0.2);
    padding: 12px 20px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

body.light-mode .card-header {{
    background: #f8fafc;
}}

.card-title {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}}

.sample-num {{
    background: #3b82f6;
    color: #ffffff;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
}}

.card-actions {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.star-btn {{
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.star-btn.starred {{
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border-color: #f59e0b;
}}

/* 4-Part Square Grid (2x2) */
.quadrant-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(320px, 1fr));
    gap: 16px;
    padding: 18px;
}}

.quadrant-cell {{
    display: flex;
    flex-direction: column;
    background: rgba(0,0,0,0.25);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
}}

body.light-mode .quadrant-cell {{
    background: #f8fafc;
}}

.cell-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 14px;
    font-size: 12px;
    font-weight: 700;
    border-bottom: 1px solid var(--border-color);
    background: rgba(0,0,0,0.25);
}}

body.light-mode .cell-header {{
    background: #f1f5f9;
}}

.tag-real-A {{
    color: var(--accent-real);
}}
.tag-fake-B {{
    color: var(--accent-fake);
}}
.tag-real-B {{
    color: var(--accent-gt);
}}
.tag-fake-A {{
    color: var(--accent-rev);
}}

.cell-domain {{
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
}}

.img-wrapper {{
    position: relative;
    width: 100%;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #000;
    cursor: zoom-in;
    min-height: 280px;
}}

.img-wrapper img {{
    width: 100%;
    max-width: var(--img-size);
    height: auto;
    display: block;
    object-fit: contain;
    transition: transform 0.2s ease;
}}

.img-wrapper:hover img {{
    transform: scale(1.02);
}}

.zoom-overlay-hint {{
    position: absolute;
    bottom: 8px;
    right: 8px;
    background: rgba(0,0,0,0.75);
    color: #fff;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
}}

.img-wrapper:hover .zoom-overlay-hint {{
    opacity: 1;
}}

/* Lightbox Modal */
.modal-overlay {{
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.94);
    z-index: 2000;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}}

.modal-overlay.open {{
    display: flex;
}}

.modal-header {{
    position: absolute;
    top: 14px;
    left: 20px;
    right: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #fff;
}}

.modal-img-container {{
    max-width: 92vw;
    max-height: 84vh;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.modal-img-container img {{
    max-width: 92vw;
    max-height: 84vh;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 8px 36px rgba(0,0,0,0.9);
}}

.modal-close-btn {{
    background: rgba(255,255,255,0.2);
    border: none;
    color: #fff;
    font-size: 20px;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.modal-close-btn:hover {{
    background: rgba(255,255,255,0.4);
}}

.modal-tabs {{
    margin-top: 14px;
    display: flex;
    gap: 10px;
}}

.modal-tab-btn {{
    background: rgba(255,255,255,0.15);
    color: #fff;
    border: none;
    padding: 7px 16px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
}}

.modal-tab-btn.active {{
    background: #3b82f6;
}}
</style>
</head>
<body>

<!-- Sticky Header -->
<header class="top-nav">
    <div class="nav-left">
        <h1><a href="{prefix_to_root}index.html" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;">🔬 Retinal Eval</a></h1>
        <select class="select-model" id="model-select" onchange="onModelSelectChange(this.value)">
            <option value="__master__">🏠 Master Evaluation Hub</option>
            <optgroup label="── Aligned Test Set (71 Pairs) ──">
"""

    aligned_keys = [k for k in all_models_dict.keys() if "71" in all_models_dict[k] or k in [
        "dual_encoder_cyclegan/test_latest",
        "cyclegan_ssim/test_latest",
        "cyclegan_paired_l1_10/test_200",
        "cyclegan_paired_l1_p5/test_200",
        "cyclegan_unaligned/test_latest"
    ]]
    other_keys = [k for k in all_models_dict.keys() if k not in aligned_keys]

    for k in aligned_keys:
        selected = ' selected' if k == current_rel_path else ''
        html += f'                <option value="{k}"{selected}>{all_models_dict[k]}</option>\n'

    html += """            </optgroup>
            <optgroup label="── Original Test Set (50 Pairs) ──">
"""
    for k in other_keys:
        selected = ' selected' if k == current_rel_path else ''
        html += f'                <option value="{k}"{selected}>{all_models_dict[k]}</option>\n'

    html += f"""            </optgroup>
        </select>
    </div>

    <div class="nav-center">
        <button class="nav-btn" onclick="prevSample()">◀ Prev</button>
        <span class="counter-badge">Sample <b id="current-idx">1</b> / {len(samples)}</span>
        <button class="nav-btn" onclick="nextSample()">Next ▶</button>
        
        <select class="select-jump" id="jump-select" onchange="jumpToSample(this.value)">
"""
    for i, sid in enumerate(samples):
        html += f'            <option value="sample-{sid}">[{i+1}] {sid}</option>\n'

    html += f"""        </select>
    </div>

    <div class="nav-right">
        <div class="size-controls">
            <button class="size-btn" onclick="setImageSize('420px', this)">Compact</button>
            <button class="size-btn active" onclick="setImageSize('540px', this)">Large</button>
            <button class="size-btn" onclick="setImageSize('750px', this)">X-Large</button>
            <button class="size-btn" onclick="setImageSize('100%', this)">Full</button>
        </div>

        <button class="theme-toggle-btn" onclick="toggleTheme()" title="Toggle Dark/Light Mode">🌙</button>
    </div>
</header>

<main class="container">
    <!-- Legend Banner -->
    <div class="info-banner">
        <div>
            <strong>Evaluation Layout:</strong> Real inputs and generated outputs are arranged in a 2×2 square grid (Real on Left, Fake on Right). Reconstructed images and vessel masks are excluded.
        </div>
        <div class="legend-tags">
            <span class="legend-item"><span class="dot" style="background:var(--accent-real)"></span> Real Fundus (real_A)</span>
            <span class="legend-item"><span class="dot" style="background:var(--accent-fake)"></span> Fake Confocal (fake_B)</span>
            <span class="legend-item"><span class="dot" style="background:var(--accent-gt)"></span> Real Confocal GT (real_B)</span>
            <span class="legend-item"><span class="dot" style="background:var(--accent-rev)"></span> Fake Fundus (fake_A)</span>
        </div>
    </div>

    <!-- Samples List -->
"""

    if is_master:
        html += """    <section class="models-hub-card">
        <div class="models-hub-header">
            <h2>📊 Model Evaluation Hub (10 Checkpoints Evaluated)</h2>
            <span style="font-size:12px;color:var(--text-secondary);">Click any model below or use the header dropdown to inspect.</span>
        </div>
        <div class="models-grid">
"""
        for m_rel, m_title in all_models_dict.items():
            is_curr = ' current' if m_rel == current_rel_path else ''
            sample_count = "71 Aligned Pairs" if "71" in m_title else "50 Pairs (Legacy)"
            html += f"""            <div class="model-entry{is_curr}">
                <div>
                    <div class="model-entry-title">{m_title}</div>
                    <div class="model-entry-meta"><span>📁 {m_rel}</span> • <span>{sample_count}</span></div>
                </div>
                <div class="model-entry-actions">
                    <a class="btn-model-link btn-primary-eval" href="{m_rel}/eval.html">🔍 4-Quadrant Eval</a>
                    <a class="btn-model-link btn-secondary-grid" href="{m_rel}/index.html" target="_blank">📋 CycleGAN Grid</a>
                </div>
            </div>
"""
        html += """        </div>
    </section>
"""

    img_prefix = "" if not is_master else f"{current_rel_path}/"

    for i, sid in enumerate(samples):
        html += f"""
    <section class="sample-card" id="sample-{sid}">
        <div class="card-header">
            <div class="card-title">
                <span class="sample-num">#{i+1}</span>
                <span>Patient ID: {sid}</span>
            </div>
            <div class="card-actions">
                <button class="star-btn" onclick="toggleStar('{sid}', this)">⭐ Star Sample</button>
            </div>
        </div>

        <div class="quadrant-grid">
            <!-- Top-Left: real_A -->
            <div class="quadrant-cell">
                <div class="cell-header">
                    <span class="tag-real-A">🟢 REAL FUNDUS (Domain A)</span>
                    <span class="cell-domain">Input Photograph</span>
                </div>
                <div class="img-wrapper" onclick="openModal('{sid}', 'real_A')">
                    <img id="img-{sid}-real_A" src="{img_prefix}images/{sid}_real_A.png" alt="{sid} real_A" loading="lazy">
                    <span class="zoom-overlay-hint">🔍 Click to zoom</span>
                </div>
            </div>

            <!-- Top-Right: fake_B -->
            <div class="quadrant-cell">
                <div class="cell-header">
                    <span class="tag-fake-B">🟣 SYNTHESIZED CONFOCAL (fake_B)</span>
                    <span class="cell-domain">A → B Output</span>
                </div>
                <div class="img-wrapper" onclick="openModal('{sid}', 'fake_B')">
                    <img id="img-{sid}-fake_B" src="{img_prefix}images/{sid}_fake_B.png" alt="{sid} fake_B" loading="lazy">
                    <span class="zoom-overlay-hint">🔍 Click to zoom</span>
                </div>
            </div>

            <!-- Bottom-Left: real_B -->
            <div class="quadrant-cell">
                <div class="cell-header">
                    <span class="tag-real-B">🔵 REAL CONFOCAL (Domain B)</span>
                    <span class="cell-domain">Target Ground Truth</span>
                </div>
                <div class="img-wrapper" onclick="openModal('{sid}', 'real_B')">
                    <img id="img-{sid}-real_B" src="{img_prefix}images/{sid}_real_B.png" alt="{sid} real_B" loading="lazy">
                    <span class="zoom-overlay-hint">🔍 Click to zoom</span>
                </div>
            </div>

            <!-- Bottom-Right: fake_A -->
            <div class="quadrant-cell">
                <div class="cell-header">
                    <span class="tag-fake-A">🟡 SYNTHESIZED FUNDUS (fake_A)</span>
                    <span class="cell-domain">B → A Output</span>
                </div>
                <div class="img-wrapper" onclick="openModal('{sid}', 'fake_A')">
                    <img id="img-{sid}-fake_A" src="{img_prefix}images/{sid}_fake_A.png" alt="{sid} fake_A" loading="lazy">
                    <span class="zoom-overlay-hint">🔍 Click to zoom</span>
                </div>
            </div>
        </div>
    </section>
"""

    html += f"""
</main>

<!-- Lightbox Modal -->
<div class="modal-overlay" id="lightbox-modal" onclick="closeModalOnOverlay(event)">
    <div class="modal-header">
        <div id="modal-title" style="font-size:15px;font-weight:700;"></div>
        <button class="modal-close-btn" onclick="closeModal()">✕</button>
    </div>
    
    <div class="modal-img-container">
        <img id="modal-img" src="" alt="Zoomed View">
    </div>

    <div class="modal-tabs">
        <button class="modal-tab-btn" id="mtab-real_A" onclick="switchModalKey('real_A')">1. Real Fundus (A)</button>
        <button class="modal-tab-btn" id="mtab-fake_B" onclick="switchModalKey('fake_B')">2. Fake Confocal (B)</button>
        <button class="modal-tab-btn" id="mtab-real_B" onclick="switchModalKey('real_B')">3. Real Confocal GT (B)</button>
        <button class="modal-tab-btn" id="mtab-fake_A" onclick="switchModalKey('fake_A')">4. Fake Fundus (A)</button>
    </div>
</div>

<script>
const isMasterDashboard = {"true" if is_master else "false"};
const currentModelRel = "{current_rel_path}";
let currentModelPrefix = "{img_prefix}";
let currentSampleId = "";
let currentModalKey = "fake_B";
const sampleKeys = {json.dumps(samples)};
let currentIdx = 0;

function onModelSelectChange(newModelRel) {{
    if (newModelRel === "__master__") {{
        window.location.href = isMasterDashboard ? "index.html" : "../../index.html";
        return;
    }}
    if (isMasterDashboard) {{
        window.location.href = newModelRel + "/eval.html";
    }} else {{
        window.location.href = `../../${{newModelRel}}/eval.html`;
    }}
}}

function setImageSize(size, btn) {{
    document.documentElement.style.setProperty('--img-size', size);
    document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
}}

function toggleTheme() {{
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    document.querySelector('.theme-toggle-btn').innerText = isLight ? "☀️" : "🌙";
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
}}

if (localStorage.getItem('theme') === 'light') {{
    document.body.classList.add('light-mode');
    document.querySelector('.theme-toggle-btn').innerText = "☀️";
}}

function jumpToSample(targetId) {{
    const el = document.getElementById(targetId);
    if (el) {{
        el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        const id = targetId.replace('sample-', '');
        currentIdx = sampleKeys.indexOf(id);
        updateCounter();
    }}
}}

function updateCounter() {{
    document.getElementById('current-idx').innerText = currentIdx + 1;
    document.getElementById('jump-select').value = "sample-" + sampleKeys[currentIdx];
}}

function nextSample() {{
    if (currentIdx < sampleKeys.length - 1) {{
        currentIdx++;
        jumpToSample("sample-" + sampleKeys[currentIdx]);
    }}
}}

function prevSample() {{
    if (currentIdx > 0) {{
        currentIdx--;
        jumpToSample("sample-" + sampleKeys[currentIdx]);
    }}
}}

function toggleStar(sampleId, btn) {{
    const starred = JSON.parse(localStorage.getItem('starred_samples') || '{{}}');
    if (starred[sampleId]) {{
        delete starred[sampleId];
        btn.classList.remove('starred');
    }} else {{
        starred[sampleId] = true;
        btn.classList.add('starred');
    }}
    localStorage.setItem('starred_samples', JSON.stringify(starred));
}}

document.addEventListener("DOMContentLoaded", () => {{
    const starred = JSON.parse(localStorage.getItem('starred_samples') || '{{}}');
    document.querySelectorAll('.sample-card').forEach(card => {{
        const sid = card.id.replace('sample-', '');
        if (starred[sid]) {{
            const btn = card.querySelector('.star-btn');
            if (btn) btn.classList.add('starred');
        }}
    }});
}});

function openModal(sampleId, key) {{
    currentSampleId = sampleId;
    currentModalKey = key;
    updateModal();
    document.getElementById('lightbox-modal').classList.add('open');
}}

function closeModal() {{
    document.getElementById('lightbox-modal').classList.remove('open');
}}

function closeModalOnOverlay(e) {{
    if (e.target.id === 'lightbox-modal') closeModal();
}}

function switchModalKey(key) {{
    currentModalKey = key;
    updateModal();
}}

function updateModal() {{
    const imgPath = `${{currentModelPrefix}}images/${{currentSampleId}}_${{currentModalKey}}.png`;
    document.getElementById('modal-img').src = imgPath;
    document.getElementById('modal-title').innerText = `${{currentSampleId}} — ${{currentModalKey.toUpperCase()}}`;
    
    document.querySelectorAll('.modal-tab-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`mtab-${{currentModalKey}}`);
    if (activeBtn) activeBtn.classList.add('active');
}}

window.addEventListener('keydown', (e) => {{
    const modal = document.getElementById('lightbox-modal');
    if (modal.classList.contains('open')) {{
        if (e.key === 'Escape') closeModal();
        if (e.key === '1') switchModalKey('real_A');
        if (e.key === '2') switchModalKey('fake_B');
        if (e.key === '3') switchModalKey('real_B');
        if (e.key === '4') switchModalKey('fake_A');
    }} else {{
        if (e.key === 'ArrowRight' || e.key === 'j') nextSample();
        if (e.key === 'ArrowLeft' || e.key === 'k') prevSample();
    }}
}});
</script>
</body>
</html>
"""
    return html

def main():
    proj_dir = Path(__file__).resolve().parent
    results_base = proj_dir / "results"

    # Automatically discover all test results directories with images/
    all_img_dirs = sorted(list(results_base.glob("**/images")))
    all_models_dict = {}

    valid_models = []
    for d in all_img_dirs:
        model_dir = d.parent
        rel_path = str(model_dir.relative_to(results_base))
        samples = get_samples_for_model(model_dir)
        if samples:
            title = format_model_title(rel_path)
            all_models_dict[rel_path] = title
            valid_models.append((model_dir, rel_path, title, samples))

    print(f"Found {len(valid_models)} valid test result directories in {results_base}:")
    for _, rel_path, title, samples in valid_models:
        print(f"  • {rel_path} ({len(samples)} samples): {title}")

    # Generate eval.html inside each model's directory
    for model_dir, rel_path, title, samples in valid_models:
        html = build_html_page(samples, rel_path, title, all_models_dict, is_master=False)
        out_p = model_dir / "eval.html"
        out_p.write_text(html, encoding="utf-8")
        print(f"Created: {out_p}")

    # Generate master results/eval.html and results/index.html using the primary model (dual_encoder_cyclegan)
    primary_samples = get_samples_for_model(results_base / "dual_encoder_cyclegan" / "test_latest")
    if primary_samples:
        master_html = build_html_page(
            primary_samples, 
            "dual_encoder_cyclegan/test_latest", 
            all_models_dict.get("dual_encoder_cyclegan/test_latest", "Dual-Encoder CycleGAN"), 
            all_models_dict, 
            is_master=True
        )
        master_eval = results_base / "eval.html"
        master_eval.write_text(master_html, encoding="utf-8")
        print(f"Created Master Dashboard: {master_eval}")

        master_index = results_base / "index.html"
        master_index.write_text(master_html, encoding="utf-8")
        print(f"Created Master Index: {master_index}")

if __name__ == "__main__":
    main()
