// ─── Element refs ─────────────────────────────────────────────
const dropZone   = document.getElementById('dropZone');
const fileInput  = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const preview    = document.getElementById('filePreview');
const previewImg = document.getElementById('previewImg');
const fileNameEl = document.getElementById('fileName');
const dropIcon   = document.getElementById('dropIcon');
const dropTitle  = document.getElementById('dropTitle');

let droppedFile = null;

// ─── Drag & Drop ──────────────────────────────────────────────
dropZone.addEventListener('click', (e) => {
    if (!e.target.closest('#analyzeBtn') && e.target !== fileInput) {
        fileInput.click();
    }
});

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) {
        droppedFile = file;
        showPreview(file);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) {
        droppedFile = null;
        showPreview(fileInput.files[0]);
    }
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src         = e.target.result;
        preview.style.display  = 'block';
        dropIcon.style.display = 'none';
        fileNameEl.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
        dropTitle.textContent  = 'Image loaded — ready for analysis';
    };
    reader.readAsDataURL(file);
}

// ─── Analyse button ────────────────────────────────────────────
analyzeBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();

    const file = droppedFile || fileInput.files[0];
    if (!file) {
        alert('Please select or drop an image first.');
        return;
    }

    // Reset previous results
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.style.display = 'none';
    document.getElementById('heatmapSection').style.display = 'none';
    document.getElementById('errorMsg').style.display       = 'none';

    // Scanning state
    document.getElementById('scanBar').style.display = 'block';
    analyzeBtn.textContent = 'Analysing…';
    analyzeBtn.disabled    = true;

    const formData = new FormData();
    formData.append('image', file, file.name);

    fetch('/upload', { method: 'POST', body: formData })
        .then(res => res.text())
        .then(html => {
            const parser    = new DOMParser();
            const doc       = parser.parseFromString(html, 'text/html');
            const resultDiv = doc.getElementById('server-result');
            if (resultDiv) {
                const result = JSON.parse(resultDiv.dataset.result);
                renderResult(result);
            } else {
                renderResult({ error: 'No result returned from server.' });
            }
        })
        .catch(err => {
            console.error('Upload error:', err);
            renderResult({ error: 'Upload failed. Please try again.' });
        })
        .finally(() => {
            document.getElementById('scanBar').style.display = 'none';
            analyzeBtn.textContent = 'Run Analysis';
            analyzeBtn.disabled    = false;
            droppedFile            = null;
        });
});

// ─── Result renderer ───────────────────────────────────────────
function renderResult(result) {
    const panel       = document.getElementById('resultPanel');
    const verdictTxt  = document.getElementById('verdictText');
    const verdictLbl  = document.getElementById('verdictLabel');
    const verdictIco  = document.getElementById('verdictIcon');
    const confFill    = document.getElementById('confFill');
    const confPct     = document.getElementById('confPct');
    const metricConf  = document.getElementById('metricConf');
    const metricVerd  = document.getElementById('metricVerdict');
    const heatSection = document.getElementById('heatmapSection');
    const heatImg     = document.getElementById('heatmapImg');
    const errorMsg    = document.getElementById('errorMsg');

    // Reset
    errorMsg.style.display    = 'none';
    heatSection.style.display = 'none';
    panel.style.display       = 'block';

    if (result.error) {
        verdictLbl.textContent = 'ANALYSIS ERROR';
        verdictTxt.textContent = 'ERROR';
        verdictTxt.className   = 'verdict-text error';
        verdictIco.className   = 'verdict-icon error';
        verdictIco.textContent = '⚠';
        errorMsg.style.display = 'block';
        errorMsg.textContent   = result.error;
        panel.scrollIntoView({ behavior: 'smooth' });
        return;
    }

    const isReal = result.prediction === 'real';
    const cls    = result.prediction;
    const pct    = result.confidence || 0;

    verdictLbl.textContent = 'ANALYSIS COMPLETE';
    verdictTxt.textContent = cls.toUpperCase();
    verdictTxt.className   = `verdict-text ${cls}`;
    verdictIco.className   = `verdict-icon ${cls}`;
    verdictIco.textContent = isReal ? '✓' : '✕';

    confFill.className = `conf-fill ${cls}`;
    setTimeout(() => { confFill.style.width = `${pct}%`; }, 50);
    confPct.textContent    = `${pct}%`;
    metricConf.textContent = `${pct}%`;
    metricVerd.textContent = cls.toUpperCase();

    // ── Forensic Panel ─────────────────────────────────────────
    if (result.heatmap) {
        // Update label
        const heatTitle = document.querySelector('.heatmap-title');
        if (heatTitle) heatTitle.textContent = 'FORENSIC ANALYSIS PANEL';

        // Image — full width, crisp rendering
        heatImg.src = result.heatmap;
        heatImg.style.cssText = `
            width: 100%;
            height: auto;
            display: block;
            border: 1px solid rgba(0,229,255,0.2);
            border-radius: 4px;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
            margin-top: 12px;
            cursor: zoom-in;
        `;

        // Click image to open full size in new tab
        heatImg.onclick = () => {
            const win = window.open();
            win.document.write(`
                <html><head><title>VERITAS — Forensic Panel</title>
                <style>body{margin:0;background:#080c10;display:flex;justify-content:center;align-items:flex-start;}
                img{max-width:100%;height:auto;}</style></head>
                <body><img src="${result.heatmap}"/></body></html>
            `);
        };

        // Download button
        let dlBtn = document.getElementById('downloadPanel');
        if (!dlBtn) {
            dlBtn = document.createElement('a');
            dlBtn.id            = 'downloadPanel';
            dlBtn.textContent   = '⬇  Download Full Panel';
            dlBtn.style.cssText = `
                display: inline-block;
                margin-top: 12px;
                padding: 8px 18px;
                background: rgba(0,229,255,0.08);
                border: 1px solid rgba(0,229,255,0.35);
                color: #00e5ff;
                font-family: 'Syne Mono', monospace;
                font-size: 0.7rem;
                letter-spacing: 0.1em;
                border-radius: 3px;
                cursor: pointer;
                text-decoration: none;
                transition: background 0.2s;
            `;
            dlBtn.onmouseover = () => dlBtn.style.background = 'rgba(0,229,255,0.15)';
            dlBtn.onmouseout  = () => dlBtn.style.background = 'rgba(0,229,255,0.08)';
            heatSection.appendChild(dlBtn);
        }
        dlBtn.href     = result.heatmap;
        dlBtn.download = `veritas-forensic-${Date.now()}.jpg`;

        heatSection.style.display = 'block';
    }

    panel.scrollIntoView({ behavior: 'smooth' });
}

// ─── Bootstrap: EJS server-side injection ─────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const resultDiv = document.getElementById('server-result');
    if (resultDiv) {
        try {
            const result = JSON.parse(resultDiv.dataset.result);
            renderResult(result);
        } catch (e) {
            console.error('Failed to parse server result:', e);
        }
    }
});