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

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragging');
});

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

// ─── Show Preview ─────────────────────────────────────────────
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
analyzeBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const file = droppedFile || fileInput.files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    // UI Reset
    const errorMsg    = document.getElementById('errorMsg');
    const resultPanel = document.getElementById('resultPanel');
    const heatSection = document.getElementById('heatmapSection');

    errorMsg.style.display    = 'none';
    resultPanel.style.display = 'none';
    heatSection.style.display = 'none';

    // Show loading
    document.getElementById('scanBar').style.display = 'block';
    analyzeBtn.textContent = 'Analysing…';
    analyzeBtn.disabled    = true;

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const text = await response.text();

        // Parse returned HTML
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        const resultScript = doc.getElementById('server-result');

        if (resultScript && resultScript.textContent.trim()) {
            const result = JSON.parse(resultScript.textContent);
            renderResult(result);
        } else {
            renderResult({ error: "No result returned from server." });
        }

    } catch (err) {
        console.error("Request error:", err);
        renderResult({ error: "Upload failed. Check server connection." });
    } finally {
        document.getElementById('scanBar').style.display = 'none';
        analyzeBtn.textContent = 'RUN ANALYSIS';
        analyzeBtn.disabled    = false;
        droppedFile            = null;
    }
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

    panel.style.display = 'block';
    errorMsg.style.display = 'none';

    if (result.error) {
        verdictLbl.textContent = 'ANALYSIS ERROR';
        verdictTxt.textContent = 'FAILED';
        verdictTxt.className   = 'verdict-text error';
        verdictIco.textContent = '⚠';
        verdictIco.className   = 'verdict-icon error';

        errorMsg.style.display = 'block';
        errorMsg.textContent   = result.error;

        panel.scrollIntoView({ behavior: 'smooth' });
        return;
    }

    const cls = (result.prediction || 'unknown').toLowerCase();
    const pct = result.confidence || 0;

    // Verdict
    verdictLbl.textContent = 'ANALYSIS COMPLETE';
    verdictTxt.textContent = cls.toUpperCase();
    verdictTxt.className   = `verdict-text ${cls}`;
    verdictIco.textContent = cls === 'real' ? '✓' : '✕';
    verdictIco.className   = `verdict-icon ${cls}`;

    // Confidence
    confPct.textContent    = `${pct}%`;
    metricConf.textContent = `${pct}%`;
    metricVerd.textContent = cls.toUpperCase();

    confFill.className = `conf-fill ${cls}`;
    setTimeout(() => {
        confFill.style.width = `${pct}%`;
    }, 50);

    // Heatmap / forensic panel
    if (result.heatmap) {
        heatImg.src = result.heatmap;
        heatSection.style.display = 'block';

        // Optional: click to open full image
        heatImg.onclick = () => {
            const win = window.open();
            win.document.write(`<img src="${result.heatmap}" style="width:100%">`);
        };
    }

    panel.scrollIntoView({ behavior: 'smooth' });
}

// ─── Bootstrap (handles page reload results) ──────────────────
window.addEventListener('DOMContentLoaded', () => {
    const el = document.getElementById('server-result');

    if (el && el.textContent.trim().length > 5) {
        try {
            const result = JSON.parse(el.textContent);
            renderResult(result);
        } catch (e) {
            console.error("Initial parse failed:", e);
        }
    }
});