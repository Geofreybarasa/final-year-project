// ─────────────────────────────────────────────────────────────
// server.js  (or add these routes into your existing Express app)
//
// This is the GLUE between the EJS frontend and the Flask backend.
//
// Flow:
//   Browser  →  POST /upload (multipart, field: image)
//   Express  →  proxies to Flask POST http://localhost:5000/predict
//   Flask    →  returns JSON  { prediction, confidence, gradcam, fft, ela, face, explanation }
//   Express  →  forwards JSON back to browser  (fetch path)
//              OR re-renders index.ejs with result injected  (no-JS fallback)
// ─────────────────────────────────────────────────────────────

const express  = require('express');
const multer   = require('multer');
const FormData = require('form-data');
const fetch    = require('node-fetch');          // npm i node-fetch@2
const path     = require('path');

const app    = express();
const upload = multer({ storage: multer.memoryStorage() });

const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5000';

// Static assets
app.use('/css', express.static(path.join(__dirname, 'public/css')));
app.use('/js',  express.static(path.join(__dirname, 'public/js')));

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// ── GET /  →  render empty template ──────────────────────────
app.get('/', (_req, res) => {
    res.render('index', { result: null });
});

// ── POST /upload  →  proxy to Flask /predict ─────────────────
app.post('/upload', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }

    try {
        // Re-pack the file into a new FormData for Flask
        const form = new FormData();
        form.append('image', req.file.buffer, {
            filename    : req.file.originalname,
            contentType : req.file.mimetype,
        });

        const flaskRes = await fetch(`${FLASK_URL}/predict`, {
            method  : 'POST',
            body    : form,
            headers : form.getHeaders(),
        });

        const result = await flaskRes.json();

        // If the request came from JS fetch → return JSON directly
        const wantsJson = req.headers['accept']?.includes('application/json')
                       || req.xhr
                       || req.headers['x-requested-with'] === 'XMLHttpRequest';

        if (wantsJson) {
            return res.json(result);
        }

        // No-JS fallback: re-render the EJS template with result injected
        return res.render('index', { result });

    } catch (err) {
        console.error('Flask proxy error:', err);
        const errPayload = { error: 'Analysis service unavailable — ' + err.message };

        if (req.xhr) return res.status(502).json(errPayload);
        return res.render('index', { result: errPayload });
    }
});

// ── Start ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`VERITAS frontend  →  http://localhost:${PORT}`);
    console.log(`Flask backend     →  ${FLASK_URL}/predict`);
});