const express      = require('express');
const path         = require('path');
const uploadRouter = require('./routes/upload');
require('dotenv').config();

const app  = express();
const PORT = process.env.PORT || 3000;

// ─── View engine ──────────────────────────────────────────────
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// ─── Static files ─────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// ─── Routes ───────────────────────────────────────────────────
app.get('/', (req, res) => {
    res.render('index', { result: null });
});

app.use('/upload', uploadRouter);

// ─── Start ────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`Node server  →  http://localhost:${PORT}`);
    console.log(`Flask server →  http://localhost:5000  (must be running separately)`);
});