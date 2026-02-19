const express = require('express');
const path = require('path');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
require('dotenv').config();
const FormData = require('form-data');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE_URL = process.env.PYTHON_URL || 'http://localhost:5000/predict';

// Set up EJS as templating engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files (CSS, client-side JS)
app.use(express.static(path.join(__dirname, 'public')));

// Configure multer for file uploads
const upload = multer({
    dest: 'public/uploads/',
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB max
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png/;
        const ext = file.originalname.toLowerCase().match(/\.(jpg|jpeg|png)$/);
        if (ext) {
            cb(null, true);
        } else {
            cb(new Error('Only .jpg, .jpeg, .png files are allowed'));
        }
    }
});

// Home route – render upload form
app.get('/', (req, res) => {
    res.render('index', { result: null });
});

// Upload and predict route
app.post('/upload', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.render('index', { result: { error: 'No image uploaded' } });
    }

    try {
        // Send the image to Python service
        const formData = new FormData();
        const fileStream = fs.createReadStream(req.file.path);
        formData.append('image', fileStream, req.file.originalname);

        const response = await axios.post(PYTHON_SERVICE_URL, formData, {
            headers: formData.getHeaders(),
            timeout: 60000 // 60 seconds
        });

        // Delete temporary file after use (optional)
        fs.unlink(req.file.path, (err) => {
            if (err) console.error('Error deleting temp file:', err);
        });

        // Render result page
        res.render('index', { result: response.data });
    } catch (error) {
        console.error('Prediction error:', error.message);
        // Clean up temp file on error
        fs.unlink(req.file.path, () => {});
        res.render('index', { result: { error: 'Prediction service unavailable' } });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        return res.render('index', { result: { error: err.message } });
    } else if (err) {
        return res.render('index', { result: { error: err.message } });
    }
    next();
});

app.listen(PORT, () => {
    console.log(`Node server running on http://localhost:${PORT}`);
});