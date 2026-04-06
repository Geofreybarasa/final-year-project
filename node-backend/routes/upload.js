const express  = require('express');
const multer   = require('multer');
const fetch    = require('node-fetch');
const FormData = require('form-data');

const router = express.Router();

const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 }
});

router.post('/', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.render('index', {
            result: { error: 'No image file received.' }
        });
    }

    try {
        const form = new FormData();

        form.append('image', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        let flaskRes;

        try {
            flaskRes = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: form,
                headers: form.getHeaders()
            });
        } catch (err) {
            return res.render('index', {
                result: { error: 'Flask server is not running on port 5000' }
            });
        }

        if (!flaskRes.ok) {
            const txt = await flaskRes.text();
            return res.render('index', {
                result: { error: `Flask error ${flaskRes.status}: ${txt}` }
            });
        }

        const result = await flaskRes.json();

        return res.render('index', { result });

    } catch (err) {
        console.error('Upload route error:', err);

        return res.render('index', {
            result: { error: 'Internal server error during upload.' }
        });
    }
});

module.exports = router;