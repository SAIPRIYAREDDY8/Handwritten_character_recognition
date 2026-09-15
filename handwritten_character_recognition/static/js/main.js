document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // DOM Element References
    // =========================================================================
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d');
    const canvasHint = document.getElementById('canvasHint');
    const strokeWidthInput = document.getElementById('strokeWidth');
    const btnClearCanvas = document.getElementById('btnClearCanvas');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const btnBrowse = document.getElementById('btnBrowse');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const btnRemoveImage = document.getElementById('btnRemoveImage');

    const btnPredict = document.getElementById('btnPredict');
    const btnReset = document.getElementById('btnReset');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const activeState = document.getElementById('activeState');

    const resCharacter = document.getElementById('resCharacter');
    const resConfidence = document.getElementById('resConfidence');
    const gaugeProgress = document.getElementById('gaugeProgress');
    const candidatesList = document.getElementById('candidatesList');

    const stepOriginal = document.getElementById('stepOriginal');
    const stepGray = document.getElementById('stepGray');
    const stepThresh = document.getElementById('stepThresh');
    const stepRoi = document.getElementById('stepRoi');
    const step28 = document.getElementById('step28');

    const btnMetricsModal = document.getElementById('btnMetricsModal');
    const metricsModal = document.getElementById('metricsModal');
    const btnCloseModal = document.getElementById('btnCloseModal');

    let isDrawing = false;
    let hasDrawn = false;
    let selectedFile = null;
    let activeTab = 'canvas';

    // =========================================================================
    // Canvas Engine Initialization
    // =========================================================================
    function initCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#ffffff';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = strokeWidthInput.value;
    }
    initCanvas();

    strokeWidthInput.addEventListener('input', (e) => {
        ctx.lineWidth = e.target.value;
    });

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        let clientX = e.clientX;
        let clientY = e.clientY;

        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        }

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        hasDrawn = true;
        canvasHint.classList.add('hidden');
        const pos = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        draw(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getCanvasCoordinates(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDrawing() {
        if (isDrawing) {
            ctx.closePath();
            isDrawing = false;
        }
    }

    // Canvas Events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    btnClearCanvas.addEventListener('click', () => {
        initCanvas();
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
    });

    // =========================================================================
    // Tab Navigation Logic
    // =========================================================================
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            activeTab = targetTab;

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            if (targetTab === 'canvas') {
                document.getElementById('tabCanvas').classList.add('active');
            } else {
                document.getElementById('tabUpload').classList.add('active');
            }
        });
    });

    // =========================================================================
    // Drag and Drop File Upload
    // =========================================================================
    btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!selectedFile) fileInput.click();
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (PNG, JPG, JPEG, BMP, WEBP).');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    btnRemoveImage.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
    });

    // =========================================================================
    // Prediction API Call
    // =========================================================================
    btnPredict.addEventListener('click', async () => {
        let payload = null;
        let isFormData = false;

        if (activeTab === 'canvas') {
            if (!hasDrawn) {
                alert('Please draw a character on the canvas first!');
                return;
            }
            const dataUrl = canvas.toDataURL('image/png');
            payload = JSON.stringify({ image: dataUrl });
        } else {
            if (!selectedFile) {
                alert('Please upload an image file first!');
                return;
            }
            const formData = new FormData();
            formData.append('file', selectedFile);
            payload = formData;
            isFormData = true;
        }

        // Set UI to loading state
        emptyState.classList.add('hidden');
        activeState.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const options = {
                method: 'POST',
                body: payload
            };
            if (!isFormData) {
                options.headers = { 'Content-Type': 'application/json' };
            }

            const response = await fetch('/predict', options);
            const data = await response.json();

            loadingState.classList.add('hidden');

            if (!data.success) {
                alert(`Error: ${data.error}`);
                emptyState.classList.remove('hidden');
                return;
            }

            // Display Prediction Results
            displayResults(data);

        } catch (err) {
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
            alert(`Failed to communicate with server: ${err.message}`);
        }
    });

    function displayResults(data) {
        activeState.classList.remove('hidden');

        // Character and confidence
        resCharacter.textContent = data.prediction;
        resConfidence.textContent = `${data.confidence.toFixed(1)}%`;

        // Update Circular SVG Gauge
        const radius = 42;
        const circumference = 2 * Math.PI * radius; // ~263.89
        const offset = circumference - (data.confidence / 100) * circumference;
        gaugeProgress.style.strokeDasharray = `${circumference}`;
        gaugeProgress.style.strokeDashoffset = `${offset}`;

        // Update Top Candidates
        candidatesList.innerHTML = '';
        if (data.top_candidates && data.top_candidates.length > 0) {
            data.top_candidates.forEach(cand => {
                const row = document.createElement('div');
                row.className = 'candidate-row';
                row.innerHTML = `
                    <div class="candidate-char">${cand.character}</div>
                    <div class="candidate-bar-bg">
                        <div class="candidate-bar-fill" style="width: ${cand.confidence}%"></div>
                    </div>
                    <div class="candidate-score">${cand.confidence.toFixed(1)}%</div>
                `;
                candidatesList.appendChild(row);
            });
        }

        // Update OpenCV Preprocessing Steps Gallery
        if (data.preprocessing_steps) {
            stepOriginal.src = data.preprocessing_steps.original;
            stepGray.src = data.preprocessing_steps.grayscale;
            stepThresh.src = data.preprocessing_steps.threshold;
            stepRoi.src = data.preprocessing_steps.roi_crop;
            step28.src = data.preprocessing_steps.processed_28x28;
        }
    }

    // =========================================================================
    // Reset Action
    // =========================================================================
    btnReset.addEventListener('click', () => {
        // Clear canvas
        initCanvas();
        hasDrawn = false;
        canvasHint.classList.remove('hidden');

        // Clear upload
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');

        // Clear result state
        activeState.classList.add('hidden');
        loadingState.classList.add('hidden');
        emptyState.classList.remove('hidden');
    });

    // =========================================================================
    // Model Performance Metrics Modal
    // =========================================================================
    btnMetricsModal.addEventListener('click', async () => {
        metricsModal.classList.remove('hidden');
        fetchMetrics();
    });

    btnCloseModal.addEventListener('click', () => {
        metricsModal.classList.add('hidden');
    });

    metricsModal.addEventListener('click', (e) => {
        if (e.target === metricsModal) {
            metricsModal.classList.add('hidden');
        }
    });

    async function fetchMetrics() {
        try {
            const res = await fetch('/metrics');
            const data = await res.json();
            if (data.success && data.metrics) {
                const m = data.metrics;
                document.getElementById('metricAccuracy').textContent = `${(m.accuracy * 100).toFixed(1)}%`;
                document.getElementById('metricPrecision').textContent = `${(m.precision_weighted * 100).toFixed(1)}%`;
                document.getElementById('metricRecall').textContent = `${(m.recall_weighted * 100).toFixed(1)}%`;
                document.getElementById('metricF1').textContent = `${(m.f1_score_weighted * 100).toFixed(1)}%`;
                document.getElementById('reportText').textContent = m.classification_report_text || 'Classification report ready.';
            }
        } catch (e) {
            console.error('Error fetching metrics:', e);
        }
    }
});
