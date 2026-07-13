/**
 * AI Cattle Analysis System — Frontend Logic
 * Handles file upload, API communication, and dynamic UI rendering.
 */
document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ──
    const dropZone = document.getElementById('dropZone');
    const imageInput = document.getElementById('imageInput');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    const uploadSection = document.getElementById('uploadSection');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImg');
    const previewName = document.getElementById('previewName');
    const removeBtn = document.getElementById('removeBtn');
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');

    let selectedFile = null;

    // ═══════════════════════════════════════════════
    //  FILE HANDLING
    // ═══════════════════════════════════════════════

    const handleFile = (file) => {
        if (!file || !file.type.startsWith('image/')) return;
        selectedFile = file;
        analyzeBtn.disabled = false;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewName.textContent = file.name;
            uploadPreview.classList.remove('hidden');
            dropZone.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    };

    dropZone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            handleFile(e.dataTransfer.files[0]);
        }
    });

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        imageInput.value = '';
        analyzeBtn.disabled = true;
        uploadPreview.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    newAnalysisBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        selectedFile = null;
        imageInput.value = '';
        analyzeBtn.disabled = true;
        uploadPreview.classList.add('hidden');
        dropZone.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // ═══════════════════════════════════════════════
    //  FORM SUBMISSION
    // ═══════════════════════════════════════════════

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = imageInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (response.ok && data.success) {
                renderResults(data.results, data.annotated_image);
                uploadSection.classList.add('hidden');
                resultsSection.classList.remove('hidden');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                showError(data.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            showError('Failed to connect to the server. Please check if the server is running.');
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });

    // ═══════════════════════════════════════════════
    //  RENDER RESULTS
    // ═══════════════════════════════════════════════

    function renderResults(res, annotatedImage) {
        // ── Annotated Image ──
        document.getElementById('resultImage').src = `/outputs/${annotatedImage}?t=${Date.now()}`;

        // ── Detection Confidence ──
        const detConf = res.confidence || (res.detection && res.detection.confidence) || 0;
        document.getElementById('detConfFill').style.width = `${detConf}%`;
        document.getElementById('detConfValue').textContent = `${detConf}%`;

        // ── Species ──
        document.getElementById('resSpecies').textContent = res.species || 'Unknown';
        const speciesConf = res.species_confidence || 0;
        document.getElementById('resSpeciesConf').textContent = speciesConf > 0 ? `${speciesConf}%` : '';

        // ── Scientific Name ──
        const sciName = res.taxonomy ? res.taxonomy.scientific_name : (res.scientific_name || '—');
        document.getElementById('resScientific').textContent = sciName;

        // ── Breed ──
        document.getElementById('resBreed').textContent = res.breed || 'Unknown';
        const breedConf = res.breed_confidence || 0;
        document.getElementById('resBreedConf').textContent = breedConf > 0 ? `${breedConf.toFixed(1)}%` : '';

        // ── Breed Top-5 ──
        renderBreedTop5(res.breed_top_5 || []);

        // ── Weight ──
        const weight = res.weight_kg || res.xgb_weight_kg || 0;
        document.getElementById('resWeight').textContent = weight.toFixed(0);
        document.getElementById('resWeightRange').textContent = res.weight_range || (res.breed_info ? res.breed_info.weight_range : '—');
        const weightMethod = res.weight_method || '';
        document.getElementById('resWeightMethod').textContent =
            weightMethod === 'ml_regressor' ? 'ML-based estimation' :
            weightMethod === 'breed_average' ? 'Breed average estimation' : '';

        // ── BCS ──
        const bcs = res.bcs || res.xgb_bcs || 3.0;
        document.getElementById('resBcs').textContent = bcs.toFixed(1);
        const bcsPercent = ((bcs - 1) / 4) * 100;
        document.getElementById('bcsMarker').style.left = `${bcsPercent}%`;
        document.getElementById('resBcsDesc').textContent = getBCSDescription(bcs);

        // Set BCS color
        const bcsEl = document.getElementById('resBcs');
        if (bcs < 2) bcsEl.style.color = '#ef4444';
        else if (bcs < 2.5) bcsEl.style.color = '#f59e0b';
        else if (bcs <= 3.5) bcsEl.style.color = '#10b981';
        else if (bcs <= 4) bcsEl.style.color = '#f59e0b';
        else bcsEl.style.color = '#ef4444';

        // ── Taxonomy ──
        if (res.taxonomy) {
            document.getElementById('taxKingdom').textContent = res.taxonomy.kingdom || 'Animalia';
            document.getElementById('taxPhylum').textContent = res.taxonomy.phylum || 'Chordata';
            document.getElementById('taxClass').textContent = res.taxonomy.class || 'Mammalia';
            document.getElementById('taxOrder').textContent = res.taxonomy.order || '—';
            document.getElementById('taxFamily').textContent = res.taxonomy.family || '—';
            document.getElementById('taxSpecies').textContent = res.taxonomy.scientific_name || '—';
        } else {
            document.getElementById('taxKingdom').textContent = 'Animalia';
            document.getElementById('taxPhylum').textContent = 'Chordata';
            document.getElementById('taxClass').textContent = 'Mammalia';
            document.getElementById('taxOrder').textContent = res.order || '—';
            document.getElementById('taxFamily').textContent = res.family || '—';
            document.getElementById('taxSpecies').textContent = res.scientific_name || '—';
        }

        // ── Breed Info Card ──
        renderBreedInfo(res.breed_info);

        // ── Measurements ──
        renderMeasurements(res.measurements);

        // ── Image Quality Warning ──
        renderQualityWarning(res.image_quality);
    }

    // ═══════════════════════════════════════════════
    //  COMPONENT RENDERERS
    // ═══════════════════════════════════════════════

    function renderBreedTop5(breeds) {
        const container = document.getElementById('breedTop5');
        container.innerHTML = '';
        if (!breeds || breeds.length === 0) return;

        breeds.slice(0, 5).forEach((b) => {
            const conf = b.confidence || 0;
            const row = document.createElement('div');
            row.className = 'breed-bar-row';
            row.innerHTML = `
                <span class="breed-bar-label" title="${b.breed}">${b.breed}</span>
                <div class="breed-bar-track">
                    <div class="breed-bar-fill" style="width: 0%"></div>
                </div>
                <span class="breed-bar-pct">${conf.toFixed(1)}%</span>
            `;
            container.appendChild(row);

            // Animate bar
            requestAnimationFrame(() => {
                const fill = row.querySelector('.breed-bar-fill');
                fill.style.width = `${Math.min(conf, 100)}%`;
            });
        });
    }

    function renderBreedInfo(info) {
        const card = document.getElementById('cardBreedInfo');
        if (!info) {
            card.classList.add('hidden');
            return;
        }
        card.classList.remove('hidden');

        document.getElementById('breedDescription').textContent = info.description || '—';

        const grid = document.getElementById('breedDetailsGrid');
        grid.innerHTML = '';

        const fields = [
            { label: 'Origin', value: info.origin_country },
            { label: 'Purpose', value: info.purpose },
            { label: 'Coat Colors', value: Array.isArray(info.coat_colors) ? info.coat_colors.join(', ') : info.coat_colors },
            { label: 'Weight Range', value: info.weight_range },
            { label: 'Height Range', value: info.height_range },
            { label: 'Milk Yield', value: info.avg_milk_yield },
            { label: 'Temperament', value: info.temperament },
            { label: 'Climate', value: info.climate_adaptability },
            { label: 'Lifespan', value: info.lifespan },
            { label: 'Horns', value: info.horn_status },
        ];

        fields.forEach((f) => {
            if (!f.value || f.value === 'Unknown') return;
            const item = document.createElement('div');
            item.className = 'breed-detail-item';
            item.innerHTML = `
                <span class="breed-detail-label">${f.label}</span>
                <span class="breed-detail-value">${f.value}</span>
            `;
            grid.appendChild(item);
        });
    }

    function renderMeasurements(measurements) {
        const grid = document.getElementById('measurementsGrid');
        grid.innerHTML = '';
        if (!measurements) return;

        const items = [
            { label: 'Body Area', value: measurements.body_area_px, unit: 'px²' },
            { label: 'Perimeter', value: measurements.body_perimeter_px, unit: 'px' },
            { label: 'Body Length', value: measurements.body_length_px, unit: 'px', decimals: 0 },
            { label: 'Body Height', value: measurements.body_height_px, unit: 'px', decimals: 0 },
            { label: 'Aspect Ratio', value: measurements.aspect_ratio, unit: '', decimals: 2 },
            { label: 'Solidity', value: measurements.solidity, unit: '', decimals: 3 },
        ];

        items.forEach((m) => {
            if (m.value === undefined || m.value === null) return;
            const val = typeof m.value === 'number'
                ? (m.decimals !== undefined ? m.value.toFixed(m.decimals) : Math.round(m.value))
                : m.value;
            const item = document.createElement('div');
            item.className = 'measurement-item';
            item.innerHTML = `
                <span class="measurement-value">${val}${m.unit ? ' ' + m.unit : ''}</span>
                <span class="measurement-label">${m.label}</span>
            `;
            grid.appendChild(item);
        });
    }

    function renderQualityWarning(quality) {
        const warning = document.getElementById('qualityWarning');
        if (!quality || !quality.warnings || quality.warnings.length === 0) {
            warning.classList.add('hidden');
            return;
        }
        warning.classList.remove('hidden');
        document.getElementById('qualityWarningText').textContent = quality.warnings.join(' ');
    }

    // ═══════════════════════════════════════════════
    //  HELPERS
    // ═══════════════════════════════════════════════

    function getBCSDescription(bcs) {
        if (bcs <= 1.5) return 'Emaciated — Severe underconditioning. Immediate nutritional intervention recommended.';
        if (bcs <= 2.0) return 'Thin — Below optimal condition. Increase feed intake recommended.';
        if (bcs <= 2.5) return 'Moderate-Thin — Slightly below ideal. Minor feed adjustments may help.';
        if (bcs <= 3.0) return 'Moderate — Good body condition. Animal appears healthy and well-maintained.';
        if (bcs <= 3.5) return 'Moderate-Fleshy — Ideal to slightly above optimal condition.';
        if (bcs <= 4.0) return 'Fleshy — Above optimal. Consider reducing energy intake.';
        if (bcs <= 4.5) return 'Fat — Overconditioned. Reduce feed to prevent health issues.';
        return 'Obese — Severely overconditioned. Health risk. Immediate dietary adjustment needed.';
    }

    function showError(message) {
        alert(`Analysis Error: ${message}`);
    }
});
