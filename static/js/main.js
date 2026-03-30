document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    feather.replace();

    // ── Demo sample scripts ──────────────────────────────────────────────────
    const DEMO_SAMPLES = {
        irs: "This is a final notice from the IRS. Legal action against you will be filed if you don't respond immediately. Your Social Security number has been suspended due to fraudulent activity. You owe back taxes and must wire transfer the full payment today to avoid arrest. This is urgent, act now or face criminal charges.",
        bank: "Your account has been suspended due to suspicious activity. Please verify your account by confirming your bank account details and credit card number immediately. Urgent response needed — call now to avoid permanent loss of access and a hold on all your funds.",
        lottery: "Congratulations, you've won the international lottery! You have been selected as our winner notification recipient for a prize of fifty thousand dollars. To claim your reward, you must confirm your identity and wire transfer a small processing fee within 24 hours. This is a one-time offer — act now before it expires!",
        safe: "Hi, this is Sarah from customer support. I'm following up on your recent service request. Everything looks good on our end and your account is in good standing. There's nothing you need to do at this time. Please feel free to call us back at our official number if you have any questions. Have a great day!"
    };

    // ── DOM Elements — Single Mode ───────────────────────────────────────────
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const audioFileInput = document.getElementById('audioFileInput');
    const audioPlayer = document.getElementById('audioPlayer');
    const audioPreview = document.getElementById('audioPreview');
    const fileInfo = document.getElementById('fileInfo');
    const analyzeButton = document.getElementById('analyzeButton');
    const transcriptionOutput = document.getElementById('transcriptionOutput');
    const predictionOutput = document.getElementById('predictionOutput');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessage = document.getElementById('errorMessage');
    const recordingStatus = document.querySelector('.recording-status');
    const clearAudioBtn = document.getElementById('clearAudio');

    // ── DOM Elements — Mode Switching ────────────────────────────────────────
    const singleModeBtn = document.getElementById('singleModeBtn');
    const continuousModeBtn = document.getElementById('continuousModeBtn');
    const singleModeContent = document.getElementById('singleModeContent');
    const continuousModeContent = document.getElementById('continuousModeContent');

    // ── DOM Elements — Continuous Mode ──────────────────────────────────────
    const startMonitoringBtn = document.getElementById('startMonitoringBtn');
    const stopMonitoringBtn = document.getElementById('stopMonitoringBtn');
    const monitoringStatusPanel = document.querySelector('.monitoring-status-panel');
    const liveTranscriptOutput = document.getElementById('liveTranscriptOutput');
    const monitoringTime = document.getElementById('monitoringTime');

    // ── DOM Elements — Alert Modal ───────────────────────────────────────────
    const scamAlertModal = new bootstrap.Modal(document.getElementById('scamAlertModal'));
    const alertTranscript = document.getElementById('alertTranscript');
    const endCallBtn = document.getElementById('endCallBtn');

    // ── DOM Elements — History ───────────────────────────────────────────────
    const historyPanel = document.getElementById('historyPanel');
    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    // ── State ────────────────────────────────────────────────────────────────
    const audioRecorder = new AudioRecorder();
    let currentAudioBlob = null;
    let fileSelected = false;

    let isMonitoring = false;
    let monitoringRecorder = new AudioRecorder();
    let monitoringInterval = null;
    let monitoringTimeCounter = 0;
    let monitoringTimer = null;
    let lastTranscriptions = [];

    let analysisHistory = [];   // 1D: session history

    // ── Event Listeners ──────────────────────────────────────────────────────
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    audioFileInput.addEventListener('change', handleFileUpload);
    analyzeButton.addEventListener('click', analyzeAudio);
    clearAudioBtn.addEventListener('click', clearAudio);

    singleModeBtn.addEventListener('click', switchToSingleMode);
    continuousModeBtn.addEventListener('click', switchToContinuousMode);

    startMonitoringBtn.addEventListener('click', startMonitoring);
    stopMonitoringBtn.addEventListener('click', stopMonitoring);
    endCallBtn.addEventListener('click', stopMonitoring);

    clearHistoryBtn.addEventListener('click', () => {
        analysisHistory = [];
        renderHistory();
    });

    // 1A: Demo sample buttons
    document.querySelectorAll('.demo-btn').forEach(btn => {
        btn.addEventListener('click', () => runDemoSample(btn.dataset.sample));
    });

    // ── Mode Switching ───────────────────────────────────────────────────────
    function switchToSingleMode() {
        if (singleModeBtn.classList.contains('active')) return;
        if (isMonitoring) stopMonitoring();
        singleModeBtn.classList.add('active');
        continuousModeBtn.classList.remove('active');
        singleModeContent.classList.add('active');
        continuousModeContent.classList.remove('active');
    }

    function switchToContinuousMode() {
        if (continuousModeBtn.classList.contains('active')) return;
        continuousModeBtn.classList.add('active');
        singleModeBtn.classList.remove('active');
        continuousModeContent.classList.add('active');
        singleModeContent.classList.remove('active');
    }

    // ── Continuous Monitoring ────────────────────────────────────────────────
    async function startMonitoring() {
        try {
            liveTranscriptOutput.innerHTML = '<p class="placeholder-text">Listening...</p>';
            errorMessage.style.display = 'none';
            await monitoringRecorder.startRecording();
            isMonitoring = true;
            startMonitoringBtn.style.display = 'none';
            stopMonitoringBtn.style.display = 'block';
            monitoringStatusPanel.style.display = 'block';
            monitoringTimeCounter = 0;
            updateMonitoringTime();
            monitoringTimer = setInterval(updateMonitoringTime, 1000);
            monitoringInterval = setInterval(captureAndAnalyzeChunk, 5000);
        } catch (error) {
            showError('Microphone access denied. Please allow microphone access and try again.');
        }
    }

    function stopMonitoring() {
        if (!isMonitoring) return;
        clearInterval(monitoringInterval);
        clearInterval(monitoringTimer);
        monitoringRecorder.cancelRecording();
        isMonitoring = false;
        startMonitoringBtn.style.display = 'block';
        stopMonitoringBtn.style.display = 'none';
        monitoringStatusPanel.style.display = 'none';
        lastTranscriptions = [];
        monitoringTimeCounter = 0;
    }

    async function captureAndAnalyzeChunk() {
        try {
            const currentRecording = await monitoringRecorder.stopRecording();
            if (currentRecording && currentRecording.size > 0) {
                analyzeAudioChunk(currentRecording);
            }
            await monitoringRecorder.startRecording();
        } catch (error) {
            console.error('Error capturing audio chunk:', error);
        }
    }

    function analyzeAudioChunk(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'chunk.webm');
        // 2C: send accumulated transcript as context for smarter detection
        if (lastTranscriptions.length > 0) {
            formData.append('context', lastTranscriptions.join(' '));
        }
        fetch('/analyze-chunk', { method: 'POST', body: formData })
            .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
            .then(data => {
                if (data.error) throw new Error(data.error);
                if (!data.empty) {
                    updateLiveTranscript(data.transcription);
                    if (data.is_spam) showScamAlert(data);
                }
            })
            .catch(err => console.error('Chunk analysis error:', err));
    }

    function updateLiveTranscript(text) {
        lastTranscriptions.push(text);
        if (lastTranscriptions.length > 5) lastTranscriptions.shift();
        liveTranscriptOutput.innerHTML = lastTranscriptions.map(t => `<p>${t}</p>`).join('');
        liveTranscriptOutput.scrollTop = liveTranscriptOutput.scrollHeight;
    }

    function showScamAlert(data) {
        const category = data.category || 'suspicious';
        const severity = data.severity || 5;
        alertTranscript.textContent = data.transcription;
        const alertModalDialog = document.querySelector('#scamAlertModal .modal-dialog');
        alertModalDialog.className = 'modal-dialog modal-dialog-centered';
        if (category === 'highly_suspicious') {
            alertModalDialog.classList.add('modal-danger');
            if (severity >= 8) {
                document.querySelector('#scamAlertModal .modal-content').style.animation = 'pulse-red 1s infinite';
            }
        } else if (category === 'suspicious') {
            alertModalDialog.classList.add('modal-warning');
        } else {
            alertModalDialog.classList.add('modal-info');
        }
        const alertTitle = document.querySelector('#scamAlertModal .modal-title');
        if (severity >= 8) alertTitle.textContent = '🚨 HIGH RISK SCAM DETECTED!';
        else if (severity >= 6) alertTitle.textContent = '⚠️ SUSPICIOUS ACTIVITY DETECTED';
        else alertTitle.textContent = '🔍 POTENTIAL SCAM WARNING';
        scamAlertModal.show();
    }

    function updateMonitoringTime() {
        monitoringTimeCounter++;
        const m = Math.floor(monitoringTimeCounter / 60).toString().padStart(2, '0');
        const s = (monitoringTimeCounter % 60).toString().padStart(2, '0');
        monitoringTime.textContent = `${m}:${s}`;
    }

    // ── Single Mode Recording ────────────────────────────────────────────────
    async function startRecording() {
        try {
            clearUI();
            await audioRecorder.startRecording();
            recordingStatus.style.display = 'flex';
            recordButton.disabled = true;
            audioFileInput.disabled = true;
        } catch (error) {
            showError('Microphone access denied. Please allow microphone access and try again.');
        }
    }

    async function stopRecording() {
        try {
            const audioBlob = await audioRecorder.stopRecording();
            currentAudioBlob = audioBlob;
            recordingStatus.style.display = 'none';
            recordButton.disabled = false;
            audioFileInput.disabled = false;
            clearAudioBtn.style.display = 'block';
            displayAudioPreview(audioBlob, 'Recording');
            analyzeButton.disabled = false;
        } catch (error) {
            showError('Error stopping recording. Please try again.');
        }
    }

    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        if (!file.type.startsWith('audio/')) {
            showError('Please upload an audio file.');
            return;
        }
        clearUI();
        currentAudioBlob = file;
        fileSelected = true;
        clearAudioBtn.style.display = 'block';
        displayAudioPreview(file, file.name);
        analyzeButton.disabled = false;
    }

    function displayAudioPreview(audioBlob, source) {
        const audioUrl = URL.createObjectURL(audioBlob);
        audioPlayer.src = audioUrl;
        audioPreview.style.display = 'block';
        const sizeInKB = Math.round(audioBlob.size / 1024);
        fileInfo.textContent = `${source} • ${sizeInKB} KB`;
    }

    // ── 1A: Demo Sample Handler ──────────────────────────────────────────────
    function runDemoSample(sampleKey) {
        const text = DEMO_SAMPLES[sampleKey];
        if (!text) return;

        clearOutputs();
        loadingIndicator.style.display = 'flex';
        errorMessage.style.display = 'none';

        fetch('/analyze-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        })
        .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
        .then(data => {
            if (data.error) throw new Error(data.error);
            displayResults(data);
        })
        .catch(err => showError('Demo analysis failed: ' + err))
        .finally(() => { loadingIndicator.style.display = 'none'; });
    }

    // ── Single Mode Analysis ─────────────────────────────────────────────────
    function analyzeAudio() {
        if (!currentAudioBlob) {
            showError('No audio to analyze. Please record or upload audio first.');
            return;
        }
        loadingIndicator.style.display = 'flex';
        analyzeButton.disabled = true;
        clearOutputs();

        const formData = new FormData();
        let filename = 'recorded_audio.webm';
        if (fileSelected && currentAudioBlob.name) filename = currentAudioBlob.name;
        formData.append('audio', currentAudioBlob, filename);

        fetch('/analyze', { method: 'POST', body: formData })
            .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
            .then(data => {
                if (data.error) throw new Error(data.error);
                displayResults(data);
            })
            .catch(err => showError('Error analyzing audio: ' + err))
            .finally(() => {
                loadingIndicator.style.display = 'none';
                analyzeButton.disabled = false;
            });
    }

    // ── 1B: Phrase Highlighting ──────────────────────────────────────────────
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function highlightTranscript(text, matches, matchedPatterns) {
        if (!text) return '';
        let result = text;

        if (matches && matches.length > 0) {
            // Sort by severity desc so higher-risk highlights take precedence
            const sorted = [...matches].sort((a, b) => b.severity - a.severity);
            sorted.forEach(match => {
                const cls = match.severity >= 8 ? 'hl-high' : match.severity >= 5 ? 'hl-med' : 'hl-low';
                const regex = new RegExp(`(${escapeRegex(match.hotword)})`, 'gi');
                result = result.replace(regex, `<mark class="${cls}">$1</mark>`);
            });
        } else if (matchedPatterns && matchedPatterns.length > 0) {
            matchedPatterns.forEach(pattern => {
                const regex = new RegExp(`(${escapeRegex(pattern)})`, 'gi');
                result = result.replace(regex, `<mark class="hl-med">$1</mark>`);
            });
        }

        return result;
    }

    // ── 1C: Confidence Gauge ─────────────────────────────────────────────────
    function buildGauge(percent, isSpam) {
        // Semi-circular SVG gauge
        const radius = 40;
        const circumference = Math.PI * radius; // ~125.66
        const fill = (percent / 100) * circumference;

        let strokeColor;
        if (percent >= 70) strokeColor = '#ef4444';
        else if (percent >= 40) strokeColor = '#f59e0b';
        else strokeColor = '#10b981';

        const labelText = isSpam ? 'Risk Score' : 'Safety Score';
        const labelClass = isSpam ? 'gauge-label-risk' : 'gauge-label-safe';

        return `
        <div class="confidence-gauge">
            <svg viewBox="0 0 100 58" class="gauge-svg" aria-label="${percent}% confidence">
                <defs>
                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" style="stop-color:#10b981"/>
                        <stop offset="50%" style="stop-color:#f59e0b"/>
                        <stop offset="100%" style="stop-color:#ef4444"/>
                    </linearGradient>
                </defs>
                <!-- Track -->
                <path d="M 10 52 A 40 40 0 0 1 90 52"
                      stroke="rgba(255,255,255,0.1)" stroke-width="10" fill="none"
                      stroke-linecap="round"/>
                <!-- Fill -->
                <path d="M 10 52 A 40 40 0 0 1 90 52"
                      stroke="${strokeColor}" stroke-width="10" fill="none"
                      stroke-linecap="round"
                      stroke-dasharray="${fill} ${circumference}"
                      stroke-dashoffset="0"/>
                <!-- Percentage text -->
                <text x="50" y="50" text-anchor="middle"
                      font-size="13" font-weight="bold" fill="${strokeColor}"
                      font-family="-apple-system, BlinkMacSystemFont, sans-serif">${percent}%</text>
            </svg>
            <div class="gauge-label ${labelClass}">${labelText}</div>
        </div>`;
    }

    // ── 1D: History Log ───────────────────────────────────────────────────────
    function addToHistory(data) {
        const now = new Date();
        const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const snippet = data.transcription.length > 90
            ? data.transcription.substring(0, 90) + '…'
            : data.transcription;

        analysisHistory.unshift({
            time,
            category: data.category || 'safe',
            confidence: Math.round(data.confidence * 100),
            is_spam: data.is_spam,
            snippet
        });

        // Cap at 20 entries
        if (analysisHistory.length > 20) analysisHistory.pop();
        renderHistory();
    }

    function renderHistory() {
        if (analysisHistory.length === 0) {
            historyPanel.style.display = 'none';
            return;
        }

        historyPanel.style.display = 'block';
        historyList.innerHTML = analysisHistory.map(entry => {
            const badgeClass = {
                highly_suspicious: 'badge-high-risk',
                suspicious: 'badge-suspicious',
                neutral: 'badge-neutral',
                safe: 'badge-safe'
            }[entry.category] || 'badge-safe';

            const badgeLabel = {
                highly_suspicious: 'HIGH RISK',
                suspicious: 'SUSPICIOUS',
                neutral: 'NEUTRAL',
                safe: 'SAFE'
            }[entry.category] || 'SAFE';

            return `
            <div class="history-entry">
                <div class="history-meta">
                    <span class="history-time">${entry.time}</span>
                    <span class="history-badge ${badgeClass}">${badgeLabel}</span>
                    <span class="history-confidence">${entry.confidence}% confidence</span>
                </div>
                <div class="history-snippet">${entry.snippet}</div>
            </div>`;
        }).join('');
    }

    // ── Display Results (1B + 1C integrated) ─────────────────────────────────
    function displayResults(data) {
        // 1B + 3C — Highlighted transcription with language badge
        const highlighted = highlightTranscript(data.transcription, data.matches, data.matched_patterns);
        const langBadge = (data.language && data.language !== 'en')
            ? `<span class="lang-badge">${data.language_name || data.language.toUpperCase()}</span>`
            : '';
        transcriptionOutput.innerHTML = `${langBadge}<p>${highlighted}</p>`;

        const isPotentialScam = data.is_spam;
        const category = data.category || (isPotentialScam ? 'suspicious' : 'safe');
        const severity = data.severity || 0;
        const confidencePercent = Math.round(data.confidence * 100);

        let predictionClass = 'prediction-safe';
        if (category === 'highly_suspicious') predictionClass = 'prediction-high-risk';
        else if (category === 'suspicious') predictionClass = 'prediction-scam';
        else if (category === 'neutral') predictionClass = 'prediction-neutral';

        const severityStyle = severity > 7
            ? 'animation: pulse-red 1.5s infinite;'
            : severity > 5 ? 'animation: pulse-yellow 2s infinite;' : '';

        // Matched phrases list
        let matchesHtml = '';
        if (data.matches && data.matches.length > 0) {
            matchesHtml = `
            <div class="matches-container">
                <h6>Detected Warning Phrases:</h6>
                <ul class="matches-list">
                    ${data.matches.map(m =>
                        `<li>${m.hotword} <span class="severity-badge">${m.severity}/10</span></li>`
                    ).join('')}
                </ul>
            </div>`;
        } else if (data.matched_patterns && data.matched_patterns.length > 0) {
            matchesHtml = `
            <div class="matches-container">
                <h6>Detected Patterns:</h6>
                <ul class="matches-list">
                    ${data.matched_patterns.map(p => `<li>${p}</li>`).join('')}
                </ul>
            </div>`;
        }

        // 1C — Gauge + prediction label + matches
        predictionOutput.innerHTML = `
            ${buildGauge(confidencePercent, isPotentialScam)}
            <div class="${predictionClass} mt-2" style="${severityStyle}">
                <span>${data.prediction}</span>
                ${severity > 0 ? `<div class="severity-meter">Severity: ${severity}/10</div>` : ''}
            </div>
            ${matchesHtml}
        `;

        // 1D — Append to history
        addToHistory(data);
    }

    // ── Utilities ─────────────────────────────────────────────────────────────
    function clearUI() {
        clearOutputs();
        errorMessage.style.display = 'none';
        audioPreview.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioPlayer.src = '';
        fileInfo.textContent = '';
        audioFileInput.value = '';
        currentAudioBlob = null;
        fileSelected = false;
        analyzeButton.disabled = true;
        clearAudioBtn.style.display = 'none';
    }

    function clearOutputs() {
        transcriptionOutput.innerHTML = '<div class="placeholder-text">Transcription will appear here...</div>';
        predictionOutput.innerHTML = '<div class="placeholder-text">Analysis result will appear here...</div>';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => { errorMessage.style.display = 'none'; }, 5000);
    }

    function clearAudio() {
        clearUI();
        recordButton.disabled = false;
        audioFileInput.disabled = false;
    }

    function checkBrowserSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.');
            recordButton.disabled = true;
            startMonitoringBtn.disabled = true;
        }
    }

    checkBrowserSupport();
});
