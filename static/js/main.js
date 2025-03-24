document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    feather.replace();

    // DOM Elements - Single Mode
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

    // DOM Elements - Mode Switching
    const singleModeBtn = document.getElementById('singleModeBtn');
    const continuousModeBtn = document.getElementById('continuousModeBtn');
    const singleModeContent = document.getElementById('singleModeContent');
    const continuousModeContent = document.getElementById('continuousModeContent');

    // DOM Elements - Continuous Mode
    const startMonitoringBtn = document.getElementById('startMonitoringBtn');
    const stopMonitoringBtn = document.getElementById('stopMonitoringBtn');
    const monitoringStatusPanel = document.querySelector('.monitoring-status-panel');
    const liveTranscriptOutput = document.getElementById('liveTranscriptOutput');
    const monitoringTime = document.getElementById('monitoringTime');

    // DOM Elements - Alert Modal
    const scamAlertModal = new bootstrap.Modal(document.getElementById('scamAlertModal'));
    const alertTranscript = document.getElementById('alertTranscript');
    const endCallBtn = document.getElementById('endCallBtn');

    // Initialize the AudioRecorder
    const audioRecorder = new AudioRecorder();
    let currentAudioBlob = null;
    let fileSelected = false;

    // Continuous monitoring variables
    let isMonitoring = false;
    let monitoringRecorder = new AudioRecorder();
    let monitoringInterval = null;
    let monitoringTimeCounter = 0;
    let monitoringTimer = null;
    let lastTranscriptions = [];
    let currentChunk = null;

    // Event Listeners - Single Mode
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    audioFileInput.addEventListener('change', handleFileUpload);
    analyzeButton.addEventListener('click', analyzeAudio);
    clearAudioBtn.addEventListener('click', clearAudio);

    // Event Listeners - Mode Switching
    singleModeBtn.addEventListener('click', switchToSingleMode);
    continuousModeBtn.addEventListener('click', switchToContinuousMode);

    // Event Listeners - Continuous Mode
    startMonitoringBtn.addEventListener('click', startMonitoring);
    stopMonitoringBtn.addEventListener('click', stopMonitoring);
    endCallBtn.addEventListener('click', stopMonitoring);

    /**
     * Switch to Single Analysis Mode
     */
    function switchToSingleMode() {
        // Only switch if not already in this mode
        if (singleModeBtn.classList.contains('active')) return;
        
        // Stop monitoring if active
        if (isMonitoring) {
            stopMonitoring();
        }
        
        // Update UI
        singleModeBtn.classList.add('active');
        continuousModeBtn.classList.remove('active');
        singleModeContent.classList.add('active');
        continuousModeContent.classList.remove('active');
    }

    /**
     * Switch to Continuous Monitoring Mode
     */
    function switchToContinuousMode() {
        // Only switch if not already in this mode
        if (continuousModeBtn.classList.contains('active')) return;
        
        // Update UI
        continuousModeBtn.classList.add('active');
        singleModeBtn.classList.remove('active');
        continuousModeContent.classList.add('active');
        singleModeContent.classList.remove('active');
    }

    /**
     * Start continuous monitoring
     */
    async function startMonitoring() {
        try {
            // Clear UI
            liveTranscriptOutput.innerHTML = '<p class="placeholder-text">Listening...</p>';
            errorMessage.style.display = 'none';
            
            // Start recording
            await monitoringRecorder.startRecording();
            
            // Update UI
            isMonitoring = true;
            startMonitoringBtn.style.display = 'none';
            stopMonitoringBtn.style.display = 'block';
            monitoringStatusPanel.style.display = 'block';
            
            // Initialize monitoring time
            monitoringTimeCounter = 0;
            updateMonitoringTime();
            monitoringTimer = setInterval(updateMonitoringTime, 1000);
            
            // Start the monitoring cycle
            monitoringInterval = setInterval(captureAndAnalyzeChunk, 5000);
            
        } catch (error) {
            showError('Microphone access denied. Please allow microphone access and try again.');
            console.error('Monitoring error:', error);
        }
    }

    /**
     * Stop continuous monitoring
     */
    function stopMonitoring() {
        if (!isMonitoring) return;
        
        // Clear intervals
        clearInterval(monitoringInterval);
        clearInterval(monitoringTimer);
        
        // Stop recording
        monitoringRecorder.cancelRecording();
        
        // Update UI
        isMonitoring = false;
        startMonitoringBtn.style.display = 'block';
        stopMonitoringBtn.style.display = 'none';
        monitoringStatusPanel.style.display = 'none';
        
        // Reset variables
        lastTranscriptions = [];
        monitoringTimeCounter = 0;
    }

    /**
     * Capture audio chunk and analyze it
     */
    async function captureAndAnalyzeChunk() {
        try {
            // Pause current recording
            const currentRecording = await monitoringRecorder.stopRecording();
            
            // If we have a valid recording
            if (currentRecording && currentRecording.size > 0) {
                // Send to server for analysis
                analyzeAudioChunk(currentRecording);
            }
            
            // Restart recording for next chunk
            await monitoringRecorder.startRecording();
            
        } catch (error) {
            console.error('Error capturing audio chunk:', error);
            // Continue monitoring even if one chunk fails
        }
    }

    /**
     * Analyze audio chunk for monitoring
     * @param {Blob} audioBlob The audio blob to analyze
     */
    function analyzeAudioChunk(audioBlob) {
        // Create form data for the API request
        const formData = new FormData();
        formData.append('audio', audioBlob, 'chunk.webm');
        
        // Send the API request
        fetch('/analyze-chunk', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // If the chunk has content
            if (!data.empty) {
                // Add to transcription history
                updateLiveTranscript(data.transcription);
                
                // Check if it's a scam
                if (data.is_spam) {
                    showScamAlert(data);
                }
            }
        })
        .catch(error => {
            console.error('Error analyzing audio chunk:', error);
        });
    }

    /**
     * Update the live transcript with new text
     * @param {string} text New transcription text
     */
    function updateLiveTranscript(text) {
        // Add to transcript history (keep last 5)
        lastTranscriptions.push(text);
        if (lastTranscriptions.length > 5) {
            lastTranscriptions.shift();
        }
        
        // Update the display
        liveTranscriptOutput.innerHTML = lastTranscriptions.map(t => `<p>${t}</p>`).join('');
        
        // Scroll to bottom
        liveTranscriptOutput.scrollTop = liveTranscriptOutput.scrollHeight;
    }

    /**
     * Show scam alert modal
     * @param {Object} data Analysis data
     */
    function showScamAlert(data) {
        // Get category and severity information
        const category = data.category || 'suspicious';
        const severity = data.severity || 5;
        
        // Update alert content
        alertTranscript.textContent = data.transcription;
        
        // Select the appropriate alert class based on category and severity
        const alertModalDialog = document.querySelector('#scamAlertModal .modal-dialog');
        alertModalDialog.className = 'modal-dialog'; // Reset class
        
        // Add the appropriate class based on category
        if (category === 'highly_suspicious') {
            alertModalDialog.classList.add('modal-danger');
            // Add pulse animation for high severity
            if (severity >= 8) {
                document.querySelector('#scamAlertModal .modal-content').style.animation = 'pulse-red 1s infinite';
            }
        } else if (category === 'suspicious') {
            alertModalDialog.classList.add('modal-warning');
        } else {
            alertModalDialog.classList.add('modal-info');
        }
        
        // Create message based on severity
        let alertTitle = document.querySelector('#scamAlertModal .modal-title');
        if (severity >= 8) {
            alertTitle.textContent = '🚨 HIGH RISK SCAM DETECTED!';
        } else if (severity >= 6) {
            alertTitle.textContent = '⚠️ SUSPICIOUS ACTIVITY DETECTED';
        } else {
            alertTitle.textContent = '🔍 POTENTIAL SCAM WARNING';
        }
        
        // Show the modal
        scamAlertModal.show();
    }

    /**
     * Update monitoring time display
     */
    function updateMonitoringTime() {
        monitoringTimeCounter++;
        const minutes = Math.floor(monitoringTimeCounter / 60).toString().padStart(2, '0');
        const seconds = (monitoringTimeCounter % 60).toString().padStart(2, '0');
        monitoringTime.textContent = `${minutes}:${seconds}`;
    }

    /**
     * Start audio recording (single mode)
     */
    async function startRecording() {
        try {
            clearUI();
            await audioRecorder.startRecording();
            recordingStatus.style.display = 'flex';
            recordButton.disabled = true;
            audioFileInput.disabled = true;
        } catch (error) {
            showError('Microphone access denied. Please allow microphone access and try again.');
            console.error('Recording error:', error);
        }
    }

    /**
     * Stop audio recording and update UI (single mode)
     */
    async function stopRecording() {
        try {
            const audioBlob = await audioRecorder.stopRecording();
            currentAudioBlob = audioBlob;
            
            // Update UI
            recordingStatus.style.display = 'none';
            recordButton.disabled = false;
            audioFileInput.disabled = false;
            clearAudioBtn.style.display = 'block';
            
            // Display audio preview
            displayAudioPreview(audioBlob, 'Recording');
            
            // Enable analyze button
            analyzeButton.disabled = false;
        } catch (error) {
            showError('Error stopping recording. Please try again.');
            console.error('Error stopping recording:', error);
        }
    }

    /**
     * Handle file upload (single mode)
     * @param {Event} event The change event
     */
    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Check if the file is an audio file
        if (!file.type.startsWith('audio/')) {
            showError('Please upload an audio file.');
            return;
        }
        
        clearUI();
        
        // Create a blob from the file
        currentAudioBlob = file;
        fileSelected = true;
        clearAudioBtn.style.display = 'block';
        
        // Display audio preview
        displayAudioPreview(file, file.name);
        
        // Enable analyze button
        analyzeButton.disabled = false;
    }

    /**
     * Display audio preview (single mode)
     * @param {Blob} audioBlob The audio blob
     * @param {string} source The source name
     */
    function displayAudioPreview(audioBlob, source) {
        const audioUrl = URL.createObjectURL(audioBlob);
        audioPlayer.src = audioUrl;
        audioPreview.style.display = 'block';
        
        // Display file info
        const sizeInKB = Math.round(audioBlob.size / 1024);
        fileInfo.textContent = `${source} • ${sizeInKB} KB`;
    }

    /**
     * Analyze the audio using the server API (single mode)
     */
    function analyzeAudio() {
        if (!currentAudioBlob) {
            showError('No audio to analyze. Please record or upload audio first.');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        analyzeButton.disabled = true;
        clearOutputs();
        
        // Create form data for the API request
        const formData = new FormData();
        
        // Add the file extension if it's a recording
        let filename = 'recorded_audio.webm';
        if (fileSelected && currentAudioBlob.name) {
            filename = currentAudioBlob.name;
        }
        
        formData.append('audio', currentAudioBlob, filename);
        
        // Send the API request
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            showError('Error analyzing audio: ' + error.message);
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
            analyzeButton.disabled = false;
        });
    }

    /**
     * Display analysis results (single mode)
     * @param {Object} data The result data
     */
    function displayResults(data) {
        // Display transcription
        transcriptionOutput.innerHTML = `<p>${data.transcription}</p>`;
        
        // Display prediction based on category and severity
        const isPotentialScam = data.is_spam;
        const category = data.category || (isPotentialScam ? 'suspicious' : 'safe');
        const severity = data.severity || 0;
        
        // Set appropriate CSS class based on category
        let predictionClass = 'prediction-safe';
        if (category === 'highly_suspicious') {
            predictionClass = 'prediction-high-risk';
        } else if (category === 'suspicious') {
            predictionClass = 'prediction-scam';
        } else if (category === 'neutral') {
            predictionClass = 'prediction-neutral';
        }
        
        const confidencePercent = Math.round(data.confidence * 100);
        
        // Create HTML for any matched hotwords if present
        let matchesHtml = '';
        if (data.matches && data.matches.length > 0) {
            matchesHtml = `
                <div class="matches-container">
                    <h6>Detected Warning Phrases:</h6>
                    <ul class="matches-list">
                        ${data.matches.map(match => 
                            `<li>${match.hotword} <span class="severity-badge">${match.severity}/10</span></li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        } else if (data.matched_patterns && data.matched_patterns.length > 0) {
            matchesHtml = `
                <div class="matches-container">
                    <h6>Detected Patterns:</h6>
                    <ul class="matches-list">
                        ${data.matched_patterns.map(pattern => 
                            `<li>${pattern}</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Add a visual severity indicator (more intense for higher severity)
        const severityStyle = severity > 7 ? 
            'animation: pulse-red 1.5s infinite;' : 
            (severity > 5 ? 'animation: pulse-yellow 2s infinite;' : '');
        
        predictionOutput.innerHTML = `
            <div class="${predictionClass}" style="${severityStyle}">
                <span>${data.prediction}</span>
                ${severity > 0 ? `<div class="severity-meter">Severity: ${severity}/10</div>` : ''}
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${isPotentialScam ? 'scam' : 'safe'}" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${confidencePercent}%</div>
            ${matchesHtml}
        `;
    }

    /**
     * Clear UI state (single mode)
     */
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

    /**
     * Clear output displays (single mode)
     */
    function clearOutputs() {
        transcriptionOutput.innerHTML = '<div class="placeholder-text">Transcription will appear here...</div>';
        predictionOutput.innerHTML = '<div class="placeholder-text">Analysis result will appear here...</div>';
    }

    /**
     * Display an error message
     * @param {string} message The error message
     */
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }

    /**
     * Clear audio and reset the UI (single mode)
     */
    function clearAudio() {
        clearUI();
        recordButton.disabled = false;
        audioFileInput.disabled = false;
    }

    // Check if the browser supports required APIs
    function checkBrowserSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Your browser does not support audio recording. Please try a modern browser like Chrome, Firefox, or Edge.');
            recordButton.disabled = true;
            startMonitoringBtn.disabled = true;
        }
    }

    // Initialize
    checkBrowserSupport();
});
