document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    feather.replace();

    // DOM Elements
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

    // Initialize the AudioRecorder
    const audioRecorder = new AudioRecorder();
    let currentAudioBlob = null;
    let fileSelected = false;

    // Event Listeners
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);
    audioFileInput.addEventListener('change', handleFileUpload);
    analyzeButton.addEventListener('click', analyzeAudio);
    clearAudioBtn.addEventListener('click', clearAudio);

    /**
     * Start audio recording
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
     * Stop audio recording and update UI
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
     * Handle file upload
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
     * Display audio preview
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
     * Analyze the audio using the server API
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
     * Display analysis results
     * @param {Object} data The result data
     */
    function displayResults(data) {
        // Display transcription
        transcriptionOutput.innerHTML = `<p>${data.transcription}</p>`;
        
        // Display prediction
        const isPotentialScam = data.is_spam;
        const predictionClass = isPotentialScam ? 'prediction-scam' : 'prediction-safe';
        const confidencePercent = Math.round(data.confidence * 100);
        
        predictionOutput.innerHTML = `
            <div class="${predictionClass}">
                <span>${data.prediction}</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill ${isPotentialScam ? 'scam' : 'safe'}" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${confidencePercent}%</div>
        `;
    }

    /**
     * Clear UI state
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
     * Clear output displays
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
     * Clear audio and reset the UI
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
        }
    }

    // Initialize
    checkBrowserSupport();
});
