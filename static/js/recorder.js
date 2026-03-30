/**
 * AudioRecorder - A wrapper class for MediaRecorder API
 * Provides functionality to record, stop, and retrieve audio
 */
class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioBlob = null;
        this.audioStream = null;
        this.isRecording = false;
    }

    /**
     * Start recording audio from the user's microphone
     * @returns {Promise} Resolves when recording starts
     */
    async startRecording() {
        try {
            this.audioChunks = [];
            
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Create a new MediaRecorder instance
            this.mediaRecorder = new MediaRecorder(this.audioStream);
            
            // Add event listener for data available event
            this.mediaRecorder.addEventListener('dataavailable', event => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            });
            
            // Start recording
            this.mediaRecorder.start();
            this.isRecording = true;
            
            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            throw error;
        }
    }

    /**
     * Stop the ongoing recording
     * @returns {Promise} Resolves with the recorded audio blob
     */
    stopRecording() {
        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
                reject(new Error('No active recording to stop.'));
                return;
            }

            // Add event listener for when recording stops
            this.mediaRecorder.addEventListener('stop', () => {
                // Create a Blob from the recorded chunks
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                
                // Stop all tracks in the stream
                if (this.audioStream) {
                    this.audioStream.getTracks().forEach(track => track.stop());
                }
                
                this.isRecording = false;
                resolve(this.audioBlob);
            }, { once: true });

            // Stop the recording
            this.mediaRecorder.stop();
        });
    }

    /**
     * Cancel the current recording
     */
    cancelRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
        
        this.audioChunks = [];
        this.audioBlob = null;
        this.isRecording = false;
    }

    /**
     * Get the current audio blob
     * @returns {Blob|null} The recorded audio blob or null
     */
    getAudioBlob() {
        return this.audioBlob;
    }

    /**
     * Check if recording is currently active
     * @returns {Boolean} True if recording is active
     */
    getIsRecording() {
        return this.isRecording;
    }

    /**
     * Create a URL for the recorded audio blob
     * @returns {string|null} URL for the audio blob or null
     */
    getAudioURL() {
        if (!this.audioBlob) return null;
        return URL.createObjectURL(this.audioBlob);
    }

    /**
     * Get the file extension for the recorded audio
     * @returns {string} The file extension
     */
    getFileExtension() {
        if (!this.audioBlob) return '';
        const mimeType = this.audioBlob.type;
        
        switch (mimeType) {
            case 'audio/webm':
                return 'webm';
            case 'audio/mp4':
                return 'm4a';
            case 'audio/ogg':
                return 'ogg';
            default:
                return 'webm';  // Default to webm
        }
    }

    /**
     * Get information about the recorded audio file
     * @returns {object|null} Information about the audio file or null
     */
    getFileInfo() {
        if (!this.audioBlob) return null;
        
        const sizeInKB = Math.round(this.audioBlob.size / 1024);
        const extension = this.getFileExtension();
        
        return {
            size: sizeInKB,
            type: this.audioBlob.type,
            extension: extension
        };
    }
}
