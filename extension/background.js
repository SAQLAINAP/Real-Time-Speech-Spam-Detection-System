/**
 * ScamShield — Background Service Worker
 *
 * Handles:
 *  - Tab audio capture via chrome.tabCapture
 *  - Chunking audio every CHUNK_INTERVAL ms
 *  - Sending chunks to the ScamShield backend
 *  - Pushing results back to the content script overlay
 */

const CHUNK_INTERVAL_MS = 5000;   // analyse every 5 seconds
const BACKEND_URL_KEY   = 'backendUrl';
const API_KEY_KEY       = 'apiKey';
const DEFAULT_BACKEND   = 'http://localhost:5000';

let mediaRecorder   = null;
let captureStream   = null;
let chunkTimer      = null;
let audioChunks     = [];
let activeTabId     = null;
let accumulatedText = '';          // rolling context for smarter detection

// ── Listen for messages from popup / content script ──────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.action) {
    case 'START_CAPTURE':
      startCapture(msg.tabId).then(sendResponse);
      return true;   // async response

    case 'STOP_CAPTURE':
      stopCapture();
      sendResponse({ status: 'stopped' });
      break;

    case 'IS_CAPTURING':
      sendResponse({ capturing: mediaRecorder !== null });
      break;
  }
});

// ── Start tab audio capture ───────────────────────────────────────────────────
async function startCapture(tabId) {
  if (mediaRecorder) return { status: 'already_capturing' };

  try {
    activeTabId = tabId;
    accumulatedText = '';

    captureStream = await new Promise((resolve, reject) => {
      chrome.tabCapture.capture({ audio: true, video: false }, stream => {
        if (chrome.runtime.lastError || !stream) {
          reject(chrome.runtime.lastError?.message || 'Capture failed');
        } else {
          resolve(stream);
        }
      });
    });

    mediaRecorder = new MediaRecorder(captureStream, { mimeType: 'audio/webm' });

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.start();
    scheduleChunk();

    return { status: 'capturing' };

  } catch (err) {
    console.error('[ScamShield] Capture error:', err);
    return { status: 'error', message: String(err) };
  }
}

// ── Schedule periodic chunk upload ────────────────────────────────────────────
function scheduleChunk() {
  chunkTimer = setInterval(async () => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

    // Pause → collect → restart
    mediaRecorder.stop();
    await new Promise(r => setTimeout(r, 100));

    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    audioChunks = [];

    mediaRecorder.start();

    if (blob.size > 1000) {
      const result = await sendChunk(blob);
      if (result) notifyContentScript(result);
    }
  }, CHUNK_INTERVAL_MS);
}

// ── Send audio chunk to backend ───────────────────────────────────────────────
async function sendChunk(blob) {
  const { [BACKEND_URL_KEY]: backendUrl = DEFAULT_BACKEND, [API_KEY_KEY]: apiKey = '' } =
    await chrome.storage.sync.get([BACKEND_URL_KEY, API_KEY_KEY]);

  const formData = new FormData();
  formData.append('audio', blob, 'chunk.webm');
  if (accumulatedText) formData.append('context', accumulatedText);

  try {
    const headers = {};
    if (apiKey) headers['X-API-Key'] = apiKey;

    const resp = await fetch(`${backendUrl}/analyze-chunk`, {
      method: 'POST',
      headers,
      body: formData
    });

    if (!resp.ok) return null;
    const data = await resp.json();

    if (!data.empty && data.transcription) {
      // Accumulate rolling context (last ~500 chars)
      accumulatedText = (accumulatedText + ' ' + data.transcription).slice(-500);
    }

    return data;

  } catch (err) {
    console.error('[ScamShield] Backend error:', err);
    return null;
  }
}

// ── Push result to content script overlay ────────────────────────────────────
function notifyContentScript(data) {
  if (!activeTabId || data.empty) return;

  chrome.tabs.sendMessage(activeTabId, {
    action: 'SCAM_RESULT',
    data
  }).catch(() => {});   // tab may have navigated away

  // Also show a Chrome notification for high-risk results
  if (data.category === 'highly_suspicious') {
    chrome.notifications.create({
      type:    'basic',
      iconUrl: 'icons/icon48.png',
      title:   '🚨 ScamShield — High Risk Detected!',
      message: data.transcription?.substring(0, 100) || 'Scam language detected in this call.'
    });
  }
}

// ── Stop capture ──────────────────────────────────────────────────────────────
function stopCapture() {
  clearInterval(chunkTimer);
  chunkTimer = null;

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  mediaRecorder = null;

  if (captureStream) {
    captureStream.getTracks().forEach(t => t.stop());
    captureStream = null;
  }

  audioChunks = [];
  activeTabId = null;
  accumulatedText = '';
}
