const BACKEND_KEY = 'backendUrl';
const API_KEY_KEY = 'apiKey';

const startBtn    = document.getElementById('startBtn');
const stopBtn     = document.getElementById('stopBtn');
const statusDot   = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const backendInput = document.getElementById('backendUrl');
const apiKeyInput  = document.getElementById('apiKey');
const saveBtn      = document.getElementById('saveBtn');
const savedMsg     = document.getElementById('savedMsg');

// ── Load saved settings ───────────────────────────────────────────────────────
chrome.storage.sync.get([BACKEND_KEY, API_KEY_KEY], ({ backendUrl, apiKey }) => {
  backendInput.value = backendUrl || 'http://localhost:5000';
  apiKeyInput.value  = apiKey    || '';
});

// ── Check if already capturing ────────────────────────────────────────────────
chrome.runtime.sendMessage({ action: 'IS_CAPTURING' }, ({ capturing }) => {
  setUI(capturing);
});

// ── Start monitoring ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab) {
    statusLabel.textContent = 'No active tab found';
    return;
  }

  const supported = [
    'meet.google.com',
    'zoom.us',
    'teams.microsoft.com',
    'teams.live.com'
  ];
  const url = new URL(tab.url);
  if (!supported.some(h => url.hostname.includes(h))) {
    statusLabel.textContent = 'Open a Meet/Zoom/Teams call first';
    return;
  }

  statusLabel.textContent = 'Starting…';

  chrome.runtime.sendMessage(
    { action: 'START_CAPTURE', tabId: tab.id },
    (resp) => {
      if (resp?.status === 'capturing') {
        setUI(true);
      } else {
        statusLabel.textContent = 'Failed: ' + (resp?.message || 'unknown error');
      }
    }
  );
});

// ── Stop monitoring ───────────────────────────────────────────────────────────
stopBtn.addEventListener('click', () => {
  chrome.runtime.sendMessage({ action: 'STOP_CAPTURE' }, () => setUI(false));
});

// ── Save settings ─────────────────────────────────────────────────────────────
saveBtn.addEventListener('click', () => {
  chrome.storage.sync.set({
    [BACKEND_KEY]: backendInput.value.trim() || 'http://localhost:5000',
    [API_KEY_KEY]: apiKeyInput.value.trim()
  }, () => {
    savedMsg.style.display = 'block';
    setTimeout(() => { savedMsg.style.display = 'none'; }, 2000);
  });
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function setUI(capturing) {
  if (capturing) {
    startBtn.style.display  = 'none';
    stopBtn.style.display   = 'block';
    statusDot.classList.add('active');
    statusLabel.textContent = 'Monitoring active';
  } else {
    startBtn.style.display  = 'block';
    stopBtn.style.display   = 'none';
    statusDot.classList.remove('active');
    statusLabel.textContent = 'Not monitoring';
  }
}
