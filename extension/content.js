/**
 * ScamShield — Content Script
 *
 * Injects a floating overlay widget onto VoIP call pages
 * (Google Meet, Zoom, Teams) and receives analysis results
 * from the background service worker.
 */

(function () {
  if (document.getElementById('scamshield-widget')) return;  // already injected

  // ── Create floating widget ────────────────────────────────────────────────
  const widget = document.createElement('div');
  widget.id = 'scamshield-widget';
  widget.innerHTML = `
    <div id="ss-header">
      <span id="ss-logo">🛡️ ScamShield</span>
      <span id="ss-status">Standby</span>
      <button id="ss-close">✕</button>
    </div>
    <div id="ss-body">
      <div id="ss-gauge-wrap">
        <svg id="ss-gauge-svg" viewBox="0 0 100 58">
          <path d="M10 52 A40 40 0 0 1 90 52" stroke="rgba(255,255,255,0.15)"
                stroke-width="10" fill="none" stroke-linecap="round"/>
          <path id="ss-gauge-fill" d="M10 52 A40 40 0 0 1 90 52"
                stroke="#10b981" stroke-width="10" fill="none"
                stroke-linecap="round"
                stroke-dasharray="0 125.66" stroke-dashoffset="0"/>
          <text id="ss-gauge-pct" x="50" y="50" text-anchor="middle"
                font-size="13" font-weight="bold" fill="#10b981"
                font-family="system-ui,sans-serif">--</text>
        </svg>
        <div id="ss-gauge-label">Risk Score</div>
      </div>
      <div id="ss-verdict">Waiting for audio…</div>
      <div id="ss-transcript"></div>
    </div>
  `;
  document.body.appendChild(widget);

  injectStyles();

  // ── Close button ──────────────────────────────────────────────────────────
  document.getElementById('ss-close').addEventListener('click', () => {
    widget.style.display = 'none';
  });

  // Make widget draggable
  makeDraggable(widget, document.getElementById('ss-header'));

  // ── Listen for results from background ───────────────────────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'SCAM_RESULT') updateWidget(msg.data);
    if (msg.action === 'CAPTURE_STARTED') setStatus('🔴 Live');
    if (msg.action === 'CAPTURE_STOPPED') setStatus('Standby');
  });

  // ── Update widget with analysis result ───────────────────────────────────
  function updateWidget(data) {
    if (!data || data.empty) return;

    const pct      = Math.round((data.confidence || 0) * 100);
    const category = data.category || 'safe';
    const colors   = {
      safe:              '#10b981',
      neutral:           '#9ca3af',
      suspicious:        '#f59e0b',
      highly_suspicious: '#ef4444'
    };
    const labels = {
      safe:              '✅ Safe',
      neutral:           '🔍 Neutral',
      suspicious:        '⚠️ Suspicious',
      highly_suspicious: '🚨 High Risk!'
    };

    const color = colors[category] || '#10b981';
    const circumference = 125.66;
    const fill = (pct / 100) * circumference;

    // Update gauge
    const gaugeFill = document.getElementById('ss-gauge-fill');
    const gaugePct  = document.getElementById('ss-gauge-pct');
    gaugeFill.setAttribute('stroke', color);
    gaugeFill.setAttribute('stroke-dasharray', `${fill} ${circumference}`);
    gaugePct.textContent  = `${pct}%`;
    gaugePct.setAttribute('fill', color);

    // Update verdict
    const verdict = document.getElementById('ss-verdict');
    verdict.textContent  = labels[category] || '✅ Safe';
    verdict.style.color  = color;

    // Update transcript snippet
    const transcript = document.getElementById('ss-transcript');
    if (data.transcription) {
      transcript.textContent = `"${data.transcription.substring(0, 100)}${data.transcription.length > 100 ? '…' : ''}"`;
    }

    // Flash widget red on high risk
    if (category === 'highly_suspicious') {
      widget.style.borderColor = '#ef4444';
      widget.style.boxShadow   = '0 0 20px rgba(239,68,68,0.6)';
      setTimeout(() => {
        widget.style.borderColor = 'rgba(255,255,255,0.15)';
        widget.style.boxShadow   = '0 8px 32px rgba(0,0,0,0.4)';
      }, 3000);
    }
  }

  function setStatus(text) {
    document.getElementById('ss-status').textContent = text;
  }

  // ── Drag support ──────────────────────────────────────────────────────────
  function makeDraggable(el, handle) {
    let ox = 0, oy = 0, mx = 0, my = 0;
    handle.addEventListener('mousedown', e => {
      e.preventDefault();
      mx = e.clientX; my = e.clientY;
      document.addEventListener('mousemove', drag);
      document.addEventListener('mouseup', () => document.removeEventListener('mousemove', drag));
    });
    function drag(e) {
      ox = mx - e.clientX; oy = my - e.clientY;
      mx = e.clientX;      my = e.clientY;
      el.style.top  = (el.offsetTop  - oy) + 'px';
      el.style.left = (el.offsetLeft - ox) + 'px';
      el.style.right = 'auto';
    }
  }

  // ── Styles ────────────────────────────────────────────────────────────────
  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      #scamshield-widget {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 220px;
        background: #1f2937;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        z-index: 2147483647;
        font-family: system-ui, -apple-system, sans-serif;
        color: #f3f4f6;
        font-size: 13px;
        overflow: hidden;
        user-select: none;
      }
      #ss-header {
        background: #111827;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: move;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }
      #ss-logo { font-weight: 700; font-size: 12px; }
      #ss-status {
        font-size: 10px;
        color: #9ca3af;
        background: rgba(255,255,255,0.07);
        padding: 2px 6px;
        border-radius: 8px;
      }
      #ss-close {
        background: none; border: none; color: #6b7280;
        cursor: pointer; font-size: 12px; padding: 0;
      }
      #ss-close:hover { color: #f3f4f6; }
      #ss-body { padding: 10px 12px; }
      #ss-gauge-wrap { display: flex; flex-direction: column; align-items: center; }
      #ss-gauge-svg { width: 120px; }
      #ss-gauge-label {
        font-size: 10px; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: -4px;
      }
      #ss-verdict {
        text-align: center; font-weight: 700; font-size: 13px;
        margin: 8px 0 6px; color: #10b981;
      }
      #ss-transcript {
        font-size: 10px; color: #6b7280; line-height: 1.4;
        font-style: italic; text-align: center;
        min-height: 28px;
      }
    `;
    document.head.appendChild(style);
  }
})();
