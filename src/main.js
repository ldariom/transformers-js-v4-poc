import { pipeline, env, ModelRegistry } from '@huggingface/transformers';

env.useWasmCache = true;
env.logLevel = 'none';

const modelId = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
const imageCaptionModelId = "Xenova/vit-gpt2-image-captioning";
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const generatingOverlay = document.getElementById('generating');
const canvas = document.getElementById('app-canvas');
const ctx = canvas.getContext('2d');
const canvasContent = document.getElementById('canvas-content');

let generator = null;
let imageCaptioner = null;
let currentPhase = 'init';
let loginHTMLCode = null;
let dashboardHTMLCode = null;
let dpr = window.devicePixelRatio || 1;
let particles = [];
let useDrawElement = typeof ctx.drawElementImage === 'function';
let paintRetryCount = 0;
let transformApplied = false;
const MAX_PAINT_RETRIES = 5;

// Screen capture history: {dataURL, caption, html, timestamp}
const screenHistory = [];
let autoCaptureInterval = null;
const AUTO_CAPTURE_MS = 10000;
let isAutoCapturing = false;

// Known valid CSS classes for sanitization
const VALID_CLASSES = new Set([
  'login-card','login-title','login-subtitle','login-label','login-input','login-btn',
  'dashboard-container','dashboard-header','dashboard-title','dashboard-subtitle',
  'dashboard-logout','dashboard-grid','dashboard-card','dashboard-card-label',
  'dashboard-card-value','dashboard-panel','dashboard-panel-title','dashboard-table',
  'badge','badge-progress','badge-planning','badge-done'
]);

// Inline styles for the canvas subtree so drawElementImage captures them
// (Chrome's experimental implementation doesn't resolve <head> styles for layoutsubtree snapshots)
const CANVAS_SUBTREE_CSS = `
#canvas-content {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  overflow: visible;
  font-family: 'Segoe UI', system-ui, sans-serif;
}

/* === LOGIN === */
.login-card {
  background: linear-gradient(145deg, #12122a 0%, #1a1a3a 100%);
  border-radius: 20px; padding: 48px;
  width: 420px; max-width: 90vw;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 80px rgba(108,92,231,0.08);
  border: 1px solid rgba(108,92,231,0.15);
}
.login-title {
  color: #fff; margin: 0 0 6px;
  font-size: 26px; font-weight: 700;
  text-align: center; letter-spacing: -0.5px;
}
.login-subtitle {
  color: #6b6b8a; margin: 0 0 36px;
  font-size: 13.5px; text-align: center;
}
.login-label {
  color: #8a8aaa; font-size: 12px;
  display: block; margin-bottom: 6px;
  font-weight: 500; text-transform: uppercase;
  letter-spacing: 0.3px;
}
.login-input {
  width: 100%; padding: 14px 16px;
  background: rgba(10,10,21,0.8);
  border: 1.5px solid #2a2a55;
  border-radius: 10px; color: #fff;
  font-size: 14px; margin-bottom: 20px;
  outline: none; transition: all 0.2s ease;
}
.login-input::placeholder { color: #4a4a6a; }
.login-input:hover { border-color: #3a3a6a; }
.login-input:focus {
  border-color: #6c5ce7;
  box-shadow: 0 0 0 3px rgba(108,92,231,0.15);
}
.login-btn {
  width: 100%; padding: 16px;
  background: linear-gradient(135deg, #6c5ce7 0%, #8e7cf5 100%);
  color: #fff; border: none;
  border-radius: 10px; font-size: 15px;
  font-weight: 600; cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 16px rgba(108,92,231,0.3);
}
.login-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(108,92,231,0.4);
}
.login-btn:active { transform: translateY(0); }

/* === DASHBOARD === */
.dashboard-container {
  width: 100%; max-width: 1100px;
  margin: 0 auto; padding: 28px;
}
.dashboard-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 20px 28px;
  background: linear-gradient(145deg, #12122a 0%, #1a1a3a 100%);
  border-radius: 16px; margin-bottom: 28px;
  border: 1px solid rgba(108,92,231,0.1);
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.dashboard-title {
  color: #fff; font-size: 20px;
  margin: 0; font-weight: 700;
}
.dashboard-subtitle {
  color: #6b6b8a; font-size: 12px;
  margin: 4px 0 0;
}
.dashboard-logout {
  padding: 10px 24px;
  background: rgba(255,107,107,0.1);
  color: #ff6b6b; border: 1px solid rgba(255,107,107,0.2);
  border-radius: 8px; cursor: pointer;
  font-size: 13px; font-weight: 500;
  transition: all 0.2s ease;
}
.dashboard-logout:hover {
  background: rgba(255,107,107,0.2);
  border-color: rgba(255,107,107,0.35);
}
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px; margin-bottom: 28px;
}
.dashboard-card {
  background: linear-gradient(145deg, #12122a 0%, #1a1a3a 100%);
  border-radius: 16px; padding: 28px;
  border: 1px solid rgba(108,92,231,0.08);
  box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  transition: all 0.2s ease;
}
.dashboard-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  border-color: rgba(108,92,231,0.15);
}
.dashboard-card-label {
  color: #8a8aaa; font-size: 12.5px;
  margin: 0 0 6px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.dashboard-card-value {
  color: #fff; font-size: 32px;
  font-weight: 700; margin: 0;
  letter-spacing: -0.5px;
}
.dashboard-panel {
  background: linear-gradient(145deg, #12122a 0%, #1a1a3a 100%);
  border-radius: 16px; padding: 28px;
  border: 1px solid rgba(108,92,231,0.08);
  box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.dashboard-panel-title {
  color: #fff; font-size: 16px;
  margin: 0 0 20px; font-weight: 600;
}
.dashboard-table { width: 100%; border-collapse: collapse; }
.dashboard-table th {
  color: #8a8aaa; font-size: 11.5px;
  text-align: left; padding: 14px 10px;
  border-bottom: 1.5px solid #1e1e3a;
  font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.3px;
}
.dashboard-table td {
  color: #c0c0d0; padding: 14px 10px;
  font-size: 13.5px;
  border-bottom: 1px solid rgba(30,30,58,0.5);
}
.dashboard-table tr:last-child td { border-bottom: none; }
.badge {
  font-size: 11px; padding: 4px 12px;
  border-radius: 20px; font-weight: 500;
  display: inline-block;
}
.badge-progress { background: rgba(33,150,243,0.12); color: #64b5f6; }
.badge-planning { background: rgba(255,152,0,0.12); color: #ffb74d; }
.badge-done { background: rgba(76,175,80,0.12); color: #81c784; }
`;

// ResizeObserver to sync canvas grid size with device pixel ratio (per WICG spec)
const resizeObserver = new ResizeObserver(([entry]) => {
  const size = entry.devicePixelContentBoxSize?.[0];
  if (size) {
    canvas.width = size.inlineSize;
    canvas.height = size.blockSize;
    dpr = window.devicePixelRatio;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
});
try {
  resizeObserver.observe(canvas, { box: 'device-pixel-content-box' });
} catch (e) {
  // Fallback if device-pixel-content-box is not supported
  function fallbackResize() {
    dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', fallbackResize);
  fallbackResize();
}

function initParticles() {
  particles = [];
  const c = Math.floor((window.innerWidth * window.innerHeight) / 22000);
  for (let i = 0; i < c; i++) {
    particles.push({
      x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5, alpha: Math.random() * 0.25 + 0.08
    });
  }
}

function drawBackground() {
  const w = window.innerWidth, h = window.innerHeight;
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, w, h);
  for (const p of particles) {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = w; if (p.x > w) p.x = 0;
    if (p.y < 0) p.y = h; if (p.y > h) p.y = 0;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(108,92,231,${p.alpha})`; ctx.fill();
  }
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 150) {
        ctx.strokeStyle = `rgba(108,92,231,${0.04 * (1 - d / 150)})`;
        ctx.lineWidth = 0.5; ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y); ctx.stroke();
      }
    }
  }
}

// WICG paint event: snapshot is recorded just prior to this event.
// drawElementImage returns a transform that maps the element's paint
// snapshot to canvas coordinates. We must apply it to the DOM element
// ONCE so that hit-testing aligns with the painted image.
// Re-applying it every frame causes drift because each snapshot is
// captured from the already-displaced position.
if (useDrawElement) {
  canvas.onpaint = () => {
    ctx.resetTransform();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBackground();
    if (canvasContent.innerHTML.trim()) {
      try {
        const transform = ctx.drawElementImage(canvasContent, 0, 0);
        paintRetryCount = 0;
        // Apply transform to DOM only once per HTML injection to avoid drift
        if (transform && !transformApplied) {
          canvasContent.style.transform = transform.toString();
          transformApplied = true;
        }
      } catch (e) {
        paintRetryCount++;
        if (paintRetryCount <= MAX_PAINT_RETRIES) {
          console.warn(`[drawElementImage] snapshot not ready (retry ${paintRetryCount}/${MAX_PAINT_RETRIES}):`, e.message || e);
          requestAnimationFrame(() => {
            try { canvas.requestPaint(); } catch (_) {}
          });
        } else {
          console.error('[drawElementImage] Max retries reached. Giving up.');
          paintRetryCount = 0;
        }
      }
    }
  };
}

function animate() {
  if (!useDrawElement) {
    drawBackground();
  }
  requestAnimationFrame(animate);
}

function sanitizeClasses(html) {
  // Replace unknown classes with empty string, keep only VALID_CLASSES
  return html.replace(/class\s*=\s*"([^"]*)"/gi, (match, classes) => {
    const valid = classes.split(/\s+/).filter(c => VALID_CLASSES.has(c));
    if (valid.length === 0) return '';
    return `class="${valid.join(' ')}"`;
  });
}

function cleanHTML(raw) {
  if (!raw) return '';
  let html = raw.replace(/```html\s*/gi, '').replace(/```\s*$/gi, '').replace(/```/g, '');
  // Extract body content if present
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  if (bodyMatch) html = bodyMatch[1];
  html = html.replace(/<!DOCTYPE[^>]*>/gi, '');
  html = html.replace(/<html[^>]*>([\s\S]*)<\/html>/i, (m, content) => {
    const b = content.match(/<body[^>]*>([\s\S]*)<\/body>/i);
    return b ? b[1] : content;
  });
  html = html.replace(/<head[^>]*>[\s\S]*?<\/head>/gi, '');
  html = html.replace(/<body[^>]*>/gi, '').replace(/<\/body>/gi, '');
  html = html.replace(/<\s*meta\s+[^>]*>/gi, '');
  html = html.replace(/<\s*link\s+[^>]*>/gi, '');
  html = html.replace(/<\s*script\s*[^>]*>[\s\S]*?<\/\s*script\s*>/gi, '');
  html = html.replace(/<\s*title\s*[^>]*>[\s\S]*?<\/\s*title\s*>/gi, '');
  // Keep inline <style> blocks inside the fragment (they may define dynamic colors)
  // Only remove external <style> tags that were erroneously generated
  // html = html.replace(/<\s*style\s*[^>]*>[\s\S]*?<\/\s*style\s*>/gi, '');
  // Remove HTML comments
  html = html.replace(/<!--[\s\S]*?-->/g, '');
  html = html.replace(/\s+onclick\s*=\s*["'][^"']*["']/gi, '');
  html = html.replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"').replace(/\u2013/g, '-').replace(/\u2014/g, '--');
  html = html.trim();
  const firstTag = html.search(/<\w+/);
  if (firstTag > 0) html = html.slice(firstTag);
  // Sanitize unknown CSS classes
  html = sanitizeClasses(html);
  return html;
}

async function generateHTML(prompt, maxTokens = 1200) {
  const sys = `You are a HTML UI generator. Output ONLY HTML. No markdown, no explanations.

Rules:
- Use ONLY these CSS classes:
  Login: login-card, login-title, login-subtitle, login-label, login-input, login-btn
  Dashboard: dashboard-container, dashboard-header, dashboard-title, dashboard-subtitle, dashboard-logout, dashboard-grid, dashboard-card, dashboard-card-label, dashboard-card-value, dashboard-panel, dashboard-panel-title, dashboard-table, badge, badge-progress, badge-planning, badge-done
- You MAY use style="background:COLOR; color:COLOR" for color changes ONLY
- NO <html>, <body>, <head>, <meta>, <script>, <link>, <title>
- NO onclick attributes
- IDs exact: loginForm, username, password, loginBtn, dashboardView, logoutBtn

LOGIN structure:
<div id="loginForm" class="login-card">
  <h1 class="login-title">Bienvenido</h1>
  <p class="login-subtitle">...</p>
  <label class="login-label">Usuario</label>
  <input id="username" type="text" placeholder="..." class="login-input">
  <label class="login-label">Contrasena</label>
  <input id="password" type="password" placeholder="..." class="login-input">
  <button id="loginBtn" class="login-btn">Acceder</button>
</div>

DASHBOARD structure:
<div id="dashboardView" class="dashboard-container">
  <div class="dashboard-header">...</div>
  <div class="dashboard-grid">...</div>
  <div class="dashboard-panel">...</div>
</div>`;

  const textInput = `<|im_start|>system\n${sys}<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
  const output = await generator(textInput, { max_new_tokens: maxTokens, temperature: 0.3, top_p: 0.9 });
  const generated = output[0].generated_text;
  const newText = generated.slice(textInput.length);
  console.log('[AI RAW length]:', newText.length);
  console.log('[AI RAW preview]:', newText.substring(0, 300));
  const cleaned = cleanHTML(newText);
  console.log('[AI CLEAN length]:', cleaned.length);
  console.log('[AI CLEAN preview]:', cleaned.substring(0, 300));
  return cleaned;
}

function forceStyleComputation(root) {
  // Force the browser to compute styles for all elements in the subtree.
  // This is needed because layoutsubtree elements may be skipped by the
  // style engine until explicitly queried.
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  while (walker.nextNode()) {
    try { getComputedStyle(walker.currentNode); } catch (_) {}
  }
}

function ensureCentering() {
  // layoutsubtree may not apply <head> styles to the live DOM for hit-testing.
  // We force critical layout styles directly on the DOM elements so that
  // hit-testing aligns with the painted snapshot.
  canvasContent.style.position = 'absolute';
  canvasContent.style.top = '0';
  canvasContent.style.left = '0';
  canvasContent.style.width = '100%';
  canvasContent.style.height = '100%';
  canvasContent.style.display = 'flex';
  canvasContent.style.justifyContent = 'center';

  const loginForm = document.getElementById('loginForm');
  if (loginForm) {
    loginForm.style.margin = 'auto';
  }
  const dashboardView = document.getElementById('dashboardView');
  if (dashboardView) {
    dashboardView.style.margin = '0 auto';
    dashboardView.style.width = '100%';
    dashboardView.style.maxWidth = '1100px';
  }
}

function injectHTML(html, verticalAlign = 'center') {
  if (!html || !html.trim()) return;
  // Reset transform state for new content
  transformApplied = false;
  canvasContent.style.transform = '';
  // Prepend a <style> block with the subtree CSS so drawElementImage
  // captures the styles even when the browser doesn't resolve <head> styles.
  const styledHTML = `<style>${CANVAS_SUBTREE_CSS}</style>${html}`;
  canvasContent.innerHTML = styledHTML;
  // Apply critical inline styles so the live DOM layout matches the snapshot.
  // layoutsubtree may ignore <head> styles for hit-testing purposes.
  ensureCentering();
  // Adjust vertical alignment: login centered, dashboard starts from top
  canvasContent.style.alignItems = verticalAlign === 'center' ? 'center' : 'flex-start';
  if (useDrawElement) {
    // Force layout/style resolution before requesting paint.
    forceStyleComputation(canvasContent);
    // Double rAF ensures the browser has processed styles & layout
    // before we ask for a paint snapshot.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        try { canvas.requestPaint(); } catch (e) {}
      });
    });
  }
  requestAnimationFrame(() => bindCurrentEvents());
}

function bindCurrentEvents() {
  if (currentPhase === 'login') bindLoginEvents();
  else if (currentPhase === 'dashboard') bindDashboardEvents();
}

function bindLoginEvents() {
  const btn = document.getElementById('loginBtn');
  const pw = document.getElementById('password');
  if (btn) btn.onclick = handleLogin;
  if (pw) pw.onkeydown = e => { if (e.key === 'Enter') handleLogin(); };
}

function bindDashboardEvents() {
  const btn = document.getElementById('logoutBtn');
  if (btn) btn.onclick = handleLogout;
}

function showGenerating(on) { generatingOverlay.classList.toggle('visible', on); }

// ===== BOTTOM PANEL INTERACTIONS =====
const panelImages = document.getElementById('panel-images');
const panelTextPreview = document.getElementById('panel-text-preview');
const panelChatInput = document.getElementById('panel-chat-input');
const panelChatSend = document.getElementById('panel-chat-send');

// ===== SCREEN CAPTURE & AI ANALYSIS =====

function dataURLToImage(dataURL) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataURL;
  });
}

async function describeScreenshot(dataURL) {
  // Always combine VLM caption with computed style snapshot for rich context
  const styleInfo = extractStyleSnapshot();
  let visualCaption = '';

  if (imageCaptioner) {
    try {
      const img = await dataURLToImage(dataURL);
      const output = await imageCaptioner(img);
      visualCaption = output[0]?.generated_text || '';
    } catch (e) {
      console.warn('[describeScreenshot] VLM error:', e.message);
    }
  }

  // Combine both sources
  const combined = visualCaption
    ? `${visualCaption}. ESTILO ACTUAL: ${styleInfo.desc}`
    : `Interfaz web. ESTILO ACTUAL: ${styleInfo.desc}`;
  return combined;
}

function rgbToHex(rgb) {
  if (!rgb || rgb === 'rgba(0, 0, 0, 0)') return 'transparent';
  const match = rgb.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return rgb;
  return '#' + [match[1], match[2], match[3]].map(x => {
    const hex = parseInt(x).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
}

function extractStyleSnapshot() {
  // Rich style extraction from the current visible UI for the prompt context
  const loginForm = document.getElementById('loginForm');
  const dashboardView = document.getElementById('dashboardView');
  const target = loginForm || dashboardView;
  if (!target) return { type: 'unknown', desc: 'interfaz desconocida' };

  const cs = getComputedStyle(target);
  const btn = target.querySelector('button');
  const btnCS = btn ? getComputedStyle(btn) : null;
  const inputs = target.querySelectorAll('input');
  const inpCS = inputs[0] ? getComputedStyle(inputs[0]) : null;

  const snapshot = {
    type: loginForm ? 'login' : 'dashboard',
    bg: rgbToHex(cs.backgroundColor) || rgbToHex(cs.backgroundImage) || '#12122a',
    textColor: rgbToHex(cs.color) || '#fff',
    borderRadius: cs.borderRadius || '16px',
    padding: cs.padding || '40px',
    fontFamily: cs.fontFamily || 'system-ui',
    button: btnCS ? {
      bg: rgbToHex(btnCS.backgroundColor) || '#6c5ce7',
      text: rgbToHex(btnCS.color) || '#fff',
      radius: btnCS.borderRadius || '8px'
    } : null,
    input: inpCS ? {
      bg: rgbToHex(inpCS.backgroundColor) || '#0a0a15',
      border: rgbToHex(inpCS.borderColor) || '#2a2a4a',
      text: rgbToHex(inpCS.color) || '#fff'
    } : null
  };

  let desc = `${snapshot.type === 'login' ? 'Formulario de login' : 'Dashboard'} con `;
  desc += `fondo ${snapshot.bg}, texto ${snapshot.textColor}, `;
  desc += `bordes redondeados ${snapshot.borderRadius}, `;
  desc += `fuente ${snapshot.fontFamily}. `;
  if (snapshot.button) {
    desc += `Boton: fondo ${snapshot.button.bg}, texto ${snapshot.button.text}, radio ${snapshot.button.radius}. `;
  }
  if (snapshot.input) {
    desc += `Input: fondo ${snapshot.input.bg}, borde ${snapshot.input.border}, texto ${snapshot.input.text}.`;
  }

  return { type: snapshot.type, desc, snapshot };
}

async function captureAndAnalyze() {
  if (!generator) return;
  try {
    // Capture canvas screenshot
    const dataURL = canvas.toDataURL('image/png');

    // Get AI description of the screenshot
    const caption = await describeScreenshot(dataURL);
    console.log('[Capture] Caption:', caption);

    // Store in history
    const entry = {
      dataURL,
      caption,
      html: canvasContent.innerHTML,
      timestamp: Date.now()
    };
    screenHistory.push(entry);

    // Update thumbnails panel
    renderThumbnails();

    // Show latest caption in preview
    if (panelTextPreview) {
      panelTextPreview.textContent = `[Analisis #${screenHistory.length}]: ${caption}`;
    }

    console.log('[Capture] Screenshot saved. Total:', screenHistory.length);
  } catch (e) {
    console.error('[captureAndAnalyze] Error:', e);
  }
}

function renderThumbnails() {
  if (!panelImages) return;
  panelImages.innerHTML = '';
  screenHistory.forEach((entry, idx) => {
    const thumb = document.createElement('div');
    thumb.className = 'img-thumb has-image';
    thumb.innerHTML = `<img src="${entry.dataURL}" alt="Capture ${idx + 1}">`;
    thumb.onclick = () => {
      // Highlight selected
      Array.from(panelImages.children).forEach(t => t.style.borderColor = 'rgba(255,255,255,0.08)');
      thumb.style.borderColor = 'rgba(108,92,231,0.5)';
      // Show caption
      if (panelTextPreview) {
        panelTextPreview.textContent = `[Analisis #${idx + 1}]: ${entry.caption}`;
      }
    };
    panelImages.appendChild(thumb);
  });
  // Auto-scroll to latest
  panelImages.scrollLeft = panelImages.scrollWidth;
}

function startAutoCapture() {
  if (autoCaptureInterval) return;
  isAutoCapturing = true;
  autoCaptureInterval = setInterval(captureAndAnalyze, AUTO_CAPTURE_MS);
  console.log('[AutoCapture] Started');
}

function stopAutoCapture() {
  if (autoCaptureInterval) {
    clearInterval(autoCaptureInterval);
    autoCaptureInterval = null;
  }
  isAutoCapturing = false;
  console.log('[AutoCapture] Stopped');
}

// ===== PANEL CHAT: GENERATE UI BASED ON SCREENSHOT + USER PROMPT =====

async function handlePanelChat() {
  const text = panelChatInput.value.trim();
  if (!text) return;
  if (!generator) {
    if (panelTextPreview) panelTextPreview.textContent = 'Error: Modelo no cargado';
    return;
  }

  // Get latest screen analysis with rich style info
  const lastEntry = screenHistory[screenHistory.length - 1];
  const styleInfo = extractStyleSnapshot();
  const currentDesc = lastEntry ? lastEntry.caption : styleInfo.desc;
  const isLogin = styleInfo.type === 'login';

  panelChatInput.value = '';
  if (panelTextPreview) {
    panelTextPreview.textContent = `Generando: "${text}"...`;
  }
  showGenerating(true);

  try {
    // Short, direct prompt for the tiny 0.5B model
    const combinedPrompt = isLogin
      ? `Recreate this login form with style changes: ${text}. Use classes: login-card, login-title, login-subtitle, login-label, login-input, login-btn. IDs: loginForm, username, password, loginBtn. Use style="" only for colors. Output HTML only.`
      : `Recreate this dashboard with style changes: ${text}. Use classes: dashboard-container, dashboard-header, dashboard-title, dashboard-subtitle, dashboard-logout, dashboard-grid, dashboard-card, dashboard-card-label, dashboard-card-value, dashboard-panel, dashboard-panel-title, dashboard-table, badge, badge-progress, badge-planning, badge-done. IDs: dashboardView, logoutBtn. Use style="" only for colors. Output HTML only.`;

    const newHTML = await generateHTML(combinedPrompt);
    console.log('[Chat] Generated HTML length:', newHTML.length);
    console.log('[Chat] Generated HTML preview:', newHTML.substring(0, 200));

    // Validate generated HTML
    if (!newHTML || newHTML.trim().length < 50) {
      throw new Error('HTML generado vacio o incompleto');
    }

    // Determine if it's login or dashboard based on content
    if (newHTML.includes('loginForm') || newHTML.includes('login-card')) {
      loginHTMLCode = newHTML;
      currentPhase = 'login';
      renderLogin();
    } else if (newHTML.includes('dashboardView') || newHTML.includes('dashboard-container')) {
      dashboardHTMLCode = newHTML;
      currentPhase = 'dashboard';
      renderDashboard();
    } else {
      // Unknown structure - fallback to current view with original HTML
      console.warn('[Chat] HTML no reconocido, usando fallback');
      if (isLogin) {
        renderFallbackLogin();
      } else {
        renderFallbackDashboard();
      }
    }

    if (panelTextPreview) {
      panelTextPreview.textContent = `[OK]: "${text}"`;
    }
  } catch (err) {
    console.error('[handlePanelChat] Generation failed:', err);
    if (panelTextPreview) panelTextPreview.textContent = 'Error: usando diseño base';
    // Fallback to original
    if (isLogin) {
      renderFallbackLogin();
    } else {
      renderFallbackDashboard();
    }
  } finally {
    showGenerating(false);
  }
}

function bindPanelEvents() {
  if (!panelImages) return;

  if (panelChatSend && panelChatInput) {
    panelChatSend.onclick = handlePanelChat;
    panelChatInput.onkeydown = e => {
      if (e.key === 'Enter') handlePanelChat();
    };
  }
}

// Bind panel events on load
bindPanelEvents();

function renderLogin() {
  if (loginHTMLCode && loginHTMLCode.trim().length > 50) injectHTML(loginHTMLCode, 'center');
  else renderFallbackLogin();
}

function renderDashboard() {
  if (dashboardHTMLCode && dashboardHTMLCode.trim().length > 50) injectHTML(dashboardHTMLCode, 'start');
  else renderFallbackDashboard();
}

async function handleLogin() {
  showGenerating(true);
  currentPhase = 'generating-dashboard';
  try {
    dashboardHTMLCode = await generateHTML(`Create a modern analytics dashboard using ONLY these CSS classes: dashboard-container, dashboard-header, dashboard-title, dashboard-subtitle, dashboard-logout, dashboard-grid, dashboard-card, dashboard-card-label, dashboard-card-value, dashboard-panel, dashboard-panel-title, dashboard-table, badge, badge-progress, badge-planning, badge-done.

1. Top bar with "Dashboard" title and "Cerrar sesion" button (id="logoutBtn", class="dashboard-logout") on the right
2. 3 metric cards: "Usuarios: 1,248", "Ventas: 42,500 EUR", "Uptime: 99.9%"
3. Recent projects table with 4 example rows (use badge classes for status)
4. Container id="dashboardView", class="dashboard-container". NO onclick.`);
    currentPhase = 'dashboard';
    renderDashboard();
  } catch (err) {
    dashboardHTMLCode = null;
    currentPhase = 'dashboard';
    renderDashboard();
  } finally { showGenerating(false); }
}

async function handleLogout() {
  showGenerating(true);
  currentPhase = 'generating-login';
  try {
    loginHTMLCode = await generateHTML(`Create a modern login form centered on screen using ONLY these CSS classes: login-card, login-title, login-subtitle, login-label, login-input, login-btn.

1. Card centered, class="login-card"
2. Title "Bienvenido", class="login-title"
3. Subtitle "Ingresa tus credenciales", class="login-subtitle"
4. Label + input "Usuario", id="username", class="login-input"
5. Label + input "Contrasena" type="password", id="password", class="login-input"
6. Button "Acceder", id="loginBtn", class="login-btn"

Container id="loginForm", class="login-card". NO onclick.`);
    currentPhase = 'login';
    renderLogin();
  } catch (err) {
    loginHTMLCode = null;
    currentPhase = 'login';
    renderLogin();
  } finally { showGenerating(false); }
}

function renderFallbackLogin() {
  const html = `
    <div id="loginForm" class="login-card">
      <h1 class="login-title">Bienvenido</h1>
      <p class="login-subtitle">Ingresa tus credenciales para acceder</p>
      <label class="login-label">Usuario</label>
      <input id="username" type="text" placeholder="usuario@ejemplo.com" class="login-input">
      <label class="login-label">Contrasena</label>
      <input id="password" type="password" placeholder="--------" class="login-input">
      <button id="loginBtn" class="login-btn">Acceder</button>
    </div>`;
  injectHTML(html, 'center');
}

function renderFallbackDashboard() {
  const html = `
    <div id="dashboardView" class="dashboard-container">
      <div class="dashboard-header">
        <div>
          <h2 class="dashboard-title">Dashboard</h2>
          <p class="dashboard-subtitle">Panel de Control</p>
        </div>
        <button id="logoutBtn" class="dashboard-logout">Cerrar sesion</button>
      </div>
      <div class="dashboard-grid">
        <div class="dashboard-card">
          <p class="dashboard-card-label">Usuarios</p>
          <p class="dashboard-card-value">1,248</p>
        </div>
        <div class="dashboard-card">
          <p class="dashboard-card-label">Ventas</p>
          <p class="dashboard-card-value">42,500 EUR</p>
        </div>
        <div class="dashboard-card">
          <p class="dashboard-card-label">Uptime</p>
          <p class="dashboard-card-value">99.9%</p>
        </div>
      </div>
      <div class="dashboard-panel">
        <h3 class="dashboard-panel-title">Proyectos Recientes</h3>
        <table class="dashboard-table">
          <tr><th>Proyecto</th><th>Estado</th><th>Progreso</th></tr>
          <tr><td>Rediseño Web</td><td><span class="badge badge-progress">En progreso</span></td><td>75%</td></tr>
          <tr><td>App Movil</td><td><span class="badge badge-planning">Planificacion</span></td><td>30%</td></tr>
          <tr><td>Migracion Cloud</td><td><span class="badge badge-done">Completado</span></td><td>100%</td></tr>
        </table>
      </div>
    </div>`;
  injectHTML(html, 'start');
}

async function init() {
  // Reset any stale transform state
  transformApplied = false;
  canvasContent.style.transform = '';

  initParticles();
  animate();

  // Show fallback login immediately
  currentPhase = 'login';
  renderFallbackLogin();

  if (!navigator.gpu) { statusText.textContent = 'WebGPU no soportado'; return; }

  try {
    statusText.textContent = 'Iniciando WebGPU...';
    const isCached = await ModelRegistry.is_pipeline_cached("text-generation", modelId, { dtype: "q4f16" });
    if (isCached) statusText.textContent = 'Modelo en cache...';

    generator = await pipeline("text-generation", modelId, {
      device: "webgpu", dtype: "q4f16",
      progress_callback: (data) => {
        if (data.status === "progress_total") {
          progressFill.style.width = Math.round(data.progress) + '%';
          statusText.textContent = `Descargando IA: ${Math.round(data.progress)}%`;
        }
      }
    });

    statusText.textContent = 'Cargando modelo de vision...';
    try {
      imageCaptioner = await pipeline("image-to-text", imageCaptionModelId, {
        device: "webgpu",
        progress_callback: (data) => {
          if (data.status === "progress_total") {
            progressFill.style.width = Math.round(data.progress) + '%';
            statusText.textContent = `Descargando Vision: ${Math.round(data.progress)}%`;
          }
        }
      });
      console.log('[Vision] Modelo cargado');
    } catch (visionErr) {
      console.warn('[Vision] No se pudo cargar modelo de vision:', visionErr.message);
      imageCaptioner = null;
    }

    statusText.textContent = 'Generando login con IA...';
    showGenerating(true);
    currentPhase = 'generating-login';

    try {
      loginHTMLCode = await generateHTML(`Create a modern login form centered on screen using ONLY these CSS classes: login-card, login-title, login-subtitle, login-label, login-input, login-btn.

1. Card centered, class="login-card"
2. Title "Bienvenido", class="login-title"
3. Subtitle "Ingresa tus credenciales", class="login-subtitle"
4. Label + input "Usuario", id="username", class="login-input"
5. Label + input "Contrasena" type="password", id="password", class="login-input"
6. Button "Acceder", id="loginBtn", class="login-btn"

Container id="loginForm", class="login-card". NO onclick.`);
      currentPhase = 'login';
      if (loginHTMLCode && loginHTMLCode.trim().length > 50) renderLogin();
    } catch (err) {
      console.error('Login gen failed:', err);
    }

    showGenerating(false);
    overlay.classList.add('hidden');

    // Start periodic screen capture after models are loaded
    startAutoCapture();
    console.log('[Init] Auto-capture iniciado cada', AUTO_CAPTURE_MS, 'ms');
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message;
  }
}

init();
