import { pipeline, env, ModelRegistry } from '@huggingface/transformers';

env.useWasmCache = true;
env.logLevel = 'none';

const modelId = "onnx-community/Qwen2.5-Coder-0.5B-Instruct";
const overlay = document.getElementById('overlay');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const generatingOverlay = document.getElementById('generating');
const canvas = document.getElementById('app-canvas');
const ctx = canvas.getContext('2d');
const canvasContent = document.getElementById('canvas-content');

let generator = null;
let currentPhase = 'init';
let loginHTMLCode = null;
let dashboardHTMLCode = null;
let dpr = window.devicePixelRatio || 1;
let particles = [];
let useDrawElement = typeof ctx.drawElementImage === 'function';
let paintRetryCount = 0;
let transformApplied = false;
const MAX_PAINT_RETRIES = 5;

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
  html = html.replace(/<\s*style\s*[^>]*>[\s\S]*?<\/\s*style\s*>/gi, '');
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

async function generateHTML(prompt) {
  const sys = `You are a strict HTML fragment generator. Output ONLY valid HTML code. No explanations, no markdown, no code blocks.

STRICT RULES:
1. ONLY use these exact CSS classes. Any other class will be removed.
   Login: login-card, login-title, login-subtitle, login-label, login-input, login-btn
   Dashboard: dashboard-container, dashboard-header, dashboard-title, dashboard-subtitle, dashboard-logout, dashboard-grid, dashboard-card, dashboard-card-label, dashboard-card-value, dashboard-panel, dashboard-panel-title, dashboard-table, badge, badge-progress, badge-planning, badge-done
   For badges ONLY use: badge-progress, badge-planning, badge-done. NEVER use badge-success, badge-info, badge-warning, badge-danger, or any other badge class.
2. NO inline styles. NO style attribute anywhere.
3. NO <html>, <body>, <head>, <meta>, <title>, <link>, <script>, <style> tags.
4. NO HTML comments <!-- -->
5. NO onclick, onsubmit, or any event attributes.
6. IDs must be exact: login -> loginForm, username, password, loginBtn. dashboard -> dashboardView, logoutBtn.
7. Use semantic tags: h1 for main title, p for text, label for labels, input for fields, button for buttons, table for tables.

LOGIN EXAMPLE (follow this exact structure):
<div id="loginForm" class="login-card">
  <h1 class="login-title">Bienvenido</h1>
  <p class="login-subtitle">Ingresa tus credenciales</p>
  <label class="login-label">Usuario</label>
  <input id="username" type="text" placeholder="Usuario" class="login-input">
  <label class="login-label">Contrasena</label>
  <input id="password" type="password" placeholder="Contrasena" class="login-input">
  <button id="loginBtn" class="login-btn">Acceder</button>
</div>

DASHBOARD EXAMPLE (follow this exact structure):
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
    ...
  </div>
  <div class="dashboard-panel">
    <h3 class="dashboard-panel-title">Proyectos Recientes</h3>
    <table class="dashboard-table">
      <tr><th>Proyecto</th><th>Estado</th><th>Progreso</th></tr>
      <tr><td>Rediseño Web</td><td><span class="badge badge-progress">En progreso</span></td><td>75%</td></tr>
      ...
    </table>
  </div>
</div>`;

  const textInput = `<|im_start|>system\n${sys}<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
  const output = await generator(textInput, { max_new_tokens: 900, temperature: 0.2, top_p: 0.85 });
  const generated = output[0].generated_text;
  const newText = generated.slice(textInput.length);
  console.log('[AI RAW]:', newText.substring(0, 1000));
  const cleaned = cleanHTML(newText);
  console.log('[AI CLEAN]:', cleaned.substring(0, 1000));
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

const MOCK_DESCRIPTIONS = [
  'Paisaje urbano nocturno con luces de neon reflejadas en el agua.',
  'Retrato artistico con paleta de colores violeta y azul oscuro.',
  'Diseno minimalista de interfaz con gradientes suaves.',
  'Arquitectura moderna con lineas geometricas puras.',
  'Textura abstracta de particulas flotantes en el espacio.',
  'Codigo fuente con sintaxis resaltada sobre fondo oscuro.'
];

function bindPanelEvents() {
  if (!panelImages) return;
  const thumbs = panelImages.querySelectorAll('.img-thumb');
  thumbs.forEach((thumb, idx) => {
    thumb.onclick = () => {
      // Highlight selected
      thumbs.forEach(t => t.style.borderColor = 'rgba(255,255,255,0.08)');
      thumb.style.borderColor = 'rgba(108,92,231,0.5)';
      // Show description
      if (panelTextPreview) {
        panelTextPreview.textContent = MOCK_DESCRIPTIONS[idx] || '';
      }
    };
  });

  if (panelChatSend && panelChatInput) {
    panelChatSend.onclick = () => {
      const text = panelChatInput.value.trim();
      if (text && panelTextPreview) {
        panelTextPreview.textContent = `[Prompt]: ${text}`;
        panelChatInput.value = '';
      }
    };
    panelChatInput.onkeydown = e => {
      if (e.key === 'Enter') panelChatSend.click();
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
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message;
  }
}

init();
