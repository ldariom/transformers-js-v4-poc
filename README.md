# Transformers.js v4 + HTML-in-Canvas PoC

Banco de pruebas de interfaces generadas por IA dentro de un `<canvas>` usando `layoutsubtree` + `drawElementImage` (WICG) y Transformers.js v4.

## Demostración

Flujo completo:
1. Se carga un modelo de generación de texto via WebGPU
2. La IA genera un formulario de login y lo renderiza dentro del canvas
3. El usuario "inicia sesión" → la IA genera un dashboard
4. El usuario "cierra sesión" → la IA regenera el login
5. **Auto-captura**: cada 10s se toma un screenshot del canvas y se analiza con un modelo de visión
6. **Chat de estilos**: el usuario escribe cómo quiere cambiar el diseño y la IA lo aplica

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| **Modelo de texto** | `onnx-community/Qwen2.5-Coder-0.5B-Instruct` (quantizado q4f16) |
| **Modelo de visión** | `Xenova/vit-gpt2-image-captioning` (image-to-text) |
| **Runtime** | Transformers.js v4 con WebGPU |
| **Renderizado** | WICG `layoutsubtree` + `drawElementImage` (Chrome experimental) |
| **Build** | Vite 8 |

---

## Requisitos

- **Chrome/Edge** con flag experimental activado:
  - Abrir `chrome://flags/#canvas-draw-element`
  - Activar **"Canvas drawElementImage API"**
  - Reiniciar el navegador
- **WebGPU** soportado (Chrome 113+)
- Servidor local (Vite dev server)

---

## Instalación

```bash
npm install
npm run dev
```

El dev server arranca en `http://localhost:5173`.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│  index.html                                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  <canvas layoutsubtree>                                 │ │
│  │  ┌───────────────────────────────────────────────────┐ │ │
│  │  │  <div id="canvas-content">                        │ │ │
│  │  │    <style>/* CANVAS_SUBTREE_CSS */</style>       │ │ │
│  │  │    <div id="loginForm" class="login-card">...</div>│ │ │
│  │  │  </div>                                            │ │ │
│  │  └───────────────────────────────────────────────────┘ │ │
│  │  </canvas>                                              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  <div id="bottom-panel">                                │ │
│  │    <div class="panel-images">/* thumbnails */</div>    │ │
│  │    <div class="panel-right">                            │ │
│  │      <div class="panel-text-preview"></div>            │ │
│  │      <input placeholder="Describe la imagen...">       │ │
│  │      <button>Enviar</button>                            │ │
│  │    </div>                                                │ │
│  │  </div>                                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de renderizado

```
┌──────────┐    layoutsubtree    ┌─────────────────┐
│   DOM    │ ─────────────────▶ │  Invisible en   │
│  (HTML)  │                    │  el documento   │
└──────────┘                    └────────────────┘
                                         │
                                         │ canvas.requestPaint()
                                         ▼
┌──────────┐    onpaint event    ┌─────────────────┐
│  Canvas  │ ◀────────────────── │  drawElement    │
│  (2D ctx)│                     │  Image()        │
└──────────┘                     └─────────────────┘
```

### Flujo de generación de UI

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Screenshot │────▶│  VLM/Caption │────▶│  screen     │
│  (canvas)   │     │  (VIT-GPT2)  │     │  History    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Nueva UI   │◀────│  text-gen    │◀────│  User prompt │
│  (canvas)   │     │  (Qwen 0.5B) │     │  + contexto  │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Archivos principales

| Archivo | Descripción |
|---|---|
| `index.html` | Estructura HTML, CSS global, overlays, bottom panel |
| `src/main.js` | Lógica completa: pipelines, generación, renderizado, eventos |

### `src/main.js` — Secciones clave

| Función | Responsabilidad |
|---|---|
| `init()` | Carga modelos, inicia partículas, renderiza fallback login |
| `generateHTML(prompt)` | Llama al pipeline `text-generation` y limpia el output |
| `injectHTML(html, align)` | Inyecta HTML en canvas-content con estilos inline |
| `captureAndAnalyze()` | Screenshot + caption VLM → guarda en `screenHistory` |
| `handlePanelChat()` | Prompt del usuario → genera nueva UI con estilos |
| `extractStyleSnapshot()` | Extrae colores, fuentes, bordes del DOM actual |
| `renderThumbnails()` | Renderiza galería de screenshots en el panel inferior |
| `ensureCentering()` | Fuerza estilos inline para alinear DOM con snapshot |
| `cleanHTML(raw)` | Sanitiza HTML generado: elimina tags no deseados, clases inválidas |

---

## CSS: Dual-source

Los estilos se definen en **dos lugares** por diseño:

1. **`<head>` de `index.html`** — Estilos para el DOM real (hit-testing, layout)
2. **`CANVAS_SUBTREE_CSS` en `main.js`** — Estilos inline inyectados dentro del subtree (para que `drawElementImage` los capture en el snapshot)

Ambos son idénticos. Esto es necesario porque la implementación experimental de Chrome no resuelve estilos del `<head>` para elementos dentro de `layoutsubtree`.

---

## Problemas conocidos y workarounds

### 1. Estilos del `<head>` no se aplican al snapshot
**Causa:** `layoutsubtree` crea un contexto de layout aislado donde el motor de estilos no resuelve reglas del documento principal.

**Solución:** Duplicar estilos como `<style>` inline dentro del subtree.

### 2. `drawElementImage` no captura `backdrop-filter`
**Causa:** La implementación experimental no soporta filtros de backdrop en el snapshot.

**Solución:** Eliminar `backdrop-filter` del CSS inline del subtree.

### 3. Drift del transform DOM
**Causa:** Aplicar el `transform` retornado por `drawElementImage` en cada frame mueve el elemento invisible, y el siguiente snapshot se captura desde la posición desplazada.

**Solución:** Aplicar el transform **una sola vez** por inyección de HTML (flag `transformApplied`).

### 4. Modelo de visión no soporta imágenes
**Causa:** `Qwen2.5-Coder-0.5B-Instruct` es solo texto.

**Solución:** Usar `Xenova/vit-gpt2-image-captioning` como modelo de visión secundario.

### 5. Modelo 0.5B limitado para prompts complejos
**Causa:** El modelo es muy pequeño y no entiende instrucciones de estilo abstractas.

**Solución:** Prompts cortos y directos, fallback robusto si la generación falla.

---

## Variables de estado

| Variable | Tipo | Descripción |
|---|---|---|
| `generator` | Pipeline | Modelo de text-generation (Qwen 0.5B) |
| `imageCaptioner` | Pipeline | Modelo de image-to-text (VIT-GPT2) |
| `screenHistory[]` | Array | Historial de capturas: `{dataURL, caption, html, timestamp}` |
| `currentPhase` | String | `'login'`, `'dashboard'`, `'generating-*'`, `'init'` |
| `useDrawElement` | Boolean | Si `drawElementImage` está disponible |
| `transformApplied` | Boolean | Si ya se aplicó el transform al DOM actual |
| `autoCaptureInterval` | Interval | Timer de auto-captura cada 10s |

---

## Futuras mejoras

- [ ] Modelo VLM real (Qwen2.5-VL-3B) para análisis visual directo
- [ ] Soporte para múltiples vistas (settings, profile, etc.)
- [ ] Persistencia de historial en localStorage
- [ ] Exportar screenshots como PNG
- [ ] Undo/redo de cambios de estilo
- [ ] Soporte para Firefox/Safari (cuando implementen `layoutsubtree`)

---

## Licencia

PoC experimental — sin licencia formal.
