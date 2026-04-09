import { pipeline, env, ModelRegistry } from '@huggingface/transformers';

env.useWasmCache = true;
env.logLevel = 'debug';

const statusEl = document.getElementById('status');
const promptInput = document.getElementById('prompt');
const generateBtn = document.getElementById('generateBtn');
const outputDiv = document.getElementById('output');

let generator = null;

// El modelo programador ultra-ligero (< 400 MB)
const modelId = "onnx-community/Qwen2.5-Coder-0.5B-Instruct"; 

// 1. EL SYSTEM PROMPT ESTRICTO
// Le decimos exactamente quién es y cómo debe devolver el código
let messages = [
  { 
    role: 'system', 
    content: `Eres un asistente de desarrollo web. Hablas en español.
    El usuario te pedirá que modifiques la interfaz de la página en la que te encuentras (por ejemplo, cambiar el color del botón "Generar Respuesta" que tiene el id "generateBtn").
    
    Cuando el usuario te pida modificar la interfaz, debes responder ÚNICAMENTE con el código JavaScript necesario para realizar ese cambio en el DOM.
    Envuelve el código en un bloque markdown de JavaScript así:
    \`\`\`javascript
    document.getElementById('generateBtn').style.backgroundColor = 'red';
    \`\`\`
    No des explicaciones, solo el código.`
  }
];

console.log("🚀 Script cargado - Iniciando WebGPU");

async function init() {
  statusEl.textContent = "Iniciando WebGPU...";
  generateBtn.disabled = true;

  if (!navigator.gpu) {
    statusEl.textContent = "❌ WebGPU no está disponible.";
    return;
  }

  try {
    const isCached = await ModelRegistry.is_pipeline_cached("text-generation", modelId, { dtype: "q4f16" });
    if (isCached) statusEl.textContent = "Cargando modelo desde caché...";

    generator = await pipeline("text-generation", modelId, {
      device: "webgpu",
      dtype: "q4f16", // Fundamental para que pese menos de 400 MB
      progress_callback: (data) => {
        if (data.status === "progress_total") {
          statusEl.textContent = `Descargando IA: ${Math.round(data.progress)}%`;
        }
      }
    });

    statusEl.textContent = "✅ ¡Modelo Coder cargado! Pídele un botón.";
    generateBtn.disabled = false;
  } catch (err) {
    console.error("🚨 ERROR:", err);
    statusEl.textContent = "Error: " + err.message;
  }
}

function appendMessage(role, text) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', role);
  msgDiv.innerHTML = `<strong>${role === 'user' ? 'Tú' : 'IA'}:</strong> <br>${text.replace(/\n/g, '<br>')}`;
  outputDiv.appendChild(msgDiv);
  outputDiv.scrollTop = outputDiv.scrollHeight;
}

generateBtn.addEventListener('click', async () => {
  const text = promptInput.value.trim();
  if (!text || !generator) return;

  appendMessage('user', text);
  promptInput.value = '';
  
  generateBtn.disabled = true;
  statusEl.textContent = "La IA está programando...";

  messages.push({ role: 'user', content: text });

  try {
    const output = await generator(messages, {
      max_new_tokens: 150, 
      temperature: 0.3,          // Temperatura baja para código (0.1 a 0.3 es ideal)
      repetition_penalty: 1.15,  // Evita bucles
      top_p: 0.9
    });

    const generatedMessage = output[0].generated_text.at(-1);

    if (generatedMessage && generatedMessage.role === 'assistant') {
      const respuestaTexto = generatedMessage.content;
      
      // Mostrar la respuesta en el chat
      appendMessage('ai', respuestaTexto);
      messages.push(generatedMessage);

      // MAGIA: Buscar si la IA generó código JavaScript
      // Usamos una expresión regular para encontrar bloques ```javascript ... ```
      const match = respuestaTexto.match(/```javascript\n([\s\S]*?)```/i) || 
                    respuestaTexto.match(/```js\n([\s\S]*?)```/i);
      
      if (match && match[1]) {
        const codigoJS = match[1].trim();
        console.log("¡La IA ha generado código JS! Ejecutándolo:\n", codigoJS);
        
        try {
          // Ejecutamos el código JavaScript devuelto por la IA
          // Usamos una función anónima para encapsular la ejecución de forma ligeramente más segura que un eval directo
          const ejecutarCambio = new Function(codigoJS);
          ejecutarCambio();
          console.log("✅ Cambio aplicado con éxito en la interfaz.");
        } catch (errorEjecucion) {
           console.error("❌ La IA generó código JS inválido:", errorEjecucion);
           appendMessage('ai', 'Intenté modificar la página, pero generé código inválido.');
        }
      }
    }

  } catch (err) {
    console.error("Error:", err);
    appendMessage('ai', 'Error al generar código.');
  } finally {
    statusEl.textContent = "✅ ¡Modelo Coder cargado! Pídele un botón.";
    generateBtn.disabled = false;
  }
});

init();