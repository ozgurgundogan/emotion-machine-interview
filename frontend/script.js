const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const backendInput = document.getElementById("backend");
const statusEl = document.getElementById("status");
const streamCheckbox = document.getElementById("use-stream");

const BACKEND_FALLBACK = "http://localhost:8000/api/query";

function appendBubble(text, role = "assistant") {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function appendHTMLBubble(html, role = "assistant") {
  const div = document.createElement("div");
  const extra = arguments.length > 2 ? arguments[2] : "";
  div.className = `bubble ${role}${extra ? ` ${extra}` : ""}`;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function formatJson(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function renderPlan(plan) {
  const candidates = plan?.candidates || [];
  const steps = plan?.plan?.steps || plan?.steps || [];
  const formatScore = (score) => {
    if (typeof score === "number") return score.toFixed(4);
    return score ?? "—";
  };

  if (candidates.length) {
    const rows = candidates
      .map((c) => {
        const params = c.parameters || {};
        const req = params.required || [];
        const opt = params.optional || [];
        const paramText = [
          req.length ? `required: ${req.map((p) => p.name || "").join(", ")}` : "",
          opt.length ? `optional: ${opt.map((p) => p.name || "").join(", ")}` : "",
        ]
          .filter(Boolean)
          .join(" | ");
        return `
          <tr>
            <td>${c.name || ""}</td>
            <td>${c.api_name || ""}</td>
            <td>${c.description || ""}</td>
            <td class="score-cell">${formatScore(c.score)}</td>
            <td>${paramText || "—"}</td>
          </tr>
        `;
      })
      .join("");
    appendHTMLBubble(
      `<div><strong>Candidates</strong></div>
       <table class="candidate-table">
         <thead><tr><th>Name</th><th>API Name</th><th>Description</th><th>Score</th><th>Parameters</th></tr></thead>
         <tbody>${rows}</tbody>
       </table>`,
      "assistant",
      "candidate"
    );
  }

  if (steps.length) {
    const rows = steps
      .map((step, idx) => {
        const args = step.arguments || {};
        const argLines = Object.entries(args)
          .map(([k, v]) => `${k}: ${v}`)
          .join(", ");
        return `
          <tr>
            <td>${idx + 1}</td>
            <td>${step.name || ""}</td>
            <td>${step.api_name || ""}</td>
            <td>${step.tool_id || ""}</td>
            <td>${argLines || "—"}</td>
          </tr>
        `;
      })
      .join("");
    appendHTMLBubble(
      `<div><strong>Plan</strong></div>
       <table class="steps-table">
         <thead><tr><th>#</th><th>Name</th><th>API Name</th><th>Tool ID</th><th>Arguments</th></tr></thead>
         <tbody>${rows}</tbody>
       </table>`,
      "assistant"
    );
  }

  if (!candidates.length && !steps.length) {
    appendBubble(formatJson(plan), "assistant");
  }
}

async function streamResponse(res) {
  const reader = res.body?.getReader();
  if (!reader) {
    const txt = await res.text();
    renderPlan({ plan: txt || "No content returned." });
    return;
  }

  const decoder = new TextDecoder();
  let acc = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    acc += decoder.decode(value, { stream: true });
  }
  acc += decoder.decode();
  try {
    const parsed = JSON.parse(acc);
    renderPlan(parsed);
  } catch {
    renderPlan({ plan: acc.trim() || "No content returned." });
  }
}

async function sendQuery(message) {
  const backend = backendInput.value || BACKEND_FALLBACK;
  statusEl.textContent = "Sending...";
  try {
    const res = await fetch(backend, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: message, stream: !!streamCheckbox.checked }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const contentType = res.headers.get("content-type") || "";
    if (res.body && (contentType.includes("text") || contentType.includes("event-stream"))) {
      await streamResponse(res);
    } else {
      const data = await res.json().catch(() => ({}));
      renderPlan(data);
    }
  } catch (err) {
    appendBubble(`Error contacting backend: ${err.message}`, "assistant");
  } finally {
    statusEl.textContent = "";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;
  appendBubble(message, "user");
  promptInput.value = "";
  await sendQuery(message);
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

appendBubble("Hi! Describe what you want to do and I'll pick the right tool.");
