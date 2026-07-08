# Hybrid Token-Efficient Routing Agent

> **AMD Developer Hackathon 2025 — Track 1** · A smart query router that minimises LLM token costs by tiered escalation: cache → local model → remote API.

![Hackathon](https://img.shields.io/badge/AMD-Developer%20Hackathon-red?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite%20%2B%20TypeScript-61dafb?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-Python%20FastAPI-009688?style=flat-square)

---

## Overview

Every LLM call costs tokens — and those costs add up fast, especially when you're calling a paid remote API for every query, no matter how simple. This project builds a **hybrid routing agent** that answers queries through three tiers, each cheaper than the last:

| Tier | Model / Source | Relative Cost | When It Fires |
|------|---------------|---------------|---------------|
| 🟢 **Cache** | Exact or semantic cache | **Free** (0 tokens) | Exact or near-exact repeat query |
| 🟡 **Local** | Gemma 2B via ROCm on AMD Instinct™ GPU | **Cheap** (local inference) | Moderate confidence from scoring |
| 🔴 **Remote** | Fireworks AI API (Mixtral / Llama 3) | **Full cost** (paid tokens) | Low confidence → escalate to strongest model |

The routing decision is based on **log-probability confidence scoring**, not self-reported confidence from the models. The system evaluates how certain the local model is about its generated tokens and escalates only when that certainty falls below a configurable threshold. This gives you provable, measurable confidence rather than a model's own (often inflated) guess.

---

## How Routing Works (Step by Step)

1. **Cache lookup** — The query is checked against an in-memory semantic cache. On a close match the cached answer is returned immediately with zero token cost.
2. **Local inference** — If the cache misses, Gemma 2B (running locally on AMD ROCm) generates an answer. The system extracts generative log probabilities from the output.
3. **Confidence scoring** — A mean log-probability score is computed across generated tokens. If the score exceeds a threshold, the local answer is accepted.
4. **Remote escalation** — If confidence is too low, the query is forwarded to a larger model on Fireworks AI. That answer is authoritative but costs the most tokens.
5. **Cache update** — Every accepted answer (local or remote) is stored back in the cache for future hits.

This means **simple, repetitive, or well-known queries never touch the remote API**, and hard/unseen queries get the full power of a frontier model.

---

## Tech Stack

### Frontend (`src/`)

| Layer | Choice |
|-------|--------|
| UI Framework | React 18 |
| Build Tool | Vite 6 |
| Language | TypeScript |
| Styling | TailwindCSS v4 |
| Icons | Lucide React |

The frontend is a single-page chat app that visualises every routing decision in real time — path taken, confidence score, token usage, and latency — alongside a session-level summary of total tokens saved.

### Backend (separate repo / service)

| Layer | Choice |
|-------|--------|
| Framework | Python FastAPI |
| Local model | **Gemma 2B** via `llama-cpp-python` with ROCm support |
| Remote API | Fireworks AI |
| Hosting | AMD Developer Cloud (ROCm-enabled instance) |

The backend exposes a REST API consumed by this frontend. It is **not included in this repository** — this repo is the web UI only.

---

## API Contract

The frontend expects the backend to expose two endpoints:

### `POST /query`

Submit a query to the routing agent.

**Request:**
```json
{
  "query": "What is the capital of France?"
}
```

**Response:**
```json
{
  "content": "The capital of France is Paris.",
  "path": "cache",
  "confidence": 0.97,
  "tokensUsed": 42,
  "latencyMs": 85
}
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | `string` | The generated answer |
| `path` | `"cache"` \| `"local"` \| `"remote"` | Which tier answered |
| `confidence` | `number` (0–1) | Log-probability confidence score |
| `tokensUsed` | `number` | Tokens consumed (0 for cache hits) |
| `latencyMs` | `number` | Total request latency in milliseconds |

> **Note:** The frontend also accepts `route` (alias for `path`), `cache_hit` (alias for `"cache"`), `answer` (alias for `content`), and `tokens_used` / `latency_ms` (snake_case variants) to handle minor backend inconsistencies.

### `GET /health`

Check whether the backend is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Setup Instructions

### 1. Clone the frontend

```bash
git clone <repo-url>
cd hybrid-routing-agent
npm install
```

### 2. Configure the backend URL

The backend URL is currently hardcoded in `src/lib/api.ts`. Replace the `BASE_URL` constant with your deployed backend's address:

```ts
const BASE_URL = 'https://your-amd-cloud-instance.ngrok-free.app';
```

### 3. Run the development server

```bash
npm run dev
```

Opens at `http://localhost:5173`.

### 4. Build for production

```bash
npm run build
```

Output goes to `dist/`.

---

## Backend Setup (for the AMD Developer Cloud)

The backend is a separate Python FastAPI service. To set it up:

### Prerequisites

- AMD Developer Cloud instance with ROCm installed
- Python 3.10+
- A Fireworks AI API key

### Dependencies

```txt
fastapi
uvicorn
llama-cpp-python       # with ROCm support — built from source on the AMD instance
fireworks-ai            # official Fireworks SDK
requests
numpy
```

### Environment Variables

```bash
# Required
FIREWORKS_API_KEY=your_key_here

# Optional — tune behaviour
ROUTING_CONFIDENCE_THRESHOLD=0.75   # default: 0.75
MODEL_PATH=/path/to/gemma-2b-Q4_K_M.gguf
CACHE_SIZE=1000                     # max cache entries
```

### Running the backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The frontend expects the backend to be reachable via HTTPS (an ngrok tunnel is typical for development).

---

## Gemma Prize Track

This project participates in the **Gemma Prize Track** by using **Gemma 2B** as the local inference model. The model runs on AMD ROCm via `llama-cpp-python` with ROCm BLAS acceleration, taking advantage of the AMD Instinct™ GPU on the Developer Cloud.

Integration follows Fireworks AI's Gemma model path for the remote fallback tier, making Gemma available at both the local and remote levels. The local Gemma handles the majority of routine queries (saving tokens), while Fireworks escalations use a larger model for complex cases.

---

## Frontend Features

- **Real-time routing visualisation** — Every answer bubble shows which tier handled it, with colour-coded badges (green 🟢 / amber 🟡 / red 🔴)
- **Confidence bars** — Horizontal gradient bars make the log-probability score instantly readable
- **Session summary** — Side panel tracks total queries, cache hit rate, local resolution rate, tokens saved, and savings vs. always-remote baseline
- **Dark / Light theme** — Respects `prefers-color-scheme` and includes a manual toggle
- **Responsive layout** — Sidebar collapses into an overlay on mobile and tablet
- **Error resilience** — Connection health check on load, inline error messages with retry, graceful degradation

---

## License

MIT — built for the AMD Developer Hackathon 2025.