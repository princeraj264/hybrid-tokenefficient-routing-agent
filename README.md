# Hybrid Token-Efficient Routing Agent

> **AMD Developer Hackathon Act II — 2026** · A smart query router that minimises LLM token costs by tiered escalation: cache → local model → remote API.

![Hackathon](https://img.shields.io/badge/AMD-Developer%20Hackathon%20Act%20II-red?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite%20%2B%20TypeScript-61dafb?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-Python%20FastAPI-009688?style=flat-square)

---

## Overview

Every LLM call costs tokens — and those costs add up fast, especially when you're calling a paid remote API for every query, no matter how simple. This project builds a **hybrid routing agent** that answers queries through three tiers, each cheaper than the last:

| Tier | Model / Source | Relative Cost | When It Fires |
|------|---------------|---------------|---------------|
| 🟢 **Cache** | Exact or semantic cache | **Free** (0 tokens) | Exact or near-exact repeat query |
| 🟡 **Local** | Gemma 2B via `llama-cpp-python` (designed for ROCm on AMD Instinct™ GPU) | **Cheap** (local inference) | Moderate confidence from scoring |
| 🔴 **Remote** | Fireworks AI API (Qwen 3.7 Plus, `reasoning_effort: 'none'`) | **Full cost** (paid tokens) | Low confidence → escalate to strongest model |

The routing decision is based on **log-probability confidence scoring**, not self-reported confidence from the models. The system evaluates how certain the local model is about its generated tokens and escalates only when that certainty falls below a configurable threshold. This gives you provable, measurable confidence rather than a model's own (often inflated) guess.

---

## How Routing Works (Step by Step)

1. **Cache lookup** — The query is checked against an in-memory semantic cache. On a close match the cached answer is returned immediately with zero token cost.
2. **Local inference** — If the cache misses, Gemma 2B (running locally via `llama-cpp-python`) generates an answer. The system extracts generative log probabilities from the output.
3. **Confidence scoring** — A mean log-probability score is computed across generated tokens. If the score exceeds a threshold, the local answer is accepted.
4. **Remote escalation** — If confidence is too low, the query is forwarded to Qwen 3.7 Plus on Fireworks AI (with `reasoning_effort: 'none'` to skip the thinking phase). That answer is authoritative but costs the most tokens.
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
| Local model | **Gemma 2B** via `llama-cpp-python` (ROCm-ready for AMD Developer Cloud deployment) |
| Remote API | Fireworks AI (Qwen 3.7 Plus, `reasoning_effort: 'none'`) |
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
  "answer": "The capital of France is Paris.",
  "path": "local",
  "confidence": 0.96,
  "tokens_used": 29,
  "latency_ms": 9506
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | The generated answer |
| `path` | `"cache"` \| `"local"` \| `"remote"` | Which tier answered |
| `confidence` | `number` (0–1) | Log-probability confidence score |
| `tokens_used` | `number` | Tokens consumed (0 for cache hits) |
| `latency_ms` | `number` | Total request latency in milliseconds |

> **Note:** The frontend also accepts `content` (alias for `answer`), `route` (alias for `path`), `tokensUsed` / `latencyMs` (camelCase variants), and `cache_hit` (alias for `"cache"` path) to handle minor inconsistencies across backend versions.

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
llama-cpp-python       # with ROCm BLAS support — built from source on the AMD instance
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

This project participates in the **Gemma Prize Track** by using **Gemma 2B** as the local inference model. The model runs via `llama-cpp-python` (ROCm BLAS support is compiled in, and deployment on AMD Instinct™ GPU via the AMD Developer Cloud is in progress).

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