const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/** Every request to the ngrok tunnel must include this header to skip the warning page. */
const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

export interface QueryResponse {
  content: string;
  path: 'cache' | 'local' | 'remote';
  confidence: number;
  tokensUsed: number;
  latencyMs: number;
}

export interface HealthResponse {
  status: string;
  [key: string]: unknown;
}

/**
 * Send a query to the hybrid routing agent.
 * No timeout is set — remote-escalated queries can take 60–120 seconds.
 */
export async function queryAgent(text: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...NGROK_HEADERS,
    },
    body: JSON.stringify({ query: text }),
    // No signal / timeout — queries can legitimately take 2+ minutes
  });

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Backend returned ${res.status}${body ? `: ${body}` : ''}`);
  }

  const data = await res.json();

  // Normalise response — the backend may return snake_case, camelCase, or 'cache_hit'
  const rawPath: string = data.path ?? data.route ?? 'remote';
  return {
    content: data.content ?? data.answer ?? '',
    path: rawPath === 'cache_hit' ? 'cache' : rawPath,
    confidence: data.confidence ?? 0,
    tokensUsed: data.tokensUsed ?? data.tokens_used ?? 0,
    latencyMs: data.latencyMs ?? data.latency_ms ?? 0,
  };
}

/** Check whether the backend is reachable. */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`, {
    headers: NGROK_HEADERS,
  });

  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }

  return res.json();
}