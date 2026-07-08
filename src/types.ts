export type RoutePath = 'cache' | 'local' | 'remote';

export interface RoutingInfo {
  path: RoutePath;
  confidence: number;
  tokensUsed: number;
  latencyMs: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  routing?: RoutingInfo;
  timestamp: number;
  error?: boolean;
}

export interface SessionStats {
  totalQueries: number;
  cacheHits: number;
  localResolutions: number;
  remoteFallbacks: number;
  totalTokensUsed: number;
  estimatedRemoteTokens: number;
}