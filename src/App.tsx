import { useState, useCallback, useEffect } from 'react';
import { Moon, Sun, BarChart3 } from 'lucide-react';
import ChatMessages from './components/ChatMessages';
import ChatInput from './components/ChatInput';
import SessionSummary from './components/SessionSummary';
import { queryAgent } from './lib/api';
import type { Message, SessionStats } from './types';
import type { QueryResponse } from './lib/api';

type Theme = 'dark' | 'light';

function now(): number {
  return Date.now();
}

let msgIdCounter = 0;
function generateId(): string {
  return `msg_${Date.now()}_${++msgIdCounter}`;
}

/**
 * Rough estimate of what the equivalent remote-only query would cost.
 * Real remote tier uses ~2× the tokens of the local/cache path.
 */
function estimateRemoteTokens(actualTokens: number): number {
  return Math.round(actualTokens * 2);
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [failedQuery, setFailedQuery] = useState<string | null>(null);
  const [stats, setStats] = useState<SessionStats>({
    totalQueries: 0,
    cacheHits: 0,
    localResolutions: 0,
    remoteFallbacks: 0,
    totalTokensUsed: 0,
    estimatedRemoteTokens: 0,
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // ── Theme ──────────────────────────────────────────
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'dark';
    return window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  });

  // Sync data-theme attribute on <html>
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  // ── Handlers ───────────────────────────────────────

  /** Process a successful API response. */
  const handleSuccess = useCallback(
    (query: string, res: QueryResponse) => {
      const userMsg: Message = {
        id: generateId(),
        role: 'user',
        content: query,
        timestamp: now(),
      };

      const assistantMsg: Message = {
        id: generateId(),
        role: 'assistant',
        content: res.content,
        routing: {
          path: res.path,
          confidence: res.confidence,
          tokensUsed: res.tokensUsed,
          latencyMs: res.latencyMs,
        },
        timestamp: now(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setFailedQuery(null);

      setStats((prev) => {
        const newCacheHits = prev.cacheHits + (res.path === 'cache' ? 1 : 0);
        const newLocalResolutions = prev.localResolutions + (res.path === 'local' ? 1 : 0);
        const newRemoteFallbacks = prev.remoteFallbacks + (res.path === 'remote' ? 1 : 0);
        return {
          totalQueries: prev.totalQueries + 1,
          cacheHits: newCacheHits,
          localResolutions: newLocalResolutions,
          remoteFallbacks: newRemoteFallbacks,
          totalTokensUsed: prev.totalTokensUsed + res.tokensUsed,
          estimatedRemoteTokens: prev.estimatedRemoteTokens + estimateRemoteTokens(res.tokensUsed),
        };
      });
    },
    [],
  );

  /** Handle an API failure. */
  const handleError = useCallback(
    (query: string, error: unknown) => {
      const errorMsg: Message = {
        id: generateId(),
        role: 'system',
        content:
          error instanceof Error
            ? error.message
            : 'An unexpected error occurred. Please try again.',
        error: true,
        timestamp: now(),
      };

      setMessages((prev) => [...prev, errorMsg]);
      setFailedQuery(query);
    },
    [],
  );

  /** Send a query to the backend. */
  const handleSend = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) return;

      setLoading(true);

      try {
        const res = await queryAgent(trimmed);
        handleSuccess(trimmed, res);
      } catch (err) {
        handleError(trimmed, err);
      } finally {
        setLoading(false);
      }
    },
    [handleSuccess, handleError],
  );

  const handleRetry = useCallback(() => {
    if (failedQuery) {
      handleSend(failedQuery);
    }
  }, [failedQuery, handleSend]);

  const handleReset = useCallback(() => {
    setMessages([]);
    setFailedQuery(null);
    setStats({
      totalQueries: 0,
      cacheHits: 0,
      localResolutions: 0,
      remoteFallbacks: 0,
      totalTokensUsed: 0,
      estimatedRemoteTokens: 0,
    });
  }, []);

  return (
    <div className="h-full flex">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center justify-between px-4 py-2.5 border-b border-border/50 bg-background/95 backdrop-blur-sm">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse shrink-0" />
            <h1 className="text-sm font-semibold text-foreground/80 truncate">
              Hybrid Token-Efficient Router
            </h1>
          </div>

          <div className="flex items-center gap-1.5">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="text-foreground/40 hover:text-foreground/70 p-1.5 rounded-lg hover:bg-muted transition-all duration-150 cursor-pointer active:scale-95"
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4" aria-hidden="true" />
              ) : (
                <Moon className="w-4 h-4" aria-hidden="true" />
              )}
            </button>

            {/* Mobile sidebar toggle — hidden on md+ (768px) */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden flex items-center gap-1.5 text-xs text-foreground/50 hover:text-foreground/80 px-2.5 py-1.5 rounded-lg border border-border/30 hover:bg-muted transition-all duration-150 cursor-pointer"
              aria-label="Toggle session stats panel"
            >
              <BarChart3 className="w-3.5 h-3.5" aria-hidden="true" />
              <span className="hidden sm:inline">Stats</span>
            </button>
          </div>
        </header>

        <ChatMessages messages={messages} loading={loading} />
        <ChatInput
          onSend={handleSend}
          onRetry={handleRetry}
          disabled={loading}
          failedQuery={failedQuery}
        />
      </div>

      <SessionSummary
        stats={stats}
        messagesCount={messages.length}
        onReset={handleReset}
        sidebarOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
    </div>
  );
}