import { useEffect, useRef, useCallback } from 'react';
import { X, BarChart3, TrendingDown, RotateCcw } from 'lucide-react';
import type { SessionStats } from '../types';

/* ─── Sub-components ─────────────────────────────── */

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className="bg-muted/60 border border-border/30 rounded-xl p-3">
      <p className="text-[10px] text-foreground/40 uppercase tracking-wider">{label}</p>
      <p
        className={`text-xl font-bold tabular-nums mt-0.5 ${
          accent ? 'text-cache' : 'text-foreground/90'
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function RouteBar({
  label,
  count,
  total,
  color,
}: {
  label: string;
  count: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-foreground/70">{label}</span>
        <span className="tabular-nums text-foreground/50">{pct}%</span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`${label}: ${pct}%`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

/* ─── Main component ──────────────────────────────── */

export interface SessionSummaryProps {
  stats: SessionStats;
  messagesCount: number;
  onReset: () => void;
  sidebarOpen: boolean;
  onClose: () => void;
}

export default function SessionSummary({
  stats,
  messagesCount,
  onReset,
  sidebarOpen,
  onClose,
}: SessionSummaryProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Computed savings
  const tokensSaved = stats.estimatedRemoteTokens - stats.totalTokensUsed;
  const savingsPct =
    stats.estimatedRemoteTokens > 0
      ? Math.round((tokensSaved / stats.estimatedRemoteTokens) * 100)
      : 0;

  // Escape key and focus trap for mobile overlay
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }
      // Focus trap: cycle Tab / Shift+Tab within the panel
      if (e.key === 'Tab' && panelRef.current) {
        const focusable = panelRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    },
    [onClose]
  );

  // Attach/detach keyboard handlers when mobile overlay opens
  useEffect(() => {
    if (!sidebarOpen) return;
    document.addEventListener('keydown', handleKeyDown);
    // Focus the close button on open
    closeRef.current?.focus();
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [sidebarOpen, handleKeyDown]);

  /* ----- Shared content ----- */
  const content = (
    <>
      {/* Session heading */}
      <div className="p-4 border-b border-border/50">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-foreground/50" aria-hidden="true" />
          <span className="text-xs text-foreground/40 uppercase tracking-wider font-medium">
            Session
          </span>
        </div>
      </div>

      {/* Scrollable stats area */}
      <div className="flex-1 overflow-y-auto scrollable p-4 space-y-4">
        {/* Stat cards */}
        <div className="grid grid-cols-2 gap-2">
          <StatCard label="Queries" value={stats.totalQueries} />
          <StatCard
            label="Saved"
            value={stats.estimatedRemoteTokens > 0 ? `${savingsPct}%` : '\u2014'}
            accent={tokensSaved > 0}
          />
        </div>

        {/* Route breakdown */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-1 h-3 rounded-full bg-foreground/20" />
            <span className="text-xs text-foreground/40 uppercase tracking-wider font-medium">
              Route Breakdown
            </span>
          </div>
          <RouteBar
            label="Cache"
            count={stats.cacheHits}
            total={stats.totalQueries}
            color="bg-cache"
          />
          <RouteBar
            label="Local"
            count={stats.localResolutions}
            total={stats.totalQueries}
            color="bg-local"
          />
          <RouteBar
            label="Remote"
            count={stats.remoteFallbacks}
            total={stats.totalQueries}
            color="bg-remote"
          />
        </div>

        {/* Token savings calculator */}
        <div className="space-y-2 pt-2 border-t border-border/30">
          <div className="flex items-center gap-2">
            <TrendingDown className="w-3.5 h-3.5 text-foreground/40" aria-hidden="true" />
            <span className="text-xs text-foreground/40 uppercase tracking-wider font-medium">
              Token Usage
            </span>
          </div>
          <div className="space-y-1.5 text-sm">
            <div className="flex justify-between">
              <span className="text-foreground/60">Used</span>
              <span className="tabular-nums text-foreground/80 font-medium">
                {stats.totalTokensUsed.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-foreground/60">Est. remote</span>
              <span className="tabular-nums text-foreground/80 font-medium">
                {stats.estimatedRemoteTokens.toLocaleString()}
              </span>
            </div>

            {/* Visual savings bar */}
            {tokensSaved > 0 && (
              <div className="pt-1.5 space-y-1.5">
                <div className="h-2 bg-muted rounded-full overflow-hidden flex" role="progressbar" aria-valuenow={savingsPct} aria-valuemin={0} aria-valuemax={100} aria-label={`${savingsPct}% tokens saved vs all-remote`}>
                  <div
                    className="h-full bg-cache rounded-l-full transition-all duration-500"
                    style={{ width: `${savingsPct}%` }}
                  />
                  <div
                    className="h-full bg-remote/60 rounded-r-full transition-all duration-500"
                    style={{ width: `${100 - savingsPct}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1 text-cache">
                      <span className="w-1.5 h-1.5 rounded-full bg-cache" aria-hidden="true" />
                      Saved
                    </span>
                    <span className="flex items-center gap-1 text-remote">
                      <span className="w-1.5 h-1.5 rounded-full bg-remote/60" aria-hidden="true" />
                      Remote
                    </span>
                  </div>
                  <span className="tabular-nums text-cache font-semibold">
                    -{tokensSaved.toLocaleString()}
                  </span>
                </div>
              </div>
            )}

            {!tokensSaved && stats.totalQueries > 0 && (
              <p className="text-[11px] text-foreground/40 italic">
                All queries routed remotely so far — no tokens saved yet.
              </p>
            )}
          </div>
        </div>

        {/* Empty-state hint */}
        {stats.totalQueries === 0 && (
          <div className="text-center py-6">
            <p className="text-xs text-foreground/40 leading-relaxed">
              Send a message to see routing stats and token savings.
            </p>
          </div>
        )}
      </div>

      {/* Reset button */}
      <div className="p-4 border-t border-border/50">
        <button
          onClick={onReset}
          disabled={messagesCount === 0}
          className="w-full flex items-center justify-center gap-1.5 text-sm px-3 py-2 rounded-lg border border-border/30 text-foreground/60 hover:text-foreground hover:border-foreground/30 hover:bg-muted transition-all duration-150 disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
          aria-label="Reset session and clear all messages"
        >
          <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
          Reset Session
        </button>
      </div>
    </>
  );

  /* ----- Mobile overlay (when sidebarOpen is true) ----- */
  if (sidebarOpen) {
    return (
      <>
        {/* Scrim */}
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm animate-fade-in"
          onClick={onClose}
          aria-hidden="true"
        />

        {/* Sidebar panel (slide-in from right) */}
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="session-summary-title"
          className="fixed inset-y-0 right-0 z-50 w-full max-w-sm bg-background border-l border-border/50 flex flex-col shadow-2xl animate-slide-in-right"
        >
          {/* Header with close button */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/50 shrink-0">
            <h2 id="session-summary-title" className="text-sm font-semibold text-foreground/80">
              Session Summary
            </h2>
            <button
              ref={closeRef}
              onClick={onClose}
              className="text-foreground/50 hover:text-foreground/80 p-1 rounded-md hover:bg-muted transition-colors duration-150 cursor-pointer focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2"
              aria-label="Close session summary"
            >
              <X className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>

          {content}
        </div>
      </>
    );
  }

  /* ----- Desktop sidebar (always visible) ----- */
  return (
    <aside className="w-72 flex-shrink-0 border-l border-border/50 bg-background/95 backdrop-blur-sm hidden md:flex flex-col shadow-card">
      {content}
    </aside>
  );
}