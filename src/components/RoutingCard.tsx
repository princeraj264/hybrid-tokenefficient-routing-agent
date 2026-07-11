import { useState } from 'react';
import { Zap, Cpu, Globe, ChevronDown } from 'lucide-react';
import type { RoutePath } from '../types';

export interface RoutingCardProps {
  path: RoutePath;
  confidence: number;
  tokensUsed: number;
  latencyMs: number;
  modelUsed?: string;
}

const routeMeta: Record<RoutePath, { label: string; icon: typeof Zap; colorClass: string }> = {
  cache: {
    label: 'Cache',
    icon: Zap,
    colorClass: 'text-cache border-cache/30 bg-cache/10',
  },
  local: {
    label: 'Local',
    icon: Cpu,
    colorClass: 'text-local border-local/30 bg-local/10',
  },
  remote: {
    label: 'Remote',
    icon: Globe,
    colorClass: 'text-remote border-remote/30 bg-remote/10',
  },
};

/** Fallback for unexpected path values from the backend. */
const fallbackMeta = {
  label: 'Route',
  icon: Globe,
  colorClass: 'text-foreground/50 border-border/30 bg-muted/50',
};

function routeDescription(path: RoutePath, model?: string): string {
  if (path === 'cache') return 'Response served from the semantic cache — instant, zero token cost.';
  if (path === 'local') return 'Resolved locally via Gemma 2B on ROCm — low latency, no API cost.';
  if (model) return `Fell back to ${model} via Fireworks API — highest quality, highest cost.`;
  return 'Fell back to Fireworks API — highest quality, highest cost.';
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);

  // Gradient shifts from green→amber→red as confidence drops
  const hue = Math.round(confidence * 120); // 120 (green) → 0 (red)
  const barColor = `hsl(${hue}, 72%, 48%)`;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-foreground/50 uppercase tracking-wider font-medium">
          Confidence
        </span>
        <span className="text-xs font-medium tabular-nums text-foreground/80">
          {pct}%
        </span>
      </div>
      <div
        className="h-2 bg-muted rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence: ${pct}%`}
      >
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

export default function RoutingCard({
  path,
  confidence,
  tokensUsed,
  latencyMs,
  modelUsed,
}: RoutingCardProps) {
  const [expanded, setExpanded] = useState(false);
  const meta = routeMeta[path] ?? fallbackMeta;

  return (
    <div className="bg-muted/60 border border-border/30 rounded-xl overflow-hidden transition-all duration-200 shadow-card">
      {/* Summary row — always visible, clickable to toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2 text-xs cursor-pointer hover:bg-muted/80 transition-colors duration-150 active:scale-[0.98]"
        aria-expanded={expanded}
        aria-controls="routing-card-detail"
      >
        <div className="flex items-center gap-2 min-w-0">
          {/* Route badge */}
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${meta.colorClass} shrink-0`}
          >
            <meta.icon className="w-3 h-3" aria-hidden="true" />
            {meta.label}
          </span>

          {/* Model name (remote only) */}
          {path === 'remote' && modelUsed && (
            <span className="text-[11px] text-foreground/50 font-mono truncate max-w-[160px] hidden sm:inline">
              {modelUsed}
            </span>
          )}

          {/* Summary stats */}
          <span className="text-foreground/40 hidden sm:inline">·</span>
          <span className="text-foreground/50 tabular-nums hidden sm:inline">
            {tokensUsed} tokens
          </span>
          <span className="text-foreground/40 hidden sm:inline">·</span>
          <span className="text-foreground/50 tabular-nums hidden sm:inline">
            {latencyMs}ms
          </span>
        </div>

        <ChevronDown
          className={`w-3.5 h-3.5 text-foreground/40 transition-transform duration-200 shrink-0 ${
            expanded ? 'rotate-180' : ''
          }`}
          aria-hidden="true"
        />
      </button>

      {/* Expanded detail */}
      <div
        id="routing-card-detail"
        className={`grid transition-all duration-200 ease-out ${
          expanded
            ? 'grid-rows-[1fr] opacity-100'
            : 'grid-rows-[0fr] opacity-0'
        }`}
      >
        <div className="overflow-hidden">
          <div className="px-3 pb-3 space-y-2.5 border-t border-border/20 pt-2.5">
            <ConfidenceBar confidence={confidence} />

            {/* Stats row with model name */}
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-xs text-foreground/60">
                <span className="tabular-nums">{tokensUsed} tokens</span>
                <span className="text-foreground/30">·</span>
                <span className="tabular-nums">{latencyMs}ms</span>
                {modelUsed && (
                  <>
                    <span className="text-foreground/30">·</span>
                    <span className="font-mono text-[11px] text-foreground/50 truncate max-w-[200px]" title={modelUsed}>
                      {modelUsed}
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Route description */}
            <p className="text-[11px] text-foreground/40 leading-relaxed">
              {routeDescription(path, modelUsed)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}