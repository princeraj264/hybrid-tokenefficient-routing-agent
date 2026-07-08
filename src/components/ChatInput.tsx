import { useState, useRef, KeyboardEvent } from 'react';
import { Send, Loader2, AlertTriangle, RotateCcw } from 'lucide-react';

interface ChatInputProps {
  onSend: (query: string) => void;
  onRetry: () => void;
  disabled?: boolean;
  failedQuery: string | null;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  onRetry,
  disabled,
  failedQuery,
  placeholder,
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  };

  return (
    <div className="border-t border-border/50 bg-background/95 backdrop-blur-sm">
      {/* Error banner — shown when the last query failed */}
      {failedQuery && !disabled && (
        <div className="mx-4 mt-3 flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm animate-fade-in">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span className="truncate">
              Request failed — check the connection and try again.
            </span>
          </div>
          <button
            onClick={onRetry}
            className="flex items-center gap-1.5 shrink-0 px-2.5 py-1 rounded-md bg-destructive/15 hover:bg-destructive/25 text-destructive font-medium text-xs transition-colors duration-150 cursor-pointer focus-visible:outline-2 focus-visible:outline-destructive focus-visible:outline-offset-2"
            aria-label="Retry failed query"
          >
            <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
            Retry
          </button>
        </div>
      )}

      <div className="flex items-end gap-2 px-4 py-3 max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder={placeholder ?? 'Ask the routing agent something\u2026'}
          disabled={disabled}
          rows={1}
          className="flex-1 bg-secondary text-foreground placeholder:text-foreground/30 rounded-xl px-4 py-2.5 text-sm resize-none outline-none border border-border/30 focus:border-accent/50 focus:ring-1 focus:ring-accent/20 transition-all duration-200 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="flex-shrink-0 w-10 h-10 flex items-center justify-center rounded-xl bg-accent text-white hover:opacity-90 active:scale-95 disabled:opacity-30 disabled:active:scale-100 transition-all duration-150 cursor-pointer"
          aria-label="Send message"
        >
          {disabled ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
}