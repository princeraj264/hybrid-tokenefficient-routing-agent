import { useEffect, useRef } from 'react';
import { Message } from '../types';
import MessageBubble from './MessageBubble';
import { MessageSquare } from 'lucide-react';

interface ChatMessagesProps {
  messages: Message[];
  loading?: boolean;
}

export default function ChatMessages({ messages, loading }: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex items-center justify-center px-4">
        <div className="text-center space-y-3 max-w-sm">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-muted border border-border/30">
            <MessageSquare className="w-6 h-6 text-foreground/40" />
          </div>
          <h2 className="text-lg font-heading font-semibold text-foreground/80">
            Hybrid Routing Agent
          </h2>
          <p className="text-sm text-foreground/50 leading-relaxed">
            Queries are routed through three tiers — cache, local Gemma on ROCm, and remote Fireworks API — to minimize token cost. Type something to get started.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto scrollable py-4">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {loading && (
        <div className="flex justify-start px-4 py-1.5">
          <div className="bg-secondary text-foreground border border-border/30 rounded-2xl rounded-bl-md px-4 py-2.5 text-sm">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-foreground/40 animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-foreground/40 animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-foreground/40 animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}