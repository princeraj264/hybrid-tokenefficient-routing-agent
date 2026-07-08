import { Message } from '../types';
import RoutingCard from './RoutingCard';

interface MessageBubbleProps {
  message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center px-4 py-2 animate-fade-in-up">
        <div className={`text-sm px-4 py-2 rounded-lg border ${
          message.error
            ? 'bg-destructive/10 border-destructive/30 text-destructive'
            : 'bg-muted/50 border-border/50 text-foreground/60'
        }`}>
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} px-4 py-1.5 animate-fade-in-up`}>
      <div className={`max-w-[80%] space-y-1.5 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Message bubble */}
        <div
          className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? 'bg-primary text-on-primary rounded-br-md'
              : 'bg-secondary text-foreground rounded-bl-md border border-border/30'
          }`}
        >
          {message.content}
        </div>

        {/* Routing detail card */}
        {message.routing && (
          <RoutingCard
            path={message.routing.path}
            confidence={message.routing.confidence}
            tokensUsed={message.routing.tokensUsed}
            latencyMs={message.routing.latencyMs}
          />
        )}

        {/* Timestamp */}
        <p className="text-[10px] text-foreground/30 px-1 tabular-nums">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
    </div>
  );
}