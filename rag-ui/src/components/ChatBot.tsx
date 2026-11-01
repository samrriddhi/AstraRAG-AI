import { useEffect, useRef, useState } from 'react';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
}

interface ChatStreamChunk {
  content?: string | null;
  tools?: unknown[] | null;
}

function parseSseEvent(event: string): string | null {
  const data = event
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');

  if (!data || data === '[DONE]') return null;

  const parsed: unknown = JSON.parse(data);
  if (typeof parsed === 'string') return parsed;

  if (
    typeof parsed === 'object' &&
    parsed !== null &&
    'content' in parsed
  ) {
    const chunk = parsed as ChatStreamChunk;
    return typeof chunk.content === 'string' ? chunk.content : null;
  }

  return null;
}

export default function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [modelError, setModelError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function fetchModels() {
      try {
        const response = await fetch('/api/models', {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Model discovery failed (HTTP ${response.status}).`);
        }
        const data: { data?: string[] } = await response.json();
        const availableModels = Array.isArray(data.data) ? data.data : [];
        setModels(availableModels);
        setSelectedModel(availableModels[0] ?? '');
        if (availableModels.length === 0) {
          setModelError('No chat model is configured on the backend.');
        }
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') return;
        setModelError(
          error instanceof Error ? error.message : 'Could not load chat models.',
        );
      }
    }

    void fetchModels();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSendMessage = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || isLoading || !selectedModel) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: text,
      role: 'user',
    };
    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      content: '',
      role: 'assistant',
    };
    const conversation = [...messages, userMessage];

    setMessages([...conversation, assistantMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat?stream=true', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          messages: conversation.map(({ role, content }) => ({ role, content })),
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Chat request failed (HTTP ${response.status}).`);
      }
      if (!response.body) throw new Error('The backend returned no response stream.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const appendContent = (content: string) => {
        if (!content) return;
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessage.id
              ? { ...message, content: message.content + content }
              : message,
          ),
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });

        let boundary = buffer.search(/\r?\n\r?\n/);
        while (boundary !== -1) {
          const event = buffer.slice(0, boundary);
          const separator = buffer.match(/\r?\n\r?\n/);
          const separatorLength = separator?.[0].length ?? 2;
          buffer = buffer.slice(boundary + separatorLength);
          try {
            const content = parseSseEvent(event);
            if (content) appendContent(content);
          } catch {
            throw new Error('The backend sent an invalid streaming event.');
          }
          boundary = buffer.search(/\r?\n\r?\n/);
        }

        if (done) break;
      }

      if (buffer.trim()) {
        const content = parseSseEvent(buffer);
        if (content) appendContent(content);
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unexpected chat error.';
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantMessage.id
            ? { ...item, content: `Error: ${message}` }
            : item,
        ),
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col overflow-hidden rounded-lg border border-border bg-card shadow-lg">
      <header className="space-y-3 border-b border-border bg-primary px-6 py-4 text-primary-foreground">
        <h1 className="text-2xl font-semibold">AstraRAG AI</h1>
        <div className="flex items-center gap-3">
          <label htmlFor="model-select" className="text-sm font-medium">
            Model
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            disabled={isLoading || models.length === 0}
            className="min-w-0 flex-1 rounded bg-primary-foreground px-3 py-2 text-sm text-primary disabled:opacity-50"
          >
            {models.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>
        {modelError && <p className="text-sm" role="alert">{modelError}</p>}
      </header>

      <main className="flex-1 space-y-3 overflow-y-auto px-6 py-4">
        {messages.length === 0 && (
          <div className="mx-auto mt-16 max-w-md text-center">
            <h2 className="text-xl font-semibold">Ask your documents</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Ask a question about the documents indexed in your AstraRAG knowledge base.
            </p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] whitespace-pre-wrap break-words rounded-lg px-4 py-3 text-sm leading-relaxed ${
                message.role === 'user'
                  ? 'rounded-br-none bg-primary text-primary-foreground'
                  : 'rounded-bl-none bg-muted text-foreground'
              }`}
            >
              {message.content || (isLoading && message.role === 'assistant' ? 'Thinking…' : '')}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </main>

      <form
        onSubmit={handleSendMessage}
        className="flex gap-2 border-t border-border bg-card px-6 py-4"
      >
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={isLoading || !selectedModel}
          className="min-w-0 flex-1 rounded-full border border-border bg-input px-4 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim() || !selectedModel}
          className="rounded-full bg-primary px-6 py-2 font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
