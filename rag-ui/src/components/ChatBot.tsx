import { useEffect, useRef, useState } from 'react';
import { Streamdown } from 'streamdown';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
}


interface Citation {
  source: string;
  pages: string[];
  chunkId: string;
}

function extractCitations(content: string): Citation[] {
  const citations = new Map<string, Citation>();
  const pattern = /\\[Source:\\s*([^\\]]+?)(?:,\\s*Page:\\s*([^\\]]+?))?,\\s*Chunk ID:\\s*([^\\]]+)\\]/g;

  for (const match of content.matchAll(pattern)) {
    const source = match[1].trim();
    const pages = match[2]
      ? match[2].split(",").map((page) => page.trim()).filter(Boolean)
      : [];
    const chunkId = match[3].trim();
    const citation = { source, pages, chunkId };
    citations.set(`${source}|${pages.join(",")}|${chunkId}`, citation);
  }

  return [...citations.values()];
}

interface ChatStreamChunk {
  content?: string | null;
  tools?: unknown[] | null;
  error?: string;
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
    if (typeof chunk.error === 'string') throw new Error(chunk.error);
    return typeof chunk.content === 'string' ? chunk.content : null;
  }

  return null;
}

export default function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
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

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus('Please choose a PDF file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    setIsUploading(true);
    setUploadStatus('Uploading and indexing PDF…');

    try {
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });
      const result: { detail?: string; chunks_indexed?: number; message?: string } = await response.json();
      if (!response.ok) {
        throw new Error(result.detail ?? 'Upload failed (HTTP ' + response.status + ').');
      }
      setUploadStatus((result.message ?? 'PDF indexed.') + ' ' + (result.chunks_indexed ?? 0) + ' chunks added.');
    } catch (error) {
      setUploadStatus(error instanceof Error ? error.message : 'PDF upload failed.');
    } finally {
      setIsUploading(false);
    }
  };

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
        <div className="flex flex-wrap items-center gap-3">
          <label className="cursor-pointer rounded-md border border-primary-foreground/30 px-3 py-2 text-sm font-medium hover:bg-primary-foreground/10">
            {isUploading ? 'Indexing…' : 'Upload PDF'}
            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleUpload}
              disabled={isUploading || isLoading}
              className="sr-only"
            />
          </label>
          {uploadStatus && <p className="text-xs" role="status">{uploadStatus}</p>}
        </div>
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
              {message.content ? (
                message.role === 'assistant' ? (
                  <>
                    <Streamdown>{message.content}</Streamdown>
                    {extractCitations(message.content).length > 0 && (
                      <section
                        aria-label="Sources"
                        className="mt-4 border-t border-border/70 pt-3"
                      >
                        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Sources
                        </h3>
                        <ul className="flex flex-wrap gap-2">
                          {extractCitations(message.content).map((citation) => (
                            <li
                              key={`${citation.source}|${citation.pages.join(',')}|${citation.chunkId}`}
                              className="max-w-full rounded-md border border-border bg-background/70 px-3 py-2 text-xs"
                            >
                              <span className="block break-all font-medium text-foreground">
                                {citation.source}
                              </span>
                              <span className="mt-1 block text-muted-foreground">
                                {citation.pages.length > 0
                                  ? `Page ${citation.pages.join(', ')}`
                                  : 'Page unavailable'}
                                {' · '}
                                Chunk {citation.chunkId}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </section>
                    )}
                  </>
                ) : (
                  message.content
                )
              ) : (isLoading && message.role === 'assistant' ? 'Thinking…' : '')}
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
