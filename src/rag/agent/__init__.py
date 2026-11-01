
"""Agent implementation for AstraRAG AI."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from rag import config
from rag.types import (
    AssistantMessage,
    Messages,
    OptionalChat,
    OptionalEmbed,
    OptionalSearch,
    SearchResult,
)

__all__ = ["Agent"]

agent_dir = Path(__file__).parent

system_prompt_txt = agent_dir / "system_prompt.txt"
tools_json = agent_dir / "tools.json"


class Agent:
    """Agent coordinating Groq, embeddings, and Qdrant."""

    system = (
        system_prompt_txt.read_text(encoding="utf-8")
        + """

SOURCE ATTRIBUTION RULES:

1. Answer document questions using retrieved document content.
2. Cite the filename and CHUNK_ID supplied in retrieved context.
3. Use this format: [Source: filename, Chunk ID: id].
4. Never invent filenames, chunk IDs, page numbers, or quotations.
5. If the retrieved content does not answer the question,
   clearly state that the available documents do not contain
   sufficient information.
6. Distinguish document facts from interpretations.
7. Do not present external knowledge as document evidence.
"""
    )

    tools = json.loads(tools_json.read_text(encoding="utf-8"))

    def __init__(
        self,
        model: str,
        _chat: OptionalChat = None,
        _search: OptionalSearch = None,
        _embed: OptionalEmbed = None,
    ) -> None:
        self.model = model
        self.chat = _chat or config.get_openai_chat()
        self.search = _search or config.get_qdrant()
        self.embed = _embed or config.get_openai_embed()

        self.tool_map = {
            "hybrid_search": self._hybrid_search_pipeline,
            "semantic_search": self._semantic_search_pipeline,
            "keyword_search": self._keyword_search_pipeline,
        }

    async def _hybrid_search_pipeline(
        self,
        query: str,
        keywords: list[str],
        limit: int = 5,
    ) -> str:
        """Perform dense and sparse hybrid retrieval."""

        print("\n[DEBUG] Starting hybrid search")
        print("[DEBUG] Query:", query)

        query_embedding = await self.embed.generate_embedding(
            text=query
        )

        results = await self.search.hybrid_search(
            query=query_embedding,
            keywords=keywords,
            limit=limit,
        )

        return self._build_template(results)

    async def _semantic_search_pipeline(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Perform semantic retrieval."""

        query_embedding = await self.embed.generate_embedding(
            text=query
        )

        results = await self.search.semantic_search(
            query=query_embedding,
            limit=limit,
        )

        return self._build_template(results)

    async def _keyword_search_pipeline(
        self,
        keywords: list[str],
        limit: int = 5,
    ) -> str:
        """Perform keyword retrieval."""

        results = await self.search.keyword_search(
            keywords=keywords,
            limit=limit,
        )

        return self._build_template(results)

    def _build_template(
        self,
        results: list[SearchResult],
    ) -> str:
        """Format retrieved chunks with source metadata."""

        if not results:
            print("[DEBUG] No relevant chunks found")
            return "NO_RELEVANT_DOCUMENT_CONTENT_FOUND"

        formatted_results = []

        for index, result in enumerate(results, start=1):
            data = result.get("data", {})

            content = data.get("content", "")
            source = data.get("source", "Unknown source")
            chunk_id = data.get("chunk_id", "Unknown chunk ID")
            score = result.get("score", "Unavailable")

            formatted_results.append(
                f"<DOCUMENT_CHUNK number='{index}'>\n"
                f"<SOURCE>{source}</SOURCE>\n"
                f"<CHUNK_ID>{chunk_id}</CHUNK_ID>\n"
                f"<RETRIEVAL_SCORE>{score}</RETRIEVAL_SCORE>\n"
                f"<CONTENT>\n{content}\n</CONTENT>\n"
                f"</DOCUMENT_CHUNK>"
            )

        print(f"[DEBUG] Retrieved {len(results)} chunks")

        return "\n\n".join(formatted_results)

    async def execute(
        self,
        tool_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Execute a registered retrieval tool."""

        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown retrieval tool: {tool_name}")

        return await self.tool_map[tool_name](
            *args,
            **kwargs,
        )

    async def generate(
        self,
        messages: Messages,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate answers and execute retrieval tools."""

        while True:
            assistant_message: AssistantMessage = {
                "role": "assistant",
                "content": "",
            }

            tool_messages: list[dict[str, Any]] = []

            async for chunk in self.chat.generate_stream(
                messages,
                model=self.model,
                tools=self.tools,
                **kwargs,
            ):
                if content := chunk.get("content"):
                    assistant_message["content"] += content
                    yield content

                if tools := chunk.get("tools"):
                    print("\n[DEBUG] GROQ TOOL CALLS RECEIVED")

                    assistant_message["tool_calls"] = tools

                    for tool in tools:
                        tool_name = tool["function"]["name"]

                        arguments = json.loads(
                            tool["function"]["arguments"]
                        )

                        print("[DEBUG] Tool:", tool_name)
                        print("[DEBUG] Arguments:", arguments)

                        result = await self.execute(
                            tool_name=tool_name,
                            **arguments,
                        )

                        print("\n[DEBUG] RETRIEVAL RESULT:")
                        print(result[:1500])

                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool["id"],
                                "content": result,
                            }
                        )

            if "tool_calls" not in assistant_message:
                break

            messages.append(assistant_message)
            messages.extend(tool_messages)

            print(
                f"[DEBUG] Added {len(tool_messages)} tool results. "
                "Requesting final answer from Groq."
            )
