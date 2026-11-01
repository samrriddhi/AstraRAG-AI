"""Agent implementation for AstraRAG AI."""

import json
import logging
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

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5

agent_dir = Path(__file__).parent
system_prompt_txt = agent_dir / "system_prompt.txt"
tools_json = agent_dir / "tools.json"


class Agent:
    """Agent coordinating chat completion, retrieval, and grounded answers."""

    system = (
        system_prompt_txt.read_text(encoding="utf-8")
        + """

SOURCE ATTRIBUTION RULES:

1. Answer document questions using retrieved document content.
2. Cite the filename and CHUNK_ID supplied in retrieved context.
3. Use this format: [Source: filename, Chunk ID: id].
4. Never invent filenames, chunk IDs, page numbers, or quotations.
5. If retrieved content does not answer the question, say that the indexed
   documents do not contain sufficient information.
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
        query_embedding = await self.embed.generate_embedding(text=query)
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
        query_embedding = await self.embed.generate_embedding(text=query)
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

    @staticmethod
    def _build_template(results: list[SearchResult]) -> str:
        """Format retrieved chunks with source metadata."""
        if not results:
            return "NO_RELEVANT_DOCUMENT_CONTENT_FOUND"

        formatted_results = []
        for index, result in enumerate(results, start=1):
            data = result.get("data", {})
            content = data.get("content", "")
            if not content:
                continue

            source = data.get("source", "Unknown source")
            chunk_id = data.get("chunk_id", "Unknown chunk ID")
            score = result.get("score", "Unavailable")
            formatted_results.append(
                f"<DOCUMENT_CHUNK number='{index}'>\\n"
                f"<SOURCE>{source}</SOURCE>\\n"
                f"<CHUNK_ID>{chunk_id}</CHUNK_ID>\\n"
                f"<RETRIEVAL_SCORE>{score}</RETRIEVAL_SCORE>\\n"
                f"<CONTENT>\\n{content}\\n</CONTENT>\\n"
                f"</DOCUMENT_CHUNK>"
            )

        if not formatted_results:
            return "NO_RELEVANT_DOCUMENT_CONTENT_FOUND"
        return "\\n\\n".join(formatted_results)

    async def execute(
        self,
        tool_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Execute a registered retrieval tool."""
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown retrieval tool: {tool_name}")
        return await self.tool_map[tool_name](*args, **kwargs)

    async def generate(
        self,
        messages: Messages,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate an answer, executing at most MAX_TOOL_ROUNDS tool cycles."""
        for round_number in range(MAX_TOOL_ROUNDS):
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

                tool_calls = chunk.get("tools")
                if not tool_calls:
                    continue

                assistant_message["tool_calls"] = tool_calls
                for tool in tool_calls:
                    tool_id = tool.get("id")
                    function = tool.get("function") or {}
                    tool_name = function.get("name")
                    raw_arguments = function.get("arguments", "{}")

                    if not tool_id or not tool_name:
                        raise ValueError("The model returned an incomplete tool call.")

                    try:
                        arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else raw_arguments
                        )
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"The model returned invalid JSON for tool {tool_name}."
                        ) from exc

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            f"Arguments for tool {tool_name} must be a JSON object."
                        )

                    result = await self.execute(tool_name=tool_name, **arguments)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }
                    )

            if "tool_calls" not in assistant_message:
                return

            messages.append(assistant_message)
            messages.extend(tool_messages)
            logger.debug(
                "Completed retrieval tool cycle %s with %s tool result(s).",
                round_number + 1,
                len(tool_messages),
            )

        raise RuntimeError(
            f"The model exceeded the maximum of {MAX_TOOL_ROUNDS} retrieval rounds."
        )
