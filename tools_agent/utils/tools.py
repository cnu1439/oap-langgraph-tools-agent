from typing import Annotated
from langchain_core.tools import StructuredTool, ToolException, tool
import aiohttp
import re
from langchain_tavily import TavilySearch


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap the tool coroutine to handle `interaction_required` MCP error.

    Tried to obtain the URL from the error, which the LLM can use to render a link."""

    old_coroutine = tool.coroutine

    async def wrapped_mcp_coroutine(**kwargs):
        try:
            response = await old_coroutine(**kwargs)
            return response
        except Exception as e:
            if "TaskGroup" in str(e) and hasattr(e, "__context__"):
                sub_exception = e.__context__
                if hasattr(sub_exception, "error"):
                    e = sub_exception

            if hasattr(e, "error") and hasattr(e.error, "code") and e.error.code == -32003 and hasattr(e.error, "data"):
                error_message = (((e.error.data or {}).get("message") or {}).get("text")) or "Required interaction"

                if url := (e.error.data or {}).get("url"):
                    error_message += f": {url}"

                raise ToolException(error_message)
            raise e

    tool.coroutine = wrapped_mcp_coroutine
    return tool


async def create_rag_tool(rag_url: str, collection_id: str, access_token: str):
    """Create a RAG tool for a specific collection.

    Args:
        rag_url: The base URL for the RAG API server
        collection_id: The ID of the collection to query
        access_token: The access token for authentication

    Returns:
        A structured tool that can be used to query the RAG collection
    """
    if rag_url.endswith("/"):
        rag_url = rag_url[:-1]

    collection_endpoint = f"{rag_url}/collections/{collection_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                collection_endpoint, headers={"Authorization": f"Bearer {access_token}"}
            ) as response:
                response.raise_for_status()
                collection_data = await response.json()

        # Get the collection name and sanitize it to match the required regex pattern
        raw_collection_name = collection_data.get("name", f"collection_{collection_id}")

        # Sanitize the name to only include alphanumeric characters, underscores, and hyphens
        # Replace any other characters with underscores
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_collection_name)

        # Ensure the name is not empty and doesn't exceed 64 characters
        if not sanitized_name:
            sanitized_name = f"collection_{collection_id}"
        collection_name = sanitized_name[:64]

        raw_description = collection_data.get("metadata", {}).get("description")

        if not raw_description:
            collection_description = (
                "Search your collection of documents for results semantically similar to the input query"
            )
        else:
            collection_description = f"Search your collection of documents for results semantically similar to the input query. Collection description: {raw_description}"

        @tool(name_or_callable=collection_name, description=collection_description)
        async def get_documents(
            query: Annotated[str, "The search query to find relevant documents"],
        ) -> str:
            """Search for documents in the collection based on the query"""

            search_endpoint = f"{rag_url}/collections/{collection_id}/documents/search"
            payload = {"query": query, "limit": 10}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        search_endpoint,
                        json=payload,
                        headers={"Authorization": f"Bearer {access_token}"},
                    ) as search_response:
                        search_response.raise_for_status()
                        documents = await search_response.json()

                formatted_docs = "<all-documents>\n"

                for doc in documents:
                    doc_id = doc.get("id", "unknown")
                    content = doc.get("page_content", "")
                    formatted_docs += f'  <document id="{doc_id}">\n    {content}\n  </document>\n'

                formatted_docs += "</all-documents>"
                return formatted_docs
            except Exception as e:
                return f"<all-documents>\n  <error>{str(e)}</error>\n</all-documents>"

        return get_documents

    except Exception as e:
        raise Exception(f"Failed to create RAG tool: {str(e)}")


def get_economics_search_tool():
    """Create a search tool for economics-related queries.

    Returns:
        A structured tool that can be used to search for economics-related information
    """
    search_tool = TavilySearch(
        name="Macro Economics Search",
        description="""A search engine optimized for comprehensive, accurate, and trusted results. Useful for when 
        you need to answer questions related macro economics utterances (inflation, gdp, unemployment, interest rates).
        It not only retrieves URLs and snippets, but offers advanced search depths, domain management, time range filters, 
        and image search, this tool delivers real-time, accurate, and citation-backed results.Input should be a search query.""",
        topic="general",
    )
    return search_tool
