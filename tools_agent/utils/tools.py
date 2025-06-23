import os
import re
import tempfile
import aiohttp

from typing import Annotated, Any, cast
from langchain_core.tools import StructuredTool, ToolException, tool
from langchain_tavily import TavilySearch
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage


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

    name = "macro_economics_search"
    description = "A search engine optimized for comprehensive, accurate, and trusted results. \
Useful for when you need to answer questions related macro economics utterances \
(inflation, gdp, unemployment, interest rates). It not only retrieves URLs and snippets, \
but offers advanced search depths, domain management, time range filters, and image search, \
this tool delivers real-time, accurate, and citation-backed results.Input should be a search query."

    @tool(name_or_callable=name, description=description)
    async def search(query: Annotated[str, "The search query to find relevant information"]):
        """Search for information related to macro economics."""

        wrapped = TavilySearch(max_results=3)
        return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

    return search


def get_text_to_sql_tool(llm):
    """Create a set of tools for text-to-SQL conversion.

    Args:
        database_url: The URL of the database to connect to
        llm: The language model to use for SQL generation

    Returns:
        A list of structured tools for text-to-SQL conversion
    """
    from .db_init import DatabaseInitializer

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_file_path = tfile.name
    tfile.close()

    db_initializer = DatabaseInitializer(db_path=db_file_path)

    data_file_path = os.path.join(os.path.dirname(__file__), "data", "hr_database.csv")
    db_initializer.create_database_from_csv(csv_file_path=data_file_path, table_name="employees")

    database_url = f"sqlite:///{db_file_path}"
    db = SQLDatabase.from_uri(database_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
        dialect=db.dialect,
        top_k=5,
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )

    name = "text_to_sql_toolkit"
    description = "Tool for converting text queries into SQL queries. and return the results from the database. \
    Useful for when you need to answer questions related to the HR database. \
    tool has access to the HR database, which contains employee information such as Employee ID,Name,Department,Role,\
    Start Date,Employee Status,FTE,Salary,Manager ID,Location,Performance Rating,Training Hours (YTD),Years of Service,\
    Retention Risk,Succession Plan Ready,Job Level,Education Level,Remote Work Eligible,Previous Company Experience, \
    Hiring Source,Recruitment Cost,Time to Hire (Days),Onboarding Status,Exit Date,Reason for Exit, \
    New Hires in Dept (Qtr),Voluntary Turnover Rate (Dept),Avg. Salary Increase (%),Diversity Group, \
    Skill Set 1,Skill Set 2,Skill Set 3,Project Allocation (%),Last Promotion Date \
    It can answer questions like 'What is the average salary of employees in the Sales department? \
    Input should be a natural language query about the HR database."

    @tool(name_or_callable=name, description=description)
    async def db_search(query: Annotated[str, "The search query to find relevant information"]):
        """Search for information related to macro economics."""

        results = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        messages = results.get("messages", [])
        if not messages:
            raise ToolException("No messages returned from the agent.")
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ToolException("Last message is not an AIMessage.")

        response_content = last_message.content
        if not response_content:
            raise ToolException("No content in the last AIMessage.")

        return response_content

    return db_search
