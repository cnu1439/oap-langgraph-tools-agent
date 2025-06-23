import logging
import os
from langchain_core.runnables import RunnableConfig
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from tools_agent.utils.tools import create_rag_tool
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools_agent.utils.tools import wrap_mcp_authenticate_tool, get_economics_search_tool, get_text_to_sql_tool


UNEDITABLE_SYSTEM_PROMPT = "\nIf the tool throws an error requiring authentication, provide the user with a Markdown link to the authentication page and prompt them to authenticate."

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that has access to a variety of tools."


MCP_API_KEY = os.getenv("MCP_API_KEY")


class RagConfig(BaseModel):
    rag_url: Optional[str] = None
    """The URL of the rag server"""
    collections: Optional[List[str]] = None
    """The collections to use for rag"""


class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )  # type: ignore
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )  # type: ignore
    """The tools to make available to the LLM"""


class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4.1",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                ],
            }
        },
    )  # type: ignore
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.0,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )  # type: ignore
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )  # type: ignore
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": f"The system prompt to use in all generations. The following prompt will always be included at the end of the system prompt:\n---{UNEDITABLE_SYSTEM_PROMPT}\n---",
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )  # type: ignore
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                # "default": {
                #     "url" : "https://ht-clear-tension-76-c67365b4a7dc5f3098eb8c5aa4445747.us.langgraph.app",
                #     "tools": ["stock_forecast_agent"]
                # }
            }
        },
    )  # type: ignore
    rag: Optional[RagConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "rag",
                # Here is where you would set the default collection. Use collection IDs
                # "default": {
                #     "rag_url": "http://74.224.99.144:8080",
                #     "collections": [
                #         "7bf3d4c7-5c44-4e90-9b9d-7d85cca69162",
                #     ]
                # },
            }
        },
    )  # type: ignore


async def graph(config: RunnableConfig):
    cfg = GraphConfigPydantic(**config.get("configurable", {}))
    tools = [get_economics_search_tool()]

    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if cfg.rag and cfg.rag.rag_url and cfg.rag.collections and supabase_token:
        for collection in cfg.rag.collections:
            rag_tool = await create_rag_tool(cfg.rag.rag_url, collection, supabase_token)
            tools.append(rag_tool)

    logging.info("MCP config: %s", cfg.mcp_config)
    # logging.info("MCP tools: %s", cfg.mcp_config.tools)
    # logging.info("MCP URL: %s", cfg.mcp_config.url)

    if cfg.mcp_config and cfg.mcp_config.url and cfg.mcp_config.tools:
        mcp_client = MultiServerMCPClient(
            connections={
                "mcp_server": {
                    "transport": "streamable_http",
                    "url": cfg.mcp_config.url.rstrip("/") + "/mcp",
                    "headers": {"X-Api-Key": f"{MCP_API_KEY}"},
                }
            }
        )
        tools.extend(
            [
                wrap_mcp_authenticate_tool(tool)
                for tool in await mcp_client.get_tools()
                if tool.name in cfg.mcp_config.tools
            ]
        )

    model = init_chat_model(
        cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    # Add text to SQL tools
    sql_agent_tool = get_text_to_sql_tool(model)
    tools.append(sql_agent_tool)

    # print all the tools
    logging.info(f"Tools: {tools}")

    # print all the tools names
    logging.info(f"Tool names: {[tool.name for tool in tools]}")

    return create_react_agent(
        prompt=cfg.system_prompt + UNEDITABLE_SYSTEM_PROMPT,
        model=model,
        tools=tools,
        config_schema=GraphConfigPydantic,
    )
