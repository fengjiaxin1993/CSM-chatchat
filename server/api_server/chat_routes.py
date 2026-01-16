from __future__ import annotations

from typing import Dict, List

from server.chat.similar_mem_chat import similar_mem_chat
from fastapi import APIRouter, Request
from langchain.prompts.prompt import PromptTemplate
from sse_starlette import EventSourceResponse

from server.api_server.api_schemas import OpenAIChatInput
from server.chat.chat import chat
from server.chat.kb_chat import kb_chat
from server.db.repository import add_message_to_db, add_conversation_to_db
from server.utils import (
    get_OpenAIClient,
    get_prompt_template
)
from settings import Settings
from utils import build_logger
from .openai_routes import openai_request, OpenAIChatOutput
from ..chat.mem_chat import mem_chat

logger = build_logger()

chat_router = APIRouter(prefix="/chat", tags=["ChatChat 对话"])

chat_router.post(
    "/chat",
    summary="与llm模型对话(通过LLMChain)",
)(chat)

chat_router.post(
    "/mem_chat",
    summary="与llm模型对话带记忆功能(通过LLMChain)",
)(mem_chat)

chat_router.post(
    "/similar_mem_chat",
    summary="与llm模型对话带记忆功能,similar(通过LLMChain)",
)(similar_mem_chat)

chat_router.post("/kb_chat", summary="知识库对话")(kb_chat)


@chat_router.post("/chat/completions", summary="兼容 openai 的统一 chat 接口")
async def chat_completions(
        request: Request,
        body: OpenAIChatInput,
) -> Dict:
    """
    请求参数与 openai.chat.completions.create 一致，可以通过 extra_body 传入额外参数
    tools 和 tool_choice 可以直接传工具名称，会根据项目里包含的 tools 进行转换
    通过不同的参数组合调用不同的 chat 功能：
    - tool_choice
        - extra_body 中包含 tool_input: 直接调用 tool_choice(tool_input)
        - extra_body 中不包含 tool_input: 通过 agent 调用 tool_choice
    - tools: agent 对话
    - 其它：LLM 对话
    以后还要考虑其它的组合（如文件对话）
    返回与 openai 兼容的 Dict
    """
    # import rich
    # rich.print(body)

    # 当调用本接口且 body 中没有传入 "max_tokens" 参数时, 默认使用配置中定义的值
    if body.max_tokens == None:
        body.max_tokens = Settings.model_settings.MAX_TOKENS

    client = get_OpenAIClient(model_name=body.model, is_async=True)
    extra = {**body.model_extra} or {}
    for key in list(extra):
        delattr(body, key)

    conversation_id = extra.get("conversation_id")

    # chat based on result from one choiced tool

    try:  # query is complex object that unable add to db when using qwen-vl-chat
        message_id = (
            add_conversation_to_db(
                chat_type="llm_chat",
                query=body.messages[-1]["content"],
                conversation_id=conversation_id,
            )
            if conversation_id
            else None
        )
    except Exception as e:
        logger.error(f"failed to add message to db")
        message_id = None

    extra_json = {
        "message_id": message_id,
        "status": None,
    }
    return await openai_request(
        client.chat.completions.create, body, extra_json=extra_json
    )
