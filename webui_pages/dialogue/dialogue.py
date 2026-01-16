import base64
import hashlib
import io
import os
import uuid
from datetime import datetime
from PIL import Image as PILImage
from typing import Dict, List
from urllib.parse import urlencode

# from audio_recorder_streamlit import audio_recorder
import openai
import streamlit as st
import streamlit_antd_components as sac
from streamlit_chatbox import *
from streamlit_extras.bottom_container import bottom
from streamlit_paste_button import paste_image_button

from settings import Settings
from server.callback_handler.agent_callback_handler import AgentStatus
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from server.knowledge_base.utils import format_reference
from server.utils import MsgType, get_config_models, get_config_platforms, get_default_llm
from webui_pages.utils import *

chat_box = ChatBox(assistant_avatar=get_img_base64("icon_blue_square_v2.png"))


def save_session(conv_name: str = None):
    """save session state to chat context"""
    chat_box.context_from_session(
        conv_name, exclude=["selected_page", "prompt", "cur_conv_name", "upload_image"]
    )


def restore_session(conv_name: str = None):
    """restore sesstion state from chat context"""
    chat_box.context_to_session(
        conv_name, exclude=["selected_page", "prompt", "cur_conv_name", "upload_image"]
    )


def rerun():
    """
    save chat context before rerun
    """
    save_session()
    st.rerun()


def get_messages_history(
        history_len: int, content_in_expander: bool = False
) -> List[Dict]:
    """
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    """

    def filter(msg):
        content = [
            x for x in msg["elements"] if x._output_method in ["markdown", "text"]
        ]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    messages = chat_box.filter_history(history_len=history_len, filter=filter)
    if sys_msg := chat_box.context.get("system_message"):
        messages = [{"role": "system", "content": sys_msg}] + messages

    return messages


@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    """
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    """
    return _api.upload_temp_docs(files).get("data", {}).get("id")


def add_conv(name: str = ""):
    conv_names = chat_box.get_chat_names()
    if not name:
        i = len(conv_names) + 1
        while True:
            name = f"会话{i}"
            if name not in conv_names:
                break
            i += 1
    if name in conv_names:
        sac.alert(
            "创建新会话出错",
            f"该会话名称 “{name}” 已存在",
            color="error",
            closable=True,
        )
    else:
        chat_box.use_chat_name(name)
        st.session_state["cur_conv_name"] = name


def del_conv(name: str = None):
    conv_names = chat_box.get_chat_names()
    name = name or chat_box.cur_chat_name

    if len(conv_names) == 1:
        sac.alert(
            "删除会话出错", f"这是最后一个会话，无法删除", color="error", closable=True
        )
    elif not name or name not in conv_names:
        sac.alert(
            "删除会话出错", f"无效的会话名称：“{name}”", color="error", closable=True
        )
    else:
        chat_box.del_chat_name(name)
        # restore_session()
    st.session_state["cur_conv_name"] = chat_box.cur_chat_name


def clear_conv(name: str = None):
    chat_box.reset_history(name=name or None)


def dialogue_page(
        api: ApiRequest,
        is_lite: bool = False,
):
    ctx = chat_box.context
    ctx.setdefault("uid", uuid.uuid4().hex)
    ctx.setdefault("file_chat_id", None)
    ctx.setdefault("llm_model", get_default_llm())
    ctx.setdefault("temperature", Settings.model_settings.TEMPERATURE)
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)

    # sac on_change callbacks not working since st>=1.34
    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name

    # st.write(chat_box.cur_chat_name)
    # st.write(st.session_state)
    # st.write(chat_box.context)

    @st.experimental_dialog("模型配置", width="large")
    def llm_model_setting():
        # 模型
        cols = st.columns(3)
        platforms = ["所有"] + list(get_config_platforms())
        platform = cols[0].selectbox("选择模型平台", platforms, key="platform")
        llm_models = list(
            get_config_models(
                model_type="llm", platform_name=None if platform == "所有" else platform
            )
        )
        llm_models += list(
            get_config_models(
                model_type="image2text", platform_name=None if platform == "所有" else platform
            )
        )
        llm_model = cols[1].selectbox("选择LLM模型", llm_models, key="llm_model")
        temperature = cols[2].slider("Temperature", 0.0, 1.0, key="temperature")
        system_message = st.text_area("System Message:", key="system_message")
        if st.button("OK"):
            rerun()

    @st.experimental_dialog("重命名会话")
    def rename_conversation():
        name = st.text_input("会话名称")
        if st.button("OK"):
            chat_box.change_chat_name(name)
            restore_session()
            st.session_state["cur_conv_name"] = name
            rerun()

    with st.sidebar:
        tab2 = st.tabs(["会话设置"])[0]

        with tab2:
            # 会话
            cols = st.columns(3)
            conv_names = chat_box.get_chat_names()

            def on_conv_change():
                print(conversation_name, st.session_state.cur_conv_name)
                save_session(conversation_name)
                restore_session(st.session_state.cur_conv_name)

            conversation_name = sac.buttons(
                conv_names,
                label="当前会话：",
                key="cur_conv_name",
                # on_change=on_conv_change, # not work
            )
            chat_box.use_chat_name(conversation_name)
            conversation_id = chat_box.context["uid"]
            if cols[0].button("新建", on_click=add_conv):
                ...
            if cols[1].button("重命名"):
                rename_conversation()
            if cols[2].button("删除", on_click=del_conv):
                ...

    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。"

    # TODO: 这里的内容有点奇怪，从后端导入Settings.model_settings.LLM_MODEL_CONFIG，然后又从前端传到后端。需要优化
    #  传入后端的内容
    llm_model_config = Settings.model_settings.LLM_MODEL_CONFIG
    chat_model_config = {key: {} for key in llm_model_config.keys()}
    for key in llm_model_config:
        if c := llm_model_config[key]:
            model = c.get("model", "").strip() or get_default_llm()
            chat_model_config[key][model] = llm_model_config[key]
    llm_model = ctx.get("llm_model")
    if llm_model is not None:
        chat_model_config["llm_model"][llm_model] = llm_model_config["llm_model"].get(
            llm_model, {}
        )

    # chat input
    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            widget_keys = ["platform", "llm_model", "temperature", "system_message"]
            chat_box.context_to_session(include=widget_keys)
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            chat_box.reset_history()
            rerun()
        # with cols[1]:
        #     mic_audio = audio_recorder("", icon_size="2x", key="mic_audio")
        prompt = cols[2].chat_input(chat_input_placeholder, key="prompt")
    if prompt:
        history = get_messages_history(
            chat_model_config["llm_model"]
            .get(next(iter(chat_model_config["llm_model"])), {})
            .get("history_len", 1)
        )


        chat_box.user_say(prompt)
        chat_box.ai_say("正在思考...")
        text = ""
        started = False

        client = openai.Client(base_url=f"{api_address()}/chat", api_key="NONE")
        messages = history + [{"role": "user", "content": prompt}]


        extra_body = dict(
            chat_model_config=chat_model_config,
            conversation_id=conversation_id,
        )
        stream = True
        params = dict(
            messages=messages,
            model=llm_model,
            stream=stream,  # TODO：xinference qwen-vl-chat 流式输出会出错，后续看更新
            extra_body=extra_body,
        )

        if Settings.model_settings.MAX_TOKENS:
            params["max_tokens"] = Settings.model_settings.MAX_TOKENS

        if stream:
            try:
                for d in client.chat.completions.create(**params):
                    # import rich
                    # rich.print(d)
                    message_id = d.message_id
                    metadata = {
                        "message_id": message_id,
                    }

                    # clear initial message
                    if not started:
                        chat_box.update_msg("", streaming=False)
                        started = True

                    if d.status == AgentStatus.error:
                        st.error(d.choices[0].delta.content)
                    elif d.status == AgentStatus.llm_start:
                        continue
                    elif d.status == AgentStatus.llm_new_token:
                        continue
                    elif d.status == AgentStatus.llm_end:
                        continue
                    elif d.status is None:  # not agent chat
                        if getattr(d, "is_ref", False):
                            context = str(d.tool_output)
                            if isinstance(d.tool_output, dict):
                                docs = d.tool_output.get("docs", [])
                                source_documents = format_reference(kb_name=d.tool_output.get("knowledge_base"),
                                                                    docs=docs,
                                                                    api_base_url=api_address(is_public=True))
                                context = "\n".join(source_documents)

                            chat_box.insert_msg(
                                Markdown(
                                    context,
                                    in_expander=True,
                                    state="complete",
                                    title="参考资料",
                                )
                            )
                            chat_box.insert_msg("")
                        elif getattr(d, "tool_call", None) == "text2images":  # TODO：特定工具特别处理，需要更通用的处理方式
                            for img in d.tool_output.get("images", []):
                                chat_box.insert_msg(Image(f"{api.base_url}/media/{img}"), pos=-2)
                        else:
                            text += d.choices[0].delta.content or ""
                            chat_box.update_msg(
                                text.replace("\n", "\n\n"), streaming=True, metadata=metadata
                            )
                    chat_box.update_msg(text, streaming=False, metadata=metadata)
            except Exception as e:
                st.error(e.body)
        else:
            try:
                d = client.chat.completions.create(**params)
                chat_box.update_msg(d.choices[0].message.content or "", streaming=False)
            except Exception as e:
                st.error(e.body)


    now = datetime.now()
    with tab2:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # st.write(chat_box.history)
