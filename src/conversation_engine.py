import os
import json
from datetime import datetime
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.tools import FunctionTool
from src.global_settings import INDEX_STORAGE, CONVERSATION_FILE,SCORES_FILE
from src.prompts import CUSTORM_AGENT_SYSTEM_TEMPLATE

def load_chat_store():
    if os.path.exists(CONVERSATION_FILE) and os.path.getsize(CONVERSATION_FILE) > 0:
        try:
            chat_store = SimpleChatStore.from_persist_path(CONVERSATION_FILE)
        except json.JSONDecodeError:
            chat_store = SimpleChatStore()
    else:
        chat_store = SimpleChatStore()
    return chat_store

def save_score(score, content, total_guess, username):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "username": username,
        "Time": current_time,
        "Score": score,
        "Content": content,
        "Total guess": total_guess
    }

    try:
        with open(SCORES_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    
    data.append(new_entry)

    with open(SCORES_FILE, "w") as f:
        json.dump(data, f, indent=4)

def initialize_chatbot(chat_store, container, username, user_info):
    memory = ChatMemoryBuffer.from_defaults(
        token_litmit=3000,
        chat_store=chat_store,
        chat_store_key=username
    )

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
    index = load_index_from_storage(storage_context, index_id="vector")
    dsm5_engine = index.as_query_engine(similarity_top_k=3)
    dsm5_tool = QueryEngineTool(
        query_engine=dsm5_engine,
        metada=ToolMetadata(
            name="dsm5",
            description=(
                f"Cung cấp các thông tin liên quan đến các bệnh
                tâm thần theo tiêu chuẩn DSM5. Sử dụng câu hòi văn bản thuần túy chi 
                tiết làm đầu vào cho công cụ"
            ),
        )
    )
    save_tool = FunctionTool.from_defaults(fn=save_score)
    agent = OpenAiAgent.from_tools(
        tools=[dsm5_tool, save_tool],
        memory=memory,
        system_prompt=CUSTORM_AGENT_SYSTEM_TEMPLATE.format(user_info=user_info)
    )
    display_messages(chat_store, container, key=username)
    return agent
