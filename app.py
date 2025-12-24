import streamlit as st
import os
import operator
from typing import List, Annotated
from langgraph.graph import StateGraph  # 导入 LangGraph 核心
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY
# --- 导入您的 LangGraph 模块 ---
# 确保 graph.py, state.py, nodes.py 在同一目录
from graph import build_travel_graph
from state import TravelPlanState  # 导入您的状态类

# --- 1. Streamlit 界面配置 ---

st.set_page_config(page_title="LangGraph 商务行程规划", layout="wide")
st.title("✈️ 商务行程规划智能体")
st.caption("基于 LangGraph 的多步骤旅行规划解决方案")

# 初始化会话历史（Session State）
if "messages" not in st.session_state:
    # 初始欢迎消息，指导用户输入
    st.session_state["messages"] = [
        AIMessage(
            content="您好！我是您的智能行程规划师。\n\n请提供您的行程需求，例如：\n`规划2025-12-25上海到深圳的行程，会议地址是深圳市南山区桃园路2号，酒店是深圳市南山区西丽街道官龙村西82号。`")
    ]
if "intermediate_steps" not in st.session_state:
    st.session_state["intermediate_steps"] = []


# --- 2. 初始化 LangGraph 和 LLM 客户端（只运行一次） ---

@st.cache_resource
def initialize_agent():
    """
    安全初始化 LangGraph 和 LLM 客户端。
    使用 st.cache_resource 确保只在应用启动时运行一次。
    """
    st.info("💡 正在初始化 LangGraph 模型和 LLM 客户端...")

    # 2.1 ✅ 密钥安全读取
    try:
        api_key = DEEPSEEK_API_KEY
    except KeyError:
        st.error("🔑 错误：未找到 OPENAI_API_KEY。")
        st.stop()

    # 2.2 初始化 LLM 客户端 (供 LangGraph 节点使用)
    llm_client = ChatOpenAI(api_key=api_key, model="deepseek-chat", temperature=0)  # 建议规划类任务使用低温度

    # 2.3 构建并编译 LangGraph
    workflow = build_travel_graph()
    # 编译图
    compiled_graph = workflow.compile()

    st.success("✅ 初始化完成！您可以开始提问了。")
    return llm_client, compiled_graph


# 初始化并存储到 session_state，供 handle_user_input 使用
if "llm_client" not in st.session_state:
    st.session_state.llm_client, st.session_state.compiled_graph = initialize_agent()


# --- 3. 核心逻辑：处理用户输入和调用 LangGraph ---

def handle_user_input(prompt):
    """处理用户输入，调用 LangGraph，并更新会话状态"""

    # 1. 构造历史记录（LangGraph 需要 BaseMessage 类型）
    # 由于行程规划是一个基于输入的单次复杂执行，我们主要传递输入文本，
    # 而不是完整的聊天历史作为 LangGraph 的核心状态。
    # ⚠️ 注意：LangGraph 的 TravelPlanState 不直接接收完整的 BaseMessage 历史，
    # 而是接收原始文本输入 `user_input`。

    # 清空旧的中间步骤记录
    st.session_state.intermediate_steps = []

    # 2. 构造 LangGraph 的输入状态 (TravelPlanState)
    # LangGraph 将从用户输入文本中解析出所需参数。
    initial_state = TravelPlanState(
        user_input=prompt,
        itinerary_items=[],
        error_message=None
        # 其他字段为 None 或默认值
    )

    # 3. 调用编译好的 LangGraph
    with st.spinner("🚀 正在执行多步骤规划..."):
        # 调用智能体，LangGraph 会自动从 START 节点开始执行
        # 传入的初始状态必须是 TravelPlanState 的实例
        result = st.session_state.compiled_graph.invoke(initial_state)

    # 4. 解析结果并更新 Streamlit 历史

    # LangGraph 的最终输出是生成报告的 Markdown 字符串，存储在 final_report 字段
    agent_response_content = result.get("final_itinerary_report", "抱歉，规划流程执行失败，请检查输入格式或查看调试信息。")

    # 记录错误信息
    if result.get("error_message"):
        agent_response_content += f"\n\n**❌ 规划流程终止：** {result['error_message']}"

    # 将用户输入和智能体回复添加到 Streamlit 的历史记录中
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=agent_response_content))

    # 5. 更新中间步骤记录 (可选，用于调试)
    # ⚠️ 由于您的 LangGraph 状态没有 'intermediate_steps' 字段，这里我们使用一个简化的调试信息
    if result.get("meeting_start_dt"):
        st.session_state.intermediate_steps.append(
            f"会议开始时间: {result['meeting_start_dt'].strftime('%Y-%m-%d %H:%M')}")
    if result.get("final_report"):
        st.session_state.intermediate_steps.append("最终报告已成功生成。")


# --- 4. 渲染聊天界面和调试工具 ---

# 遍历并显示所有历史消息
for msg in st.session_state.messages:
    # 自动识别 HumanMessage 或 AIMessage
    st.chat_message(msg.type).write(msg.content)

# 用户输入框
if user_prompt := st.chat_input("输入您的行程规划需求..."):
    # 调用处理函数
    handle_user_input(user_prompt)
    # 强制重新运行脚本以显示最新消息
    st.rerun()

# 调试侧边栏 (可选)
with st.sidebar:
    st.header("调试信息与控制")

    if st.checkbox("清除聊天历史"):
        st.session_state.messages = [AIMessage(content="您好！我是您的智能体，有什么可以帮您？")]
        st.session_state.intermediate_steps = []
        st.rerun()

    # 显示简化的执行日志
    if st.checkbox("显示执行日志 (关键步骤)"):
        st.subheader("流程执行关键点")
        if st.session_state.intermediate_steps:
            for i, step in enumerate(st.session_state.intermediate_steps):
                st.write(f"- {step}")
        else:
            st.write("暂无执行日志。请先输入一个规划请求。")