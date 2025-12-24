# graph.py (完整修正与更新)

from langgraph.graph import StateGraph, END, START
from typing import Literal
from state import TravelPlanState
from nodes import (
    check_constraints,
    geocode_locations,
    traffic_query,
    select_transport_by_llm,
    calculate_final_transport, pre_meeting_plan, generate_final_itinerary, post_meeting_plan
)


# --- 定义图中的决策函数 (Conditional Edge) ---

def decide_next_step(state: TravelPlanState) -> Literal["geocode_locations", "end"]:
    """
    check_constraints 后的决策：成功则进入地理编码。
    """
    if state.get("error_message"):
        print(f"❌ 流程终止：{state['error_message']}")
        return "end"
    else:
        print("✅ 校验通过，进入地理编码阶段。")
        return "geocode_locations"


# 修正：判断 `transport_options` 是否存在已在 `traffic_query` 内部被处理，
# 并且我们现在使用 `select_transport_by_llm` 作为目标节点名。
def decide_after_traffic_query(state: TravelPlanState) -> Literal["select_transport_by_llm", "end"]:
    """
    交通查询后决定下一步。
    由于 traffic_query 已经将选项合并，这里检查合并后的选项是否缺失。
    """
    # 理论上，如果 traffic_query 失败，它会设置 error_message
    if state.get("error_message"):
        print(f"❌ 流程终止：交通查询失败。")
        return "end"
    else:
        print("✅ 交通选项已获取，进入 LLM 决策阶段。")
        # 修正目标节点名
        return "select_transport_by_llm"

    # 修正：判断 LLM 决策成功后，进入 calculate_final_transport 节点。


def decide_after_llm_select(state: TravelPlanState) -> Literal["calculate_final_transport", "end"]:
    """
    LLM 班次选择后决定下一步。
    """
    # 检查 LLM 是否在 selected_option_raw 中返回了有效数据
    if state.get("selected_option_raw"):
        print("✅ LLM 班次已选定，进入精确计算阶段。")
        return "calculate_final_transport"
    else:
        print("❌ 流程终止：LLM 交通决策失败。")
        return "end"


# 新增：判断最终交通计算后，进入 pre_meeting_plan 节点。
def decide_after_traffic_calculation(state: TravelPlanState) -> Literal["pre_meeting_plan", "end"]:
    """
    交通计算后决定下一步。
    """
    if state.get("selected_transport"):
        print("✅ 交通行程条目已创建，进入会议前行程规划。")
        return "pre_meeting_plan"
    else:
        print("❌ 流程终止：交通精确计算失败。")
        return "end"


# --- 构建 LangGraph ---
def build_travel_graph() -> StateGraph:
    workflow = StateGraph(TravelPlanState)

    # 1. 添加节点 (Nodes)
    workflow.add_node("check_constraints", check_constraints)  # 节点 1
    workflow.add_node("geocode_locations", geocode_locations)  # 节点 1.5
    workflow.add_node("traffic_query", traffic_query)  # 节点 2
    workflow.add_node("select_transport_by_llm", select_transport_by_llm)  # 节点 3a (LLM 决策)
    workflow.add_node("calculate_final_transport", calculate_final_transport)  # 节点 3b (精确计算)

    # 💡 确保添加了 post_meeting_plan 节点
    workflow.add_node("pre_meeting_plan", pre_meeting_plan)  # 节点 4
    workflow.add_node("post_meeting_plan", post_meeting_plan)  # 节点 5 (缺失节点已补齐)
    workflow.add_node("generate_final_itinerary", generate_final_itinerary)  # 节点 6

    # 2. 定义起点 (Entry Point)
    workflow.add_edge(START, "check_constraints")

    #

    # 3. 定义边 (Edges)

    # 边 1: 校验 -> (地理编码 或 结束)
    workflow.add_conditional_edges("check_constraints", decide_next_step,
                                   {"geocode_locations": "geocode_locations", "end": END})

    # 边 1.5: 地理编码 -> 交通查询
    workflow.add_edge("geocode_locations", "traffic_query")

    # 边 2: 交通查询 -> (LLM 决策 或 结束)
    workflow.add_conditional_edges("traffic_query", decide_after_traffic_query,
                                   {"select_transport_by_llm": "select_transport_by_llm", "end": END})
    workflow.add_conditional_edges("select_transport_by_llm", decide_after_llm_select,
                                   {"calculate_final_transport": "calculate_final_transport", "end": END})

    # 边 3: 精确交通计算 -> (会议前规划 或 结束)
    # 流程必须继续，如果精确计算成功，则进入会议前规划
    workflow.add_conditional_edges("calculate_final_transport", decide_after_traffic_calculation,
                                   {"pre_meeting_plan": "pre_meeting_plan", "end": END})

    # 💡 边 4: 会议前规划 -> 会议后规划 (修正：流程必须继续到下一个规划阶段)
    workflow.add_edge("pre_meeting_plan", "post_meeting_plan")

    # 💡 边 5: 会议后规划 -> 报告生成 (新增边)
    workflow.add_edge("post_meeting_plan", "generate_final_itinerary")

    # 💡 边 6: 报告生成 -> 结束 (终点)
    workflow.add_edge("generate_final_itinerary", END)

    return workflow