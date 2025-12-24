# main.py (最终修正版本)

from graph import build_travel_graph
from llm_agent import llm_parse_user_input
from state import TravelPlanState
from datetime import datetime, timedelta
from pprint import pprint

# 模拟用户输入数据
# INITIAL_INPUT = {
#     'user_data': {
#         'origin_city': '上海',
#         'destination_city': '深圳',
#         'departure_date': '2025-12-25',
#         'meeting_start': '2025-12-25 16:00',
#         'meeting_duration_h': 1,
#         'home_address': '上海市浦东新区川沙新镇黄赵路310号',
#         'meeting_address': '深圳市南山区桃园路2号',
#         'hotel_address': '深圳市南山区西丽街道官龙村西82号'
# }
# }
INITIAL_INPUT = {
    'user_data': {
        'origin_city': '上海',
        'destination_city': '深圳',
        'departure_date': '2025-12-25',
        'meeting_start': '2025-12-25 16:00',
        'meeting_duration_h': 1,
        'home_address': '上海市浦东新区川沙新镇黄赵路310号',
        'meeting_address': '深圳市南山区桃园路2号',
        'hotel_address': '深圳市南山区西丽街道官龙村西82号'
}
}


def run_planner():
    # 编译图结构
    app = build_travel_graph().compile()
    INITIAL_INPUT['user_data'] = llm_parse_user_input("规划2025-12-25上海到深圳的行程：从上海市浦东新区川沙新镇黄赵路310号出发，会议地址是深圳市南山区桃园路2号，开始时间是2025-12-25 16:00，开一小时。酒店是深圳市南山区西丽街道官龙村西82号，")

    print("--- ✈️ 行程规划助手启动 ---")
    print(
        f"初始输入: {INITIAL_INPUT['user_data']['origin_city']} -> {INITIAL_INPUT['user_data']['destination_city']} ({INITIAL_INPUT['user_data']['departure_date']})")

    # 1. 预处理用户数据：将时间字符串转换为 datetime 对象
    user_data = INITIAL_INPUT['user_data'].copy()

    # 核心修正：将 'meeting_start' 字符串解析为 datetime 对象
    meeting_start_str = user_data['meeting_start']
    meeting_start_dt = datetime.strptime(meeting_start_str, "%Y-%m-%d %H:%M")

    # 将 datetime 对象存入状态，供后续节点使用
    user_data['meeting_start_dt'] = meeting_start_dt

    # 2. 初始化 Graph 状态
    initial_state = TravelPlanState(user_data=user_data)

    # 3. 运行图
    try:
        # 使用 .invoke() 运行
        final_state = app.invoke(initial_state, config={"recursion_limit": 10})

    except Exception as e:
        print(f"\n❌ LangGraph 运行异常！")
        print(f"具体错误: {e}")
        return

    print("\n--- 🏁 运行总结 ---")

    # 检查流程是否因错误终止
    if final_state.get('error_message'):
        print(f"❌ 流程在中间步骤终止。错误信息: {final_state['error_message']}")
    else:
        print("✅ 前期规划运行成功。数据状态如下：")

    # --- 打印中间结果摘要 ---
    selected = final_state.get('selected_transport')
    actual_arrival = final_state.get('user_data', {}).get('actual_arrival_at_venue')
    meeting_start_dt = final_state.get('user_data', {}).get('meeting_start_dt')

    print("\n**【节点 1, 2, 3 结果】**")
    print(f"   - 会议开始时间: {meeting_start_dt.strftime('%Y-%m-%d %H:%M') if meeting_start_dt else 'N/A'}")
    print(
        f"   - 交通选项数量: {len(final_state.get('flight_options', [])) + len(final_state.get('train_options', []))}")
    print("-" * 30)

    print("   - 选定班次: ")
    if selected:
        commute_info = selected.get('details', {})
        # 假设 PRE_DEPARTURE_BUFFER_MINUTES = 90
        buffer_minutes = 90
        departure_dt = selected['start_time']
        actual_start_time = departure_dt - timedelta(minutes=buffer_minutes)

        print(f"       > 类型/ID: {selected['description']}")
        print(
            f"       > 班次时间: {departure_dt.strftime('%H:%M')} (起飞/发车) -> {selected['end_time'].strftime('%H:%M')} (到达)")
        print(f"       > 价格: {commute_info.get('price', 'N/A')} 元")
        print(f"       > 需在 {actual_start_time.strftime('%H:%M')} 从家出发 (含缓冲)。")
        print(
            f"       > 预估到达会议地时间: {actual_arrival.strftime('%H:%M') if actual_arrival else 'N/A'} (远早于会议开始时间)。")
        print(f"       > 到达枢纽: {selected['location']['name']}")

    else:
        print("   - 选定班次: 无 (流程在节点 3a 之后结束，或 3b 失败)")

    # --- 核心修正：打印最终报告内容 ---
    # NOTE: 我们知道 LLM 报告内容已生成，但 LangGraph 状态同步失败导致 final_report 为空。
    # 打印逻辑不变，但您需要确保 nodes.py 已经修复。
    final_report = final_state.get('final_itinerary_report')

    if final_report and len(final_report.strip()) > 0:
        print("\n\n*** ✅ 最终商务行程报告 (Markdown) ***")
        print("=" * 60)
        # 💡 直接打印报告内容
        print(final_report)
        print("=" * 60)
    else:
        # ⚠️ 既然调试日志已经打印了 AIMessage(content=...)，我们知道内容存在，
        # 如果这里仍然是空，说明状态同步失败。
        print("\n\n*** ⚠️ 最终报告生成失败或为空 ***")
        print("问题可能出在 **nodes.py** 中 'generate_final_itinerary' 函数的 **状态返回逻辑**，报告内容未正确写入 LangGraph 状态。")
        print(f"状态中的 final_itinerary_report 长度: {len(final_report or '')}")


    # --- 打印完整的 TravelPlanState (摘要) ---
    print("\n\n*** 🔍 完整的 TravelPlanState 状态内容 (摘要) ***")

    # 创建一个摘要状态字典
    summary_state = final_state.copy()

    # 摘要处理列表
    summary_state['flight_options'] = f"<{len(final_state.get('flight_options', []))} 趟航班>"
    summary_state['train_options'] = f"<{len(final_state.get('train_options', []))} 趟高铁>"
    summary_state['pre_meeting_route'] = f"<{len(final_state.get('pre_meeting_route', []))} 个行程条目>"
    summary_state['post_meeting_route'] = f"<{len(final_state.get('post_meeting_route', []))} 个行程条目>"

    # 修正 final_itinerary 的摘要
    report_len = len(final_state.get('final_itinerary_report', ''))
    summary_state['final_itinerary_report'] = f"<{report_len} 字符的最终报告>"

    pprint(summary_state, indent=2)


if __name__ == "__main__":
    run_planner()