# nodes.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
from config import POST_ARRIVAL_BUFFER_MINUTES, COMPANY_VISIT_DURATION_MINUTES
from llm_agent import llm_choose_transport, llm_plan_route_pre_meeting, get_final_report_by_llm, \
    get_company_scores_by_llm, llm_parse_user_input
from planning_tools import filter_companies_by_area_by_time, plan_multi_company_visit
from state import TravelPlanState, Location, ItineraryItem
from api_tools import query_flight_api, query_train_api, amap_geocode, get_amap_driving_time


def check_constraints(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 1: 信息与约束校验。
    检查用户输入是否完整、日期时间格式是否正确，并初始化 Location 结构。
    """
    user_input = state['user_input']
    user_data = llm_parse_user_input(user_input)
    # 1. 检查关键信息完整性
    required_keys = ['origin_city', 'destination_city', 'departure_date',
                     'meeting_start', 'meeting_duration_h', 'home_address', 'meeting_address',
                     'hotel_address']
    missing_keys = [k for k in required_keys if not user_data.get(k)]

    if missing_keys:
        return {"error_message": f"缺少关键输入信息: {', '.join(missing_keys)}"}

    try:
        # 2. 校验时间格式并转换为 datetime 对象
        meeting_start_dt = datetime.strptime(user_data['meeting_start'], '%Y-%m-%d %H:%M')

        # 3. 初始化 Location 结构
        home_location: Location = {
            'city': user_data['origin_city'],
            'address': user_data['home_address'],
            'name': 'Home/Start Point',
            'lat': None, 'lon': None
        }
        meeting_location: Location = {
            'city': user_data['destination_city'],
            'address': user_data['meeting_address'],
            'name': 'Meeting Venue',
            'lat': None, 'lon': None
        }
        hotel_location: Location = {
            'city': user_data['destination_city'],
            'address': user_data['hotel_address'],
            'name': 'Hotel',
            'lat': None, 'lon': None
        }

        # 4. 更新 state
        user_data['meeting_start_dt'] = meeting_start_dt

        return {
            "user_data": user_data,
            "home_location": home_location,
            "meeting_location": meeting_location,
            "hotel_location": hotel_location,
            "error_message": None
        }

    except ValueError:
        return {"error_message": "日期或时间格式不正确，请使用 YYYY-MM-DD HH:MM 格式。"}
    except Exception as e:
        return {"error_message": f"初始化校验过程中发生未知错误: {e}"}



def geocode_locations(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 2: 地理编码。
    调用高德 API 获取 home, meeting, hotel 的精确经纬度 (lat/lon)。
    """
    print("\n--- 📍 节点 2: 地理编码开始 ---")

    home_loc = state['home_location']
    meeting_loc = state['meeting_location']
    hotel_loc = state['hotel_location']

    locations_to_update = [home_loc, meeting_loc, hotel_loc]
    updated_locations = {}

    for loc in locations_to_update:
        address = loc['address']
        city = loc['city']

        # 调用地理编码工具
        coords = amap_geocode(address, city)

        if coords:
            loc['lat'] = coords['lat']
            loc['lon'] = coords['lon']
            print(f"   -> 编码成功: {loc['name']} ({loc['city']}) -> ({loc['lat']}, {loc['lon']})")
        else:
            # 如果编码失败，流程可以继续，但会降低后续路径计算的准确性
            print(f"   -> ⚠️ 编码失败: {loc['name']}，使用 None 坐标。")

    # 返回更新后的 Location 结构
    return {
        "home_location": home_loc,
        "meeting_location": meeting_loc,
        "hotel_location": hotel_loc,
        "error_message": None  # 确保没有新增错误
    }


def traffic_query(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 3: 交通查询。
    并行调用航班和高铁 API，获取所有选项，并分别存入状态。
    """
    user_data = state['user_data']

    origin = state['home_location']['city']
    destination = state['meeting_location']['city']

    meeting_start_dt = user_data['meeting_start_dt']
    target_date = meeting_start_dt.strftime('%Y-%m-%d')
    previous_date = (meeting_start_dt - timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"\n--- 🚅 节点 3: 交通查询开始 ({origin} -> {destination}) ---")

    # 1. 航班查询
    flight_options_target = query_flight_api(origin, destination, target_date)
    flight_options_prev = query_flight_api(origin, destination, previous_date)
    flight_options = flight_options_prev + flight_options_target
    # 2. 高铁查询
    train_options_target = query_train_api(origin, destination, target_date)
    train_options_prev = query_train_api(origin, destination, previous_date)
    train_options = train_options_prev + train_options_target

    total_count = len(flight_options) + len(train_options)


    if total_count == 0:
        return {"error_message": f"未查询到 {origin} 到 {destination} 的任何交通选项。"}

    print(f"✅ 查询完成。共找到 {total_count} 个交通选项。")

    return {
        "flight_options": flight_options,
        "train_options": train_options,
        "error_message": None
    }


def select_transport_by_llm(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 4: 交通选择。
    并行调用航班和高铁 API，获取所有选项，并分别存入状态。
    """
    user_data = state['user_data']
    home_loc = state['home_location']
    meeting_loc = state['meeting_location']

    # 1. 统一交通选项
    flight_options = state.get('flight_options', [])#如果 state 里 没有 flight_options 这个 key，那就返回一个 空列表，而不是报错或返回 None
    train_options = state.get('train_options', [])
    transport_options = flight_options + train_options

    if not transport_options:
        return {"error_message": "交通选择失败：无任何交通选项可供选择。"}

    print("\n--- 🧠 节点 4: LLM 班次决策开始 ---")

    # --- GeoCode 参考枢纽以计算参考通勤时间 ---
    ref_option = transport_options[0]

    # GeoCode 参考出发枢纽
    ref_dep_hub_name = ref_option['departure_hub']
    ref_dep_coords = amap_geocode(ref_dep_hub_name, home_loc['city'])
    if not ref_dep_coords:
        return {"error_message": f"无法对出发枢纽 '{ref_dep_hub_name}' 进行地理编码，流程终止。"}

    # GeoCode 参考到达枢纽
    ref_arr_hub_name = ref_option['arrival_hub']
    ref_arr_coords = amap_geocode(ref_arr_hub_name, meeting_loc['city'])
    if not ref_arr_coords:
        return {"error_message": f"无法对到达枢纽 '{ref_arr_hub_name}' 进行地理编码，流程终止。"}

    # 构造包含坐标的 Location 结构（**是解包操作，而'address': ref_dep_hub_name, 'name': ref_dep_hub_name会对前面解包后的键值进行覆盖）
    ref_origin_hub_loc: Location = {**home_loc, 'address': ref_dep_hub_name, 'name': ref_dep_hub_name, **ref_dep_coords}
    ref_arrival_hub_loc: Location = {**meeting_loc, 'address': ref_arr_hub_name, 'name': ref_arr_hub_name, **ref_arr_coords}

    # 计算参考通勤时间
    home_commute_minutes = get_amap_driving_time(home_loc, ref_origin_hub_loc)
    home_commute_minutes = home_commute_minutes if home_commute_minutes is not None else 60.0
    arrival_commute_minutes = get_amap_driving_time(ref_arrival_hub_loc, meeting_loc)
    arrival_commute_minutes = arrival_commute_minutes if arrival_commute_minutes is not None else 60.0

    print(f"   -> 参考通勤时间：{home_commute_minutes:.1f} (家->枢纽) / {arrival_commute_minutes:.1f} (枢纽->会议地)")

    # 2. 调用 DeepSeek LLM 决策
    selected_option_dict = llm_choose_transport(
        transport_options,
        user_data,
        home_commute_minutes,
        arrival_commute_minutes
    )

    if not selected_option_dict or 'departure_time' not in selected_option_dict:
        return {"error_message": "LLM未返回有效或完整的交通选择。"}

    print(f"✅ LLM初步选定班次: {selected_option_dict['type']} {selected_option_dict['id']}")

    # 返回 LLM 选定的原始数据字典
    return {
        "selected_option_raw": selected_option_dict,  # 新增一个中间状态字段
        "error_message": None,
    }



def calculate_final_transport(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 5: 交通精确计算和行程条目创建。
    针对 LLM 选定的班次，计算精确通勤时间，并创建最终的 ItineraryItem 结构。
    """
    selected_option_raw = state.get('selected_option_raw')
    if not selected_option_raw:
        return {"error_message": "交通精确计算失败：LLM 未提供选定班次。"}

    user_data = state['user_data']
    home_loc = state['home_location']
    meeting_loc = state['meeting_location']

    # 1. 解析时间
    try:
        departure_time_str = selected_option_raw['departure_time']
        arrival_time_str = selected_option_raw['arrival_time']

        # 完整的出发/到达日期
        departure_date = user_data['departure_date']
        start_time_dt = datetime.strptime(f"{departure_date} {departure_time_str}", '%Y-%m-%d %H:%M')

        # 💡 注意：跨天交通（例如夜车或长途航班）需要特殊处理，这里简化为默认在同一天
        end_time_dt = datetime.strptime(f"{departure_date} {arrival_time_str}", '%Y-%m-%d %H:%M')
        if end_time_dt < start_time_dt:
            end_time_dt += timedelta(days=1)

    except Exception as e:
        return {"error_message": f"交通精确计算失败：时间解析错误 {e}"}

    # 2. 重新地理编码枢纽 (如果尚未编码，或需要精确的枢纽名称)
    dep_hub_name = selected_option_raw['departure_hub']
    arr_hub_name = selected_option_raw['arrival_hub']

    home_city = home_loc['city']
    meeting_city = meeting_loc['city']

    dep_hub_coords = amap_geocode(dep_hub_name, home_city)
    arr_hub_coords = amap_geocode(arr_hub_name, meeting_city)

    # 🔍 优化 GeoCode 查询：如果失败，尝试加上“站”后缀
    if not dep_hub_coords and not dep_hub_name.endswith('站'):
        dep_hub_coords = amap_geocode(f"{dep_hub_name}站", home_city)

    if not arr_hub_coords and not arr_hub_name.endswith('站'):
        arr_hub_coords = amap_geocode(f"{arr_hub_name}站", meeting_city)

    if not dep_hub_coords or not arr_hub_coords:
        return {"error_message": "交通精确计算失败：无法对选定班次的枢纽进行地理编码。"}

    # 3. 构造 Location 结构进行精确路径规划
    dep_hub_loc: Location = {**home_loc, 'address': dep_hub_name, 'name': dep_hub_name, **dep_hub_coords}
    arr_hub_loc: Location = {**meeting_loc, 'address': arr_hub_name, 'name': arr_hub_name, **arr_hub_coords}

    # 4. 计算精确通勤时间 (家->枢纽, 枢纽->会议地)
    home_commute_minutes = get_amap_driving_time(home_loc, dep_hub_loc) or 60.0
    arrival_commute_minutes = get_amap_driving_time(arr_hub_loc, meeting_loc) or 60.0

    print(f"\n--- ⏱️ 节点 5: 交通精确计算 ---")
    print(f"   -> 选定班次: {selected_option_raw['id']} ({selected_option_raw['type']})")
    print(f"   -> 精确通勤时间：{home_commute_minutes:.1f} (家->枢纽) / {arrival_commute_minutes:.1f} (枢纽->会议地)")

    # 5. 计算最终到达会议地时间
    # 班次到达时间 + 枢纽到会议地的通勤时间
    final_arrival_at_venue = end_time_dt + timedelta(minutes=arrival_commute_minutes)

    # 6. 构造最终的 ItineraryItem
    selected_transport: ItineraryItem = {
        'type': 'transport',
        'description': f"{selected_option_raw['type']} {selected_option_raw['id']} ({dep_hub_name} -> {arr_hub_name})",
        'start_time': start_time_dt,
        'end_time': end_time_dt,
        'location': arr_hub_loc,  # 使用到达枢纽的位置，或者直接使用目的地城市坐标
        'details': {
            'raw_option': selected_option_raw,
            'price': selected_option_raw.get('price'),
            'duration': selected_option_raw.get('duration'),
            'home_commute_min': home_commute_minutes,
            'arrival_commute_min': arrival_commute_minutes,
            'final_arrival_at_venue': final_arrival_at_venue,
        }
    }

    # 7. 更新状态
    user_data['actual_arrival_at_venue'] = final_arrival_at_venue  # 更新用户数据中的精确到达时间

    return {
        "selected_transport": selected_transport,
        "user_data": user_data,
        "error_message": None
    }



def pre_meeting_plan(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 6: 会议前行程规划。
    使用混合评分（LLM吸引力/可行性 + 时间成本）和贪婪算法进行多企业调研规划。
    并实现【简单粗暴的截断回退机制】。
    """
    print("\n--- 🧭 节点 6: 会议前行程规划开始 ---")


    selected_transport = state.get('selected_transport')
    user_data = state['user_data']
    meeting_loc = state['meeting_location']

    # 1. 变量初始化
    pre_meeting_route_final: List[ItineraryItem] = []  # 最终行程列表
    final_arrival_time: datetime = None

    if not selected_transport:
        print("❌ 会议前规划失败：未选定主要交通方式。")
        return {"pre_meeting_route": pre_meeting_route_final,
                "error_message": "未选定主要交通方式，跳过会议前规划。"}

    # 2. 确定到达枢纽的位置和时间
    arrival_hub_loc = selected_transport['location']
    arrival_at_hub_dt = selected_transport['end_time']
    arrival_commute_min = selected_transport['details']['arrival_commute_min']

    # ⚠️ 时间计算 (使用正确的变量名 latest_arrival_needed)
    meeting_start_dt = user_data['meeting_start_dt']
    latest_arrival_needed = meeting_start_dt - timedelta(minutes=POST_ARRIVAL_BUFFER_MINUTES)
    time_window_available = latest_arrival_needed - arrival_at_hub_dt
    available_minutes = time_window_available.total_seconds() / 60

    print(f"   -> 枢纽到达时间: {arrival_at_hub_dt.strftime('%H:%M')}")
    print(f"   -> 最晚需到达时间: {latest_arrival_needed.strftime('%H:%M')} (含 {POST_ARRIVAL_BUFFER_MINUTES}min 缓冲)")
    print(f"   -> 规划可用空闲时间: {available_minutes:.1f} 分钟")

    # 3. 企业筛选和时间成本计算 (与原逻辑保持一致)
    max_driving_time_to_meeting = arrival_commute_min
    first_filtered_companies = filter_companies_by_area_by_time(
        center_location=arrival_hub_loc,
        max_driving_minutes=int(max_driving_time_to_meeting * 2)
    )

    print(f"🌍 正在对 {len(first_filtered_companies)} 家潜在企业进行基于时间的精确筛选...")
    available_companies = []

    # 3b. 二次筛选：计算完整行程时间并检查可行性
    for company in first_filtered_companies:
        company_loc = company['location']
        T_hub_to_i = get_amap_driving_time(arrival_hub_loc, company_loc)
        T_i_to_meeting = get_amap_driving_time(company_loc, meeting_loc)  # 修正：这里不再有 ['location']

        if T_hub_to_i is None or T_i_to_meeting is None:
            continue

        time_needed_for_visit = T_hub_to_i + COMPANY_VISIT_DURATION_MINUTES + T_i_to_meeting

        if time_needed_for_visit <= available_minutes:
            company['T_hub_to_i'] = T_hub_to_i
            company['T_i_to_meeting'] = T_i_to_meeting
            company['T_total_trip'] = T_hub_to_i + T_i_to_meeting
            # ❗ T_buffer 是关键的可行性指标，必须计算
            company['T_buffer'] = available_minutes - time_needed_for_visit
            available_companies.append(company)
            print(f"   -> ✅ 纳入 {company['name']} (总耗时: {time_needed_for_visit:.1f} min)")

    if not available_companies:
        print("⚠️ 未找到顺路且时间可行的调研企业。")
        return {"pre_meeting_route": pre_meeting_route_final, "error_message": None}

    # 4. 混合评分和贪婪规划 (与原逻辑保持一致)
    print(f"🌍 正在对 {len(available_companies)} 家企业进行 LLM 智能评分...")
    scored_companies_llm_output = get_company_scores_by_llm(available_companies, available_minutes)
    if not scored_companies_llm_output:
        print("❌ LLM 评分阶段失败，本次行程无会议前调研。")
        return {"pre_meeting_route": pre_meeting_route_final, "error_message": None}

    # 4b. 数据合并/回填 (确保数据结构完整)
    merged_companies_for_planning = []
    original_companies_map = {c['name']: c for c in available_companies}
    for scored_item in scored_companies_llm_output:
        company_name = scored_item['name']
        original_data = original_companies_map.get(company_name)
        if original_data:
            merged_item = original_data.copy()
            merged_item.update(scored_item)
            try:
                merged_item['S_attract'] = float(merged_item['S_attract'])
                merged_item['S_feas'] = float(merged_item['S_feas'])
                merged_companies_for_planning.append(merged_item)
            except (ValueError, TypeError):
                print(f"⚠️ 警告：企业 {company_name} 的 LLM 评分数据格式不正确，跳过。")

    # 4c. 贪婪规划 (获取按 S_final 降序排序的完整序列)
    print("🧠 正在使用混合评分和贪婪算法进行多企业规划...")
    final_visit_plan_data = plan_multi_company_visit(
        merged_companies_for_planning,
        available_minutes,
        arrival_at_hub_dt,
        arrival_hub_loc,
        meeting_loc
    )

    if not final_visit_plan_data:
        print("⚠️ 混合评分机制未能规划任何调研企业。")
        return {"pre_meeting_route": pre_meeting_route_final, "error_message": None}

    print(f"✅ LLM/混合评分成功规划 {len(final_visit_plan_data)} 个调研企业。")

    # --- 5. 简单粗暴的【截断回退】循环 ---
    print("--- 🔄 开始截断回退，寻找最大可行子集 ---")

    while final_visit_plan_data:

        current_loc = arrival_hub_loc
        current_time = arrival_at_hub_dt
        current_route_items: List[ItineraryItem] = []  # 用于本次循环的检查
        all_routes_planned = True

        # 5a. 临时构建路由并计算到达时间
        for i, visit_item in enumerate(final_visit_plan_data):
            company_name = visit_item['name']
            company_loc = visit_item['location']

            # 1. 交通段：上一个点 -> 当前企业
            T_prev_to_i = get_amap_driving_time(current_loc, company_loc)

            if T_prev_to_i is None:
                all_routes_planned = False
                break

            travel_start_dt = current_time
            travel_end_dt = travel_start_dt + timedelta(minutes=T_prev_to_i)

            # 存储交通段
            current_route_items.append({
                'type': 'transport',
                'description': f"驾车前往调研企业 {company_name}",
                'start_time': travel_start_dt,
                'end_time': travel_end_dt,
                'location': company_loc,
                'details': {'duration_min': T_prev_to_i}
            })

            # 2. 活动段：企业调研
            visit_start_dt = travel_end_dt
            visit_end_dt = visit_start_dt + timedelta(minutes=COMPANY_VISIT_DURATION_MINUTES)

            # 存储活动段
            current_route_items.append({
                'type': 'company_visit',
                'description': f"企业调研/拜访: {company_name}",
                'start_time': visit_start_dt,
                'end_time': visit_end_dt,
                'location': company_loc,
                'details': {'company_name': company_name}
            })

            # 3. 更新状态
            current_time = visit_end_dt
            current_loc = company_loc

        # 5b. 最终交通段：最后一个活动地点 -> 会议地点 (检查可行性)
        if not all_routes_planned:
            # 移除得分最低的企业并重新尝试
            removed_company = final_visit_plan_data.pop()
            print(f"❌ 内部路线规划中断，移除得分最低企业: {removed_company['name']}。")
            continue

        final_commute_min = get_amap_driving_time(current_loc, meeting_loc)

        if final_commute_min is None:
            removed_company = final_visit_plan_data.pop()
            print(f"❌ 警告：无法获取最后一个企业到会议地点的路线。移除企业: {removed_company['name']}。")
            continue

        final_arrival_time = current_time + timedelta(minutes=final_commute_min)

        # 5c. 检查可行性
        if final_arrival_time <= latest_arrival_needed:
            # 行程可行！添加最后的交通段并保存

            current_route_items.append({
                'type': 'transport',
                'description': "驾车前往会议地点",
                'start_time': current_time,
                'end_time': final_arrival_time,
                'location': meeting_loc,
                'details': {'duration_min': final_commute_min}
            })

            pre_meeting_route_final = current_route_items  # 保存最终可行路由

            print(
                f"✅ 找到最大可行行程，共 {len(final_visit_plan_data)} 个企业。最终到达时间: {final_arrival_time.strftime('%H:%M')}。")
            break  # 退出 while 循环
        else:
            # 行程不可行，移除得分最低的（最后一个）企业，重新循环
            removed_company = final_visit_plan_data.pop()
            print(
                f"❌ 行程不可行 (到达 {final_arrival_time.strftime('%H:%M')} 晚于 {latest_arrival_needed.strftime('%H:%M')})，"
                f"移除得分最低企业: {removed_company['name']}。尝试 {len(final_visit_plan_data)} 个企业。"
            )

    # 6. 如果循环结束，final_visit_plan_data 为空
    if not pre_meeting_route_final:
        print("⚠️ 无法在可用时间内规划任何企业调研活动。")

    # 7. 保存最终结果到状态
    # 确保 final_arrival_time 不为 None，即使 pre_meeting_route_final 是空列表
    if final_arrival_time is None:
        # 如果没有调研，最终到达时间是枢纽到达时间 + 初始通勤时间
        final_arrival_time = arrival_at_hub_dt + timedelta(minutes=arrival_commute_min)

    print(f"✅ 会议前规划完成，共生成 {len(pre_meeting_route_final)} 个行程条目。")
    print(f"   -> 最终到达会议地时间: {final_arrival_time.strftime('%H:%M')}")

    return {
        "pre_meeting_route": pre_meeting_route_final,
        "final_arrival_at_venue": final_arrival_time,
        "error_message": None
    }

def post_meeting_plan(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 5: 会议后行程规划。
    规划会议结束到酒店的行程，并整合会议条目。
    """
    print("\n--- 🏨 节点 5: 会议后行程规划开始 ---")

    user_data = state['user_data']
    meeting_loc = state['meeting_location']
    hotel_loc = state['hotel_location']

    # 1. 确定会议的开始和结束时间
    meeting_start_dt = user_data['meeting_start_dt']
    # 从 user_data 获取会议持续时间，默认 2 小时
    meeting_duration_h = user_data.get('meeting_duration_h', 2)
    meeting_end_dt = meeting_start_dt + timedelta(hours=meeting_duration_h)

    current_time = meeting_end_dt
    current_loc = meeting_loc

    post_meeting_route: List[ItineraryItem] = []

    # 2. 创建会议活动条目
    meeting_item: ItineraryItem = {
        'type': 'meeting',
        'description': '商务会议',
        'start_time': meeting_start_dt,
        'end_time': meeting_end_dt,
        'location': meeting_loc,
        'details': {'duration_h': meeting_duration_h}
    }

    # 将会议条目添加到会议后行程列表
    post_meeting_route.append(meeting_item)

    # 3. 规划从会议地点到酒店的交通
    # 假设 get_amap_driving_time 是一个已实现的函数
    commute_to_hotel_min = get_amap_driving_time(current_loc, hotel_loc) or 30.0

    arrival_at_hotel_dt = current_time + timedelta(minutes=commute_to_hotel_min)

    # 创建交通条目
    post_meeting_route.append({
        'type': 'transport',
        'description': "驾车前往酒店",
        'start_time': current_time,
        'end_time': arrival_at_hotel_dt,
        'location': hotel_loc,
        'details': {'duration_min': commute_to_hotel_min}
    })
    current_time = arrival_at_hotel_dt
    current_loc = hotel_loc

    # 4. 创建酒店入住条目
    # 预留 30 分钟办理入住时间
    post_meeting_route.append({
        'type': 'hotel',
        'description': f"入住酒店: {hotel_loc['name']}",
        'start_time': current_time,
        'end_time': current_time + timedelta(minutes=30),
        'location': hotel_loc,
        'details': {'status': 'check-in'}
    })

    print(f"✅ 会议后规划完成，共生成 {len(post_meeting_route)} 个行程条目。")
    print(f"   -> 预计入住时间: {arrival_at_hotel_dt.strftime('%H:%M')}")

    return {
        "post_meeting_route": post_meeting_route,
        "error_message": None
    }


def generate_final_itinerary(state: TravelPlanState) -> Dict[str, Any]:
    """
    节点 6: 整合所有行程条目，调用 LLM 生成最终的格式化报告。
    """
    print("\n--- 📝 节点 6: 生成最终行程报告开始 ---")

    # 1. 整合所有行程条目
    itinerary_items: List[ItineraryItem] = []

    # 获取主交通段行程 (通常包含 家->枢纽, 枢纽活动, 主交通)
    if state.get('selected_transport'):
        main_route = state['selected_transport']['details'].get('itinerary', [])
        itinerary_items.extend(main_route)

    # 获取会议前行程 (包含枢纽到第一个调研公司，调研公司之间的交通，最后一个调研公司到会议地点的交通)
    if state.get('pre_meeting_route'):
        itinerary_items.extend(state['pre_meeting_route'])

    # 获取会议后行程 (包含会议本身、会议到酒店的交通、酒店入住)
    if state.get('post_meeting_route'):
        # 修复点 1：将 state.post_meeting_route 改为 state['post_meeting_route']
        itinerary_items.extend(state['post_meeting_route'])

    # 2. 按时间排序所有条目 (重要：确保时间顺序正确)
    itinerary_items.sort(key=lambda x: x['start_time'])

    print(f"   -> 已整合 {len(itinerary_items)} 个行程条目。")

    # 3. 调用 LLM 生成报告
    # 修复点 2：将 state.user_data 改为 state['user_data']
    final_report_markdown = get_final_report_by_llm(
        state['user_data'], # <--- **关键修复点**
        itinerary_items
    )

    # 4. 返回状态更新
    return {
        "final_itinerary_report": final_report_markdown,
        "final_itinerary": itinerary_items,
        "error_message": None
    }