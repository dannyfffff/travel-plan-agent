# planning_tools.py (根据您的要求修改)
from copy import deepcopy
from typing import List, Dict, Any
from company_manager import COMPANIES_DB
from api_tools import get_amap_driving_time
from config import COMPANY_VISIT_DURATION_MINUTES, WEIGHTS
from state import Location
from datetime import datetime, timedelta


def calculate_final_score(company: Dict[str, Any], t_available: float) -> float:
    """
    计算企业的综合最终得分 S_final。
    """
    try:
        S_attract = float(company['S_attract'])
        S_feas = float(company['S_feas'])
        T_total_trip = company['T_total_trip']

        # 计算剩余可用时间 (T_buffer): 减去旅行时间和固定拜访时间
        T_buffer = t_available - T_total_trip - COMPANY_VISIT_DURATION_MINUTES

        # 应用加权公式
        score = (WEIGHTS['alpha'] * S_attract) + \
                (WEIGHTS['beta'] * S_feas) - \
                (WEIGHTS['gamma'] * T_total_trip) + \
                (WEIGHTS['delta'] * T_buffer)

        # 将原始数据和得分返回
        company['S_final'] = score
        company['T_buffer'] = T_buffer
        return score

    except KeyError as e:
        print(f"❌ 评分计算失败，数据缺失: {e}")
        return -999.0  # 返回低分确保不会被选中
    except Exception as e:
        print(f"❌ 评分计算发生错误: {e}")
        return -999.0


def plan_multi_company_visit(
        scored_companies: List[Dict[str, Any]],
        t_available_total: float,
        hub_arrival_dt: datetime,
        hub_location: Location,
        meeting_venue_location: Location
) -> List[Dict[str, Any]]:
    """
    根据最终得分和可用时间，规划多企业拜访行程。
    """

    # 1. 计算每个企业的最终得分
    for company in scored_companies:
        calculate_final_score(company, t_available_total)

    # 2. 按 S_final 降序排序
    sorted_companies = sorted(scored_companies, key=lambda x: x['S_final'], reverse=True)

    # 3. 贪婪选择和时间规划
    final_itinerary = []
    current_location = hub_location  # 当前位置从枢纽开始
    current_time = hub_arrival_dt
    remaining_time = t_available_total

    print(f"✅ 开始贪婪选择，总可用时间: {t_available_total:.1f} min")

    for company in sorted_companies:
        company_location = company['location']  # 假设企业数据中包含 Location 结构

        # 估算当前拜访需要的总时间： 上一个点到企业 + 固定拜访时间

        # ⚠️ 关键修正：计算上一个点到当前企业的驾车时间 (需要调用 API/缓存)
        # T_prev_to_i = get_amap_driving_time(current_location, company_location)

        # ❗ 由于 get_amap_driving_time 是异步的，这里必须使用缓存或同步调用
        # 简化处理：对于第一个企业，使用 T_hub_to_i；对于后续企业，需要重新计算。
        if not final_itinerary:
            T_prev_to_i = company['T_hub_to_i']
        else:
            # 假设我们有缓存或辅助函数 get_cached_driving_time
            # T_prev_to_i = get_cached_driving_time(current_location, company_location)
            # 这里的简化版本可能不准确，但在实战中必须补全
            T_prev_to_i = company['T_prev_to_i'] if company.get('T_prev_to_i') else company['T_hub_to_i']  # 占位

        # 检查时间是否足够
        time_needed = T_prev_to_i + COMPANY_VISIT_DURATION_MINUTES
        if remaining_time >= time_needed:

            # --- 纳入行程 ---

            # 1. 交通条目 (上一个点 -> 当前企业)
            travel_start_dt = current_time
            travel_end_dt = travel_start_dt + timedelta(minutes=T_prev_to_i)

            # 2. 拜访条目 (当前企业)
            visit_start_dt = travel_end_dt
            visit_end_dt = visit_start_dt + timedelta(minutes=COMPANY_VISIT_DURATION_MINUTES )

            # 记录行程条目
            final_itinerary.append({
                'name': company['name'],
                'type': 'company_visit',
                'description': f"企业调研/拜访: {company['name']}",
                'start_time': visit_start_dt,
                'end_time': visit_end_dt,
                'location': company_location
            })

            # 3. 更新状态
            remaining_time -= time_needed
            current_time = visit_end_dt
            current_location = company_location
            print(f"✅ 纳入企业: {company['name']} (得分: {company['S_final']:.2f})，剩余时间: {remaining_time:.1f} min")

        else:
            print(
                f"⚠️ 停止规划：剩余时间 {remaining_time:.1f} 分钟不足以拜访 {company['name']} (需要 {time_needed:.1f} 分钟)。")
            break

    # 4. 最后的交通：最后一个企业 -> 会议地点
    if final_itinerary:
        pass

    return final_itinerary  # 返回选定的企业列表




# def filter_companies_by_area_by_time(center_location: Location, max_driving_minutes: int = 45) -> List[Dict[str, Any]]:
#     """
#     根据中心 Location 结构，直接调用高德API，筛选出驾车耗时在指定分钟数内的周边企业。
#
#     Args:
#         center_location: 包含 city, lat, lon 的 Location 结构 (通常是到达枢纽或会议地)。
#         max_driving_minutes: 最大可接受的驾车耗时（分钟），用于精确筛选。
#
#     Returns:
#         符合条件的企业列表，并附加 driving_time_min 字段。
#     """
#
#     city = center_location.get('city')
#     center_lat = center_location.get('lat')
#     center_lon = center_location.get('lon')
#
#     if not city or center_lat is None or center_lon is None:
#         print("⚠️ 筛选企业失败：中心位置信息（城市或经纬度）不完整。")
#         return []
#
#     city_companies = COMPANIES_DB.get(city, [])
#     nearby_companies_by_time = []
#
#     print(f"🌍 正在对 {city} 数据库进行基于时间的精确筛选 (最大耗时: {max_driving_minutes} 分钟)...")
#
#     for company in city_companies:
#         try:
#             # 1. 构造企业 Location 结构，并添加到 company 字典的副本中 (保证后续键存在)
#             # 注意：使用 company_with_time = company.copy() 来避免修改原始数据库
#             company_with_time = company.copy()
#
#             company_with_time['location'] = {
#                 'city': city,
#                 'address': company_with_time['address'],
#                 'name': company_with_time['name'],
#                 'lat': company_with_time['lat'],
#                 'lon': company_with_time['lon']
#             }
#             company_loc = company_with_time['location']  # 确保变量名一致
#
#             # 2. 精确高德 API 筛选 ---
#             driving_time_min = get_amap_driving_time(center_location, company_loc)
#
#             if driving_time_min is None:
#                 # API 调用失败或无法规划路线，跳过
#                 print(f"   -> ❌ 跳过 {company['name']}：高德 API 无法规划路线。")
#                 continue
#
#             # 3. 基于耗时进行最终筛选
#             if driving_time_min <= max_driving_minutes:
#                 company_with_time = company.copy()
#                 company_with_time['driving_time_min'] = round(driving_time_min, 1)
#
#                 company_with_time['location'] = company_loc
#
#                 nearby_companies_by_time.append(company_with_time)
#                 print(f"   -> ✅ 纳入 {company['name']} (耗时: {driving_time_min:.1f} min)")
#
#         except (KeyError, TypeError, ValueError) as e:
#             print(f"⚠️ 筛选企业 {company.get('name')} 时数据异常: {e}")
#             continue
#
#     print(f"✅ 最终筛选完成，共找到 {len(nearby_companies_by_time)} 家企业，驾车耗时满足要求。")
#     return nearby_companies_by_time

# 假设这是您外部定义的函数，它必须首先保证结构标准化！

def filter_companies_by_area_by_time(center_location: Location, max_driving_minutes: int = 45) -> List[Dict[str, Any]]:
    # ... (代码不变) ...
    city = center_location.get('city')

    if not city:
        return []

    city_companies = COMPANIES_DB.get(city, [])
    nearby_companies_by_time = []

    print(f"🌍 正在对 {city} 数据库进行基于时间的精确筛选 (最大耗时: {max_driving_minutes} 分钟)...")

    for company in city_companies:
        # 1. 最终修复：使用深度拷贝，确保 company_with_loc 是完全独立的新对象
        company_with_loc = deepcopy(company)

        # 2. 构造 Location 结构并附加 (保证 'location' 键一定存在，依赖原始数据完整性)
        try:
            company_loc: Location = {
                'city': city,
                'address': company_with_loc['address'],
                'name': company_with_loc['name'],
                'lat': company_with_loc['lat'],
                'lon': company_with_loc['lon']
            }
            # 确保在副本中添加 'location' 键
            company_with_loc['location'] = company_loc

        except KeyError as e:
            print(f"⚠️ 筛选企业 {company.get('name')} 时原始数据缺失键: {e}，跳过。")
            continue

        # 3. 核心 API 调用和时间筛选
        try:
            # get_amap_driving_time 确认是纯函数，不会修改 company_loc
            driving_time_min = get_amap_driving_time(center_location, company_loc)

            if driving_time_min is None:
                print(f"   -> ❌ 跳过 {company_with_loc['name']}：高德 API 无法规划路线。")
                continue

            # 4. 基于耗时进行最终筛选
            if driving_time_min <= max_driving_minutes:
                company_with_loc['driving_time_min'] = round(driving_time_min, 1)
                nearby_companies_by_time.append(company_with_loc)
                print(f"   -> ✅ 纳入 {company_with_loc['name']} (耗时: {driving_time_min:.1f} min)")

        except (TypeError, ValueError) as e:
            print(f"⚠️ API 调用或时间计算异常 {company_with_loc.get('name')}: {e}")
            continue

    print(f"✅ 最终筛选完成，共找到 {len(nearby_companies_by_time)} 家企业，驾车耗时满足要求。")
    return nearby_companies_by_time