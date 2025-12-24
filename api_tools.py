# api_tools.py
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
from config import AMAP_API_KEY, AMAP_GEOCODE_URL, AMAP_ROUTE_URL, SERPAPI_FLIGHTS_API_KEY, GOOGLE_FLIGHTS_URL, \
    JUHE_TRAIN_API_KEY, JUHE_TRAIN_QUERY_URL
from state import Location
import requests

MAX_RETRIES = 3 # 最大重试次数
INITIAL_WAIT_TIME = 1.0 # 初始等待时间（秒）

def amap_geocode(address: str, city: str) -> Optional[Dict[str, float]]:
    """调用高德地理编码API，返回经纬度"""
    if not AMAP_API_KEY:
        print("❌ 致命错误：AMAP_API_KEY 未配置，无法进行地理编码。")
        return None

    params = {
        "key": AMAP_API_KEY,
        "address": address,
        "city": city,
        "output": "json"
    }

    try:
        response = requests.get(AMAP_GEOCODE_URL, params=params, timeout=5)
        response.raise_for_status()  # 检查 HTTP 错误
        data = response.json()

        # 高德 API 成功响应检查
        if data.get("status") == "1" and int(data.get("count", 0)) > 0:
            # 取第一个结果
            geocodes = data["geocodes"][0]
            location_str = geocodes.get("location")  # 格式如 "116.397428,39.90923"

            if location_str:
                lon, lat = map(float, location_str.split(','))
                return {"lat": lat, "lon": lon}

        print(f"⚠️ 高德地理编码失败。状态码: {data.get('status')}, 原因: {data.get('info')}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 高德 API 请求失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 处理高德 API 响应时发生错误: {e}")
        return None


def get_amap_driving_time(origin: Location, destination: Location) -> Optional[float]:
    """
    实际调用高德路径规划API，计算两个地点间的驾车耗时（分钟）。
    加入延时和指数退避重试机制，以解决 QPS 超限问题。

    Args:
        origin: 起点 Location 结构 (需要 lat/lon)。
        destination: 终点 Location 结构 (需要 lat/lon)。

    Returns:
        驾车耗时（分钟），失败返回 None。
    """
    if not AMAP_API_KEY:
        print("❌ 致命错误：AMAP_API_KEY 未配置，无法计算驾车时间。")
        return None

    # 1. 检查经纬度是否可用
    if not origin.get('lat') or not destination.get('lat'):
        print(f"⚠️ 无法计算驾车时间: 起点或终点的经纬度缺失。")
        return 35.0  # 使用经验值回退

    # 2. 构造请求参数
    origin_coords = f"{origin['lon']},{origin['lat']}"
    destination_coords = f"{destination['lon']},{destination['lat']}"

    params = {
        "key": AMAP_API_KEY,
        "origin": origin_coords,
        "destination": destination_coords,
        "output": "json",
        "extensions": "base",
        "strategy": 0
    }

    wait_time = INITIAL_WAIT_TIME

    # === 循环重试机制开始 ===
    for attempt in range(MAX_RETRIES):
        try:
            # 1. 发送请求
            response = requests.get(AMAP_ROUTE_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            # 2. 检查高德 API 状态码
            if data.get("status") == "1" and int(data.get("count", 0)) > 0:
                # 路径规划成功，返回结果
                route = data['route']['paths'][0]
                duration_seconds = int(route.get('duration', 0))
                return round(duration_seconds / 60.0, 1)

            # 3. API 错误处理，特别是针对 QPS 超限
            error_reason = data.get('info', '未知错误')

            # 检查是否为 QPS 或配额相关错误 (状态码通常为 0，错误信息包含 LIMIT/QUOTA等关键词)
            is_limit_error = (data.get("status") == "0" and
                              ('LIMIT' in error_reason.upper() or
                               'QUOTA' in error_reason.upper()))

            if is_limit_error:
                if attempt < MAX_RETRIES - 1:
                    # 进行重试
                    print(f"🚦 QPS 超限，尝试第 {attempt + 1} 次重试，等待 {wait_time:.1f} 秒...")
                    time.sleep(wait_time)
                    wait_time *= 2  # 指数退避：1.0s, 2.0s, 4.0s...
                    continue  # 跳转到下一个循环
                else:
                    # 达到最大重试次数
                    print(f"❌ 高德路径规划失败: 已达最大重试次数，原因: {error_reason}")
                    return None
            else:
                # 其他 API 错误（例如参数错误等），不重试
                print(f"⚠️ 高德路径规划 API 返回失败。状态码: {data.get('status')}, 原因: {error_reason}")
                return None

        except requests.exceptions.RequestException as e:
            # 网络或 HTTP 错误，通常表示瞬时网络问题
            if attempt < MAX_RETRIES - 1:
                print(f"❌ API 请求失败 (网络错误)，尝试第 {attempt + 1} 次重试，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
                wait_time *= 2
                continue
            else:
                print(f"❌ 高德路径规划 API 请求失败: {e}")
                return None

        except Exception as e:
            # 捕获其他未知错误 (如 JSON 解析错误)
            print(f"❌ 处理高德路径规划 API 响应时发生错误: {e}")
            return None

    return None  # 如果循环自然结束（不应该发生），返回 None


CITY_TO_PRIMARY_IATA = {
    "北京": "PEK",
    "上海": "PVG",
    "深圳": "SZX",
    "广州": "CAN",
    "杭州": "HGH",
    "成都": "CTU"
}


def get_iata_code(city_name: str) -> Optional[str]:
    """根据城市名获取其主要 IATA 代码。"""
    return CITY_TO_PRIMARY_IATA.get(city_name.strip(), None)


def query_flight_api(origin: str, destination: str, date: str) -> List[Dict]:
    """
    使用 SerpApi 的 google_flights 引擎查询航班，输入使用 IATA 代码。
    """
    print(f"✈️ 正在查询 {origin} 到 {destination} 的航班，日期: {date}")

    # 1. IATA 代码转换 (核心步骤)
    departure_iata = get_iata_code(origin)
    arrival_iata = get_iata_code(destination)

    if not departure_iata or not arrival_iata:
        print(f"⚠️ 无法获取 {origin} 或 {destination} 的 IATA 代码，跳过航班查询。")
        return []


    params = {
        "engine": "google_flights",
        "departure_id": departure_iata,  # ❗ 使用 IATA 代码
        "arrival_id": arrival_iata,  # ❗ 使用 IATA 代码
        "outbound_date": date,
        "currency": "CNY",
        "hl": "zh-cn",
        "api_key": SERPAPI_FLIGHTS_API_KEY,
        "type": "2",  # 单程
        "stops": "0"  # 直飞
    }

    try:
        # 增加延时以缓解 QPS 限制问题
        time.sleep(1)

        response = requests.get(GOOGLE_FLIGHTS_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        all_flights = []

        # 收集所有航班列表：'best_flights' 和 'other_flights'
        flight_groups = data.get('best_flights', []) + data.get('other_flights', [])

        for group in flight_groups:
            # 简化：只处理直飞或单段行程 (即 group['flights'] 列表只有一个元素)
            flight_segment = group.get('flights', [{}])[0]

            if not flight_segment or not group.get('price'):
                continue

            # --- 提取和解析时间 ---
            departure_dt_str = flight_segment.get('departure_airport', {}).get('time')
            arrival_dt_str = flight_segment.get('arrival_airport', {}).get('time')

            if not departure_dt_str or not arrival_dt_str:
                continue

            # SerpApi 格式通常为 'YYYY-MM-DD HH:MM'
            time_format = '%Y-%m-%d %H:%M'

            try:
                departure_dt = datetime.strptime(departure_dt_str, time_format)
                arrival_dt = datetime.strptime(arrival_dt_str, time_format)
            except ValueError:
                continue

            # 航班的枢纽是 IATA 代码
            departure_iata = flight_segment.get('departure_airport', {}).get('id')
            arrival_iata = flight_segment.get('arrival_airport', {}).get('id')

            # --- 构造标准化字典 ---
            all_flights.append({
                # 保持 type 字段一致
                "type": "Flight",
                # 保持 id 字段一致 (航班号)
                "id": flight_segment.get('flight_number', 'N/A'),

                # 保持时刻字段一致
                "departure_time": departure_dt.strftime('%H:%M'),
                "arrival_time": arrival_dt.strftime('%H:%M'),

                # 保持价格、时长字段一致
                "price": group['price'],
                "duration": group['total_duration'],  # SerpApi返回的是分钟，与高铁 API 的格式可能不完全一致，但类型一致

                # 保持枢纽字段一致 (IATA 代码对应火车站名称)
                "departure_hub": departure_iata,
                "arrival_hub": arrival_iata,

                # 保持日期字段一致
                "departure_date": departure_dt.strftime('%Y-%m-%d'),
                "arrival_date": arrival_dt.strftime('%Y-%m-%d')
            })

        print(f"✅ 航班查询成功。共找到 {len(all_flights)} 个航班选项。")
        return all_flights

    except requests.exceptions.RequestException as e:
        print(f"❌ SerpApi 请求失败: {e}")
        return []
    except Exception as e:
        print(f"❌ 处理 SerpApi 响应时发生错误: {e}")
        return []


def query_train_api(origin: str, destination: str, date: str, filter: str = "G") -> List[Dict]:
    """
    实际调用聚合数据 API 进行高铁查询，返回 List[Dict]。
    """
    print(f"🚄 正在调用聚合 API 查询 {date} 从 {origin} 到 {destination} 的高铁")

    if not JUHE_TRAIN_API_KEY:
        print("❌ 致命错误：JUHE_TRAIN_API_KEY 未配置，使用模拟数据。")
        # 回退逻辑保持简单
        return [
            {"type": "Train", "id": "G101", "departure_time": "07:30", "arrival_time": "13:30", "price": 600,
             "duration": "6h00m", "departure_hub": f"{origin} 火车站", "arrival_hub": f"{destination} 火车站"},
        ]

    params = {
        "key": JUHE_TRAIN_API_KEY,
        "search_type": "1",
        "departure_station": origin,
        "arrival_station": destination,
        "date": date,
        "enable_booking": "1",
        "filter": filter  # 修正 2：应用筛选条件
    }

    try:
        # ... (API 调用和响应处理逻辑保持不变) ...
        response = requests.get(JUHE_TRAIN_QUERY_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("error_code") != 0:
            print(f"⚠️ 聚合数据高铁查询失败。日期: {date}, 原因: {data.get('reason')}")
            return []

        # 转换 API 返回结果为我们内部需要的 List[Dict] 格式
        train_options = []
        for item in data.get("result", []):

            second_class_price_item = next(
                (p for p in item.get("prices", []) if p.get("seat_name") == "二等座"),
                {"price": 0}
            )

            departure_time_str = item["departure_time"]
            arrival_time_str = item["arrival_time"]

            # 1. 创建出发和到达的 datetime 对象 (初始都假设在出发日期)
            departure_date_str = date
            start_dt = datetime.strptime(f"{departure_date_str} {departure_time_str}", '%Y-%m-%d %H:%M')
            arrival_dt = datetime.strptime(f"{departure_date_str} {arrival_time_str}", '%Y-%m-%d %H:%M')

            # 2. 跨天修正：如果到达时刻早于出发时刻，则到达日期加一天
            if arrival_dt < start_dt:
                arrival_dt += timedelta(days=1)

            # 3. 提取最终的到达日期字符串
            arrival_date_str = arrival_dt.strftime('%Y-%m-%d')

            # --- 💡 修正点：将日期信息添加到字典中 ---
            train_options.append({
                "type": "Train",
                "id": item["train_no"],

                # 原始 API 返回的时刻
                "departure_time": departure_time_str,
                "arrival_time": arrival_time_str,

                "price": second_class_price_item["price"],
                "duration": item["duration"],
                "departure_hub": item["departure_station"],
                "arrival_hub": item["arrival_station"],

                # ✅ 关键新增字段：让 LLM 知道班次对应的日期
                "departure_date": departure_date_str,
                "arrival_date": arrival_date_str
            })

        return train_options

    except requests.exceptions.RequestException as e:
        print(f"❌ 聚合数据 API 请求失败 (网络/超时/HTTP错误): {e}")
        return []
    except Exception as e:
        print(f"❌ 处理聚合数据 API 响应时发生错误: {e}")
        return []







