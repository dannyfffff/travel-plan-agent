# state.py
from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime



# --- 定义通用数据结构 ---

class Location(TypedDict):
    """地理位置信息"""
    city: str
    address: Optional[str]
    name: Optional[str]
    # 实际应用中需要经纬度
    lat: Optional[float]
    lon: Optional[float]


class ItineraryItem(TypedDict):
    """行程中的一个活动或交通段"""
    type: str  # 'transport', 'company_visit', 'meeting', 'hotel'
    description: str
    start_time: datetime
    end_time: datetime
    location: Location
    details: Dict[str, Any]


# --- 定义 LangGraph 状态 ---

class TravelPlanState(TypedDict):
    """
    LangGraph 共享状态，用于在节点间传递数据。
    """
    # 初始用户输入和校验后的数据
    user_input: str
    user_data: Dict[str, Any]
    # 交通查询结果
    flight_options: List[Dict]
    train_options: List[Dict]

    # LLM选定的交通方案--原始方案和最终方案
    selected_option_raw: Optional[Dict[str, Any]]
    selected_transport: Optional[ItineraryItem]

    # 会议前行程、会议、会议后行程
    pre_meeting_route: List[ItineraryItem]
    post_meeting_route: List[ItineraryItem]

    # 最终行程
    final_itinerary: List[ItineraryItem]
    final_itinerary_report: str

    # 关键地点信息
    home_location: Location
    meeting_location: Location
    hotel_location: Location

    # 流程控制/辅助信息
    error_message: Optional[str]

