from typing import List
from pydantic import BaseModel, Field

class PlannedVisit(BaseModel):
    name: str = Field(description="调研企业的完整名称。")
    order: int = Field(description="调研的顺序，从 1 开始。")

class PreMeetingPlanOutput(BaseModel):
    planned_visits: List[PlannedVisit] = Field(description="根据时间约束、顺路程度和调研价值优化的调研企业列表。")

class SelectedTransport(BaseModel):
    type: str = Field(description="选定的交通工具类型，如 'Flight' 或 'Train'。")
    id: str = Field(description="选定班次的唯一标识符/编号。")
    reasoning: str = Field(description="选择该班次的推理理由，需要解释如何满足调研时间最大化。")

class UserInputParams(BaseModel):
    """用户输入中必须提取的全部关键参数。"""
    origin_city: str = Field(description="出发城市名，例如 '上海'。")
    destination_city: str = Field(description="目的地城市名，例如 '深圳'。")
    departure_date: str = Field(description="出发日期，格式为 'YYYY-MM-DD'。")
    meeting_start: str = Field(description="会议开始日期和时间，精确到小时和分钟，格式为 'YYYY-MM-DD HH:MM'。")
    meeting_duration_h: int = Field(description="会议持续时长，以小时为单位。")
    home_address: str = Field(description="用户的出发地详细地址，例如 '上海市浦东新区川沙新镇黄赵路310号'。")
    meeting_address: str = Field(description="会议的详细地址。")
    hotel_address: str = Field(description="预订或计划入住的酒店详细地址。")

