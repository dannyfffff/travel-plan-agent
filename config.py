# config.py
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
JUHE_TRAIN_API_KEY = os.getenv("JUHE_TRAIN_API_KEY")
SERPAPI_FLIGHTS_API_KEY = os.getenv("SERPAPI_FLIGHTS_API_KEY")

LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.5

# --- 外部服务 URL ---
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_ROUTE_URL = "https://restapi.amap.com/v3/direction/driving"
JUHE_TRAIN_QUERY_URL = "https://apis.juhe.cn/fapigw/train/query"
GOOGLE_FLIGHTS_URL = "https://serpapi.com/search.json"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 假设权重和固定拜访时间
WEIGHTS = {
    'alpha': 0.5,
    'beta': 0.3,
    'gamma': 0.1,  # 每多一分钟总驾车时间，惩罚 0.1 分
    'delta': 0.05  # 每多一分钟可用时间，奖励 0.05 分
}

# --- 规划约束/常数 ---
PRE_DEPARTURE_BUFFER_MINUTES = 90
POST_ARRIVAL_BUFFER_MINUTES = 30#参会缓冲时间
COMPANY_VISIT_DURATION_MINUTES = 45#企业调研时间
DEFAULT_REFERENCE_COMMUTE_MINUTES = 45.0
