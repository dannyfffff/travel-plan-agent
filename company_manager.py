# company_manager.py
import json
from typing import List, Dict, Any
from api_tools import amap_geocode

# 数据文件路径
DATA_FILE = 'companies_data.json'


def _load_data() -> Dict[str, List[Dict[str, Any]]]:
    """从 JSON 文件加载所有企业数据。"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，返回一个空字典，以便后续创建
        return {}
    except json.JSONDecodeError:
        print(f"❌ 警告: {DATA_FILE} 文件内容格式错误，将使用空数据。")
        return {}


def _save_data(data: Dict[str, List[Dict[str, Any]]]) -> None:
    """将所有企业数据保存到 JSON 文件。"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# --- CRUD 操作函数 ---

# C: Create/Add
def add_company(city: str, name: str, address: str) -> None:
    """
    向指定城市添加一个新企业。
    ID格式为：[城市名][三位序号]，并根据地址自动获取经纬度。
    """
    data = _load_data()

    # 城市名作为字典的键，保持与数据文件格式一致（例如："深圳"）
    if city not in data:
        data[city] = []

    prefix = city
    new_index = len(data[city]) + 1
    new_id = f"{prefix}{new_index:03}"

    # 2. 自动获取经纬度
    coords = amap_geocode(address, city)
    lat = coords['lat'] if coords else 0.0
    lon = coords['lon'] if coords else 0.0

    # 3. 构造完整的企业信息字典
    company_info = {
        'id': new_id,  # 例如：深圳005
        'name': name,
        'address': address,
        'lat': lat,
        'lon': lon,
        'industry': '待分类',
        'description': '用户新增'
    }

    data[city].append(company_info)
    _save_data(data)
    print(f"✅ 成功添加企业: {name} (ID: {new_id}) 到 {city}。")
    print(f"   -> 经纬度自动获取: Lat={lat}, Lon={lon}")



# R: Read/Get (会被 planning_tools 调用)
def get_companies_by_city(city: str) -> List[Dict[str, Any]]:
    """获取指定城市的所有企业列表。"""
    data = _load_data()
    return data.get(city, [])


# U: Update
def update_company(city: str, company_id: str, updates: Dict[str, Any]) -> bool:
    """更新指定城市和ID的企业信息。"""
    data = _load_data()
    if city not in data:
        return False

    for i, company in enumerate(data[city]):
        if company.get('id') == company_id:
            data[city][i].update(updates)
            _save_data(data)
            print(f"✅ 成功更新企业: {company['name']} ({company_id})")
            return True

    return False


# D: Delete
def delete_company(city: str, company_id: str) -> bool:
    """删除指定城市和ID的企业。"""
    data = _load_data()
    if city not in data:
        return False

    initial_count = len(data[city])
    data[city] = [c for c in data[city] if c.get('id') != company_id]

    if len(data[city]) < initial_count:
        _save_data(data)
        print(f"✅ 成功删除企业 ID: {company_id} 从 {city}")
        return True

    return False


COMPANIES_DB = _load_data()