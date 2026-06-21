from ..helpers.config import get_settings
import os
import random
import string


class BaseController:
    def __init__(self):

        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(
            os.path.dirname(__file__)
        )  # get the dir of the whole project so from src

    def get_lsoa_from_police_station(self, police_station: str, lsoa_map: dict) -> list:
        key = police_station.strip().lower().replace(" ", "-")
        force_data = lsoa_map.get(key, [])

        hotspots = force_data.get("hotspots", [])
        non_hotspots = force_data.get("non_hotspots", [])

        return hotspots + non_hotspots
    
    def get_lsoa_status_from_police_station(self, police_station: str, lsoa_map: dict) -> dict:
        key = police_station.strip().lower().replace(" ", "-")
        force_data = lsoa_map.get(key, {})
        
        hotspot_codes = set(force_data.get("hotspots", []))
        non_hotspot_codes = set(force_data.get("non_hotspots", []))

        lsoa_status = {}

        for code in hotspot_codes:
            lsoa_status[code] = True

        for code in non_hotspot_codes:
            lsoa_status[code] = False

        return lsoa_status
