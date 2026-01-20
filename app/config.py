import os
from datetime import datetime

# football-data.org
FD_BASE = 'https://api.football-data.org/v4'
FD_API_TOKEN = os.getenv('FD_API_TOKEN', 'YOUR_API_KEY')
FD_SEASON = os.getenv('FD_SEASON', '2025')

# data dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
os.makedirs(SAVE_DIR, exist_ok=True)

def today_stamp() -> str:
    return datetime.now().strftime('%d-%m-%Y')
