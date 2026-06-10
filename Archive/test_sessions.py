import sys
import json
sys.path.insert(0, r'c:\Users\Irak\Desktop\AI_Agent\DigitalHistory\openrecall')
from app import app

client = app.test_client()
response = client.get('/api/activity_sessions')
data = response.get_json()

print(f"Total sessions returned: {len(data) if data else 0}")
if data:
    for i, session in enumerate(data[:3]): # print details for first 3 sessions
        print(f"\n--- Session {i+1} ---")
        print(f"Category: {session.get('category')}")
        print(f"Summary: {session.get('summary')}")
        
        detail_points = session.get('detail_points', [])
        print(f"Detail Points Count: {len(detail_points)}")
        
        for idx, pt in enumerate(detail_points):
            print(f"  {idx+1}. [{pt.get('icon')}] {pt.get('time')} - {pt.get('action')}")
