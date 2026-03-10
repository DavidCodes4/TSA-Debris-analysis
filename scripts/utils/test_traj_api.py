import sys
sys.path.insert(0, '/home/roshant/TSA')
from dashboard_api import app

client = app.test_client()

r = client.get('/api/debris/trajectories/all?downsample=50&limit=3')
import json
data = json.loads(r.data)
print('HTTP:', r.status_code)
print('count:', data.get('count'))
if data.get('objects'):
    o = data['objects'][0]
    print('norad_id:', o['norad_id'], '| regime:', o['regime'], '| alt:', o['mean_alt_km'])
    print('n_points:', o['n_points'])
    print('lats[:3]:', o['lats'][:3])
    print('lons[:3]:', o['lons'][:3])
else:
    print('RESPONSE:', data)
