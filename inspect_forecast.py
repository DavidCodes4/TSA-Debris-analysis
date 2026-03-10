import json, pandas as pd

with open('Output/forecast_results.json') as f:
    d = json.load(f)

print("TOP LEVEL KEYS:", list(d.keys()))
print("sample_id:", d['sample_id'])
print("forecast_steps:", d['forecast_steps'])
print("best_model:", d['best_model'])
print()

bm = d['benchmark']
print("BENCHMARK TYPE:", type(bm))
if isinstance(bm, list):
    for row in bm:
        print(" ", row)
elif isinstance(bm, dict):
    print(pd.DataFrame(bm).to_string())

print()
print("FORECAST KEYS:")
fcast = d['forecasts']
for k, v in fcast.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} pts | first3 = {v[:3]}")
    else:
        print(f"  {k}: {type(v)}")
