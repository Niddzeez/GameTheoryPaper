import zstandard as zstd
import json
import io
from collections import defaultdict
import matplotlib.pyplot as plt

file_path = "../../../Downloads/reddit/comments/RC_2015-06.zst"

target_subs = {"fatpeoplehate", "politics", "worldnews"}

# 🔴 Define explicit time window (1 week)
START = 1433116800  # June 1, 2015
END = START + 7 * 24 * 3600  # 7 days

data_list = []

with open(file_path, "rb") as f:
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    stream = dctx.stream_reader(f)
    text_stream = io.TextIOWrapper(stream, encoding="utf-8")

    for line in text_stream:
        data = json.loads(line)

        # ✅ Filter by subreddit
        if data.get("subreddit") not in target_subs:
            continue

        # ✅ Filter by time window
        t = int(data["created_utc"])
        if t < START or t > END:
            continue

        data_list.append({
            "author": data["author"],
            "parent_id": data["parent_id"],
            "score": data["score"],
            "time": t,
            "subreddit": data["subreddit"]
        })

print("Collected:", len(data_list))

# ----------------------------
# 🔴 TIME SERIES
# ----------------------------

time_series = defaultdict(list)

for d in data_list:
    day = d["time"] // (24 * 3600)
    time_series[day].append(d)

x_t = {}  # normalized activity
T_t = {}  # trust

# 🔥 Normalize x(t)
max_activity = max(len(v) for v in time_series.values())

for day, items in sorted(time_series.items()):
    x_t[day] = len(items) / max_activity  # normalized [0,1]
    T_t[day] = sum(d["score"] for d in items) / len(items)

print("\nTime Series:")
for day in sorted(x_t.keys()):
    print(f"Day {day}: x={x_t[day]:.3f}, T={T_t[day]:.3f}")

# ----------------------------
# 🔴 NETWORK (still partial)
# ----------------------------

edges = []

for d in data_list:
    if d["parent_id"].startswith("t1_"):
        edges.append((d["author"], d["parent_id"]))

print("\nEdges:", len(edges))