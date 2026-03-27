import zstandard as zstd
import json
import io

file_path = "../../../Downloads/reddit/comments/RC_2015-06.zst"

with open(file_path, "rb") as f:
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)  # 🔥 FIX
    stream = dctx.stream_reader(f)
    text_stream = io.TextIOWrapper(stream, encoding="utf-8")

    for i, line in enumerate(text_stream):
        data = json.loads(line)

        print({
            "author": data.get("author"),
            "subreddit": data.get("subreddit"),
            "parent_id": data.get("parent_id"),
            "score": data.get("score"),
            "created_utc": data.get("created_utc")
        })

        if i == 10:
            break