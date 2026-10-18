import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# 用法:
#   python environment_monitor.py                       正常监测
#   python environment_monitor.py --test-flag           不用麦克风，立刻写一次“有风险”标记（用来检查主程序是否读得到）
#   python environment_monitor.py --flag-file D:\...\data\environment_flag.json   指定标记文件路径
# 标记文件默认是“本脚本所在文件夹/data/environment_flag.json”，必须和主程序（main.py）所在文件夹下的 data 一致。
# 也可以用环境变量 ENV_FLAG_FILE 指定。
# ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--flag-file", help="标记文件路径")
parser.add_argument("--test-flag", action="store_true", help="写入一次“有风险”标记后退出")
args = parser.parse_args()

# 已经验证可以工作的 Audiohub Nano 输入设备
DEVICE_INDEX = 1
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_SIZE = 4800  # 100 ms

# 状态判断参数：后续可根据实验调整
BASELINE_SECONDS = 30
SUDDEN_CHANGE_DB = 12.0
ELEVATED_DB = 5.0
HIGH_DB = 10.0

# 相对基线升高达到这个值，就视为“有风险”（默认 = 声音显著升高，突然增响也包含在内）
RISK_CHANGE_DB = HIGH_DB

FLAG_FILE = Path(
    args.flag_file
    or os.environ.get("ENV_FLAG_FILE")
    or Path(__file__).resolve().parent / "data" / "environment_flag.json"
).resolve()
FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)

last_risky_at = None   # 最近一次“有风险”的时刻（epoch 秒）
_last_warn = 0.0


def _r(x):
    return None if x is None else round(float(x), 2)


def write_flag(state, level, change, risky):
    """每秒写一次标记文件（所以文件存在 = 监测脚本在运行）。
    主程序只看 risky_at：距现在 30 秒内算“有风险”。脚本停止后标记会自动失效，不需要清除。"""
    global last_risky_at, _last_warn
    now = time.time()
    if risky:
        last_risky_at = now
    data = {
        "updated_at": now,          # 心跳：最近一次写入的时刻
        "risky": bool(risky),       # 这一秒是否有风险
        "state": state,
        "level_dbfs": _r(level),
        "change_db": _r(change),
        "risky_at": last_risky_at,  # 最近一次有风险的时刻（从未有过则为 null）
        "risky_at_iso": (datetime.fromtimestamp(last_risky_at).astimezone().isoformat(timespec="seconds")
                         if last_risky_at else None),
    }
    tmp = FLAG_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(FLAG_FILE)
        return True
    except OSError as e:
        # Windows 上主程序恰好在读取时可能写入失败：下一秒会再写。持续失败则提示（最多每 10 秒一次）
        if now - _last_warn > 10:
            print(f"警告：无法写入标记文件 {FLAG_FILE}：{e}")
            _last_warn = now
        return False


print(f"标记文件: {FLAG_FILE}")

if args.test_flag:
    ok = write_flag("测试：模拟有风险", 0.0, 99.0, True)
    print("已写入测试标记。接下来 30 秒内在页面上做的选择，权重应为 0.8 倍。" if ok else "写入失败。")
    sys.exit(0 if ok else 1)

import sounddevice as sd  # noqa: E402  （--test-flag 不需要麦克风库，所以放在后面导入）


def rms_dbfs(audio):
    audio = audio.astype(np.float64)
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(max(rms, 1e-12))

# 保存最近 30 秒的每秒音量，用于建立参考基线
baseline_history = deque(maxlen=BASELINE_SECONDS)
previous_history = deque(maxlen=10)

csv_file = open(
    "environment_log.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
)
writer = csv.writer(csv_file)
writer.writerow([
    "timestamp",
    "level_dbfs",
    "baseline_dbfs",
    "change_db",
    "environment_state",
    "sudden_change",
    "risky"
])

print("正在监测环境声音。按 Ctrl+C 停止。")
print("前 30 秒用于建立初始基线，请保持正常环境。")

try:
    with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
    ) as stream:

        while True:
            # 收集约 1 秒音频，合并后计算 RMS
            chunks = []
            for _ in range(10):
                audio, overflowed = stream.read(BLOCK_SIZE)
                chunks.append(audio)

            audio_1s = np.concatenate(chunks, axis=0)
            level = rms_dbfs(audio_1s)

            timestamp = datetime.now().isoformat(
                timespec="seconds"
            )

            # 只使用当前时刻之前的记录作为参考
            if len(baseline_history) >= 10:
                baseline = float(np.median(baseline_history))
                change = level - baseline

                # 检测相对于近期环境基线的突增
                sudden = change >= SUDDEN_CHANGE_DB

                if change >= HIGH_DB:
                    state = "声音显著升高"
                elif change >= ELEVATED_DB:
                    state = "声音有所升高"
                else:
                    state = "声音接近基线"

                if sudden:
                    state = "检测到突然增响"

                risky = change >= RISK_CHANGE_DB
                write_flag(state, level, change, risky)

                print(
                    f"{timestamp} | "
                    f"音量 {level:6.1f} dBFS | "
                    f"基线 {baseline:6.1f} | "
                    f"变化 {change:+5.1f} dB | "
                    f"{state}"
                    f"{' | 有风险' if risky else ''}"
                )

                writer.writerow([
                    timestamp,
                    round(level, 2),
                    round(baseline, 2),
                    round(change, 2),
                    state,
                    sudden,
                    risky
                ])
                csv_file.flush()

            else:
                baseline = None
                change = None
                sudden = False
                state = "正在建立基线"
                write_flag(state, level, None, False)

                print(
                    f"{timestamp} | "
                    f"音量 {level:6.1f} dBFS | "
                    f"{state} ({len(baseline_history) + 1}/"
                    f"{BASELINE_SECONDS} 秒)"
                )

                writer.writerow([
                    timestamp,
                    round(level, 2),
                    "",
                    "",
                    state,
                    False,
                    False
                ])
                csv_file.flush()

            baseline_history.append(level)
            previous_history.append(level)

except KeyboardInterrupt:
    print("\n监测结束，数据已保存至 environment_log.csv")

except Exception as e:
    print(f"\n发生错误：{e}")

finally:
    csv_file.close()
