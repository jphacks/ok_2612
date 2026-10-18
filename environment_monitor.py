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
# このプログラムは main.py の起動時にバックグラウンドで自動的に起動されます（単体でも実行できます: python environment_monitor.py）
#   --flag-file PATH   フラグファイルのパス（既定: このスクリプトのフォルダ/data/environment_flag.json）
#   --owner-pid PID    main.py から渡されます。フラグファイル内の owner_pid が別の値になった
#                      （= main が再起動された）場合、このプロセスは自動的に終了します
#   --quiet            1秒ごとの表示をやめ、起動情報・flag の変化・エラーだけを表示します（main.py からの起動時に使用）
#   --test-flag        マイクを使わずに「異常音」を模擬します（flag=1 を HOLD_SECONDS 秒間保ち、その後 0 に戻す）。動作確認用
# 環境変数 ENV_HOLD_SECONDS: 異常を検出した後、flag を 1 に保つ秒数（既定 30）
# ---------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--flag-file")
parser.add_argument("--owner-pid", type=int)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--test-flag", action="store_true")
args = parser.parse_args()

# 動作確認済みの Audiohub Nano 入力デバイス
DEVICE_INDEX = 1
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_SIZE = 4800  # 100 ms

# 状態判定のパラメータ：実験に合わせて調整してください
BASELINE_SECONDS = 30
SUDDEN_CHANGE_DB = 12.0
ELEVATED_DB = 5.0
HIGH_DB = 10.0

# ベースラインからの上昇がこの値以上なら「異常な音の変動」とみなす（既定 = 「音が著しく上昇」。「突然の増音」も含む）
RISK_CHANGE_DB = HIGH_DB
# 異常を検出した後、flag を 1 に保つ秒数。この後 0 に戻る
HOLD_SECONDS = float(os.environ.get("ENV_HOLD_SECONDS", 30))

FLAG_FILE = Path(
    args.flag_file
    or os.environ.get("ENV_FLAG_FILE")
    or Path(__file__).resolve().parent / "data" / "environment_flag.json"
).resolve()

last_abnormal_at = None   # 最後に異常を検出した時刻（epoch 秒）
current_flag = 0
superseded = False        # main が再起動され、新しい監視プロセスに引き継がれた
_last_warn = 0.0


def say(msg):
    if not args.quiet:
        print(msg)


def _r(x):
    return None if x is None else round(float(x), 2)


def _owner_in_file():
    try:
        return json.loads(FLAG_FILE.read_text(encoding="utf-8")).get("owner_pid")
    except Exception:
        return None


def _save(data):
    global _last_warn
    tmp = FLAG_FILE.with_suffix(".tmp")
    try:
        FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(FLAG_FILE)
        return True
    except OSError as e:
        # Windows では主プログラムが読み込み中だと書き込みに失敗することがある。次の1秒でまた書くので問題ない。
        # 失敗が続く場合だけコンソールに警告を出す（最大10秒に1回）
        if time.time() - _last_warn > 10:
            print(f"[環境モニター] 警告：フラグファイル {FLAG_FILE} に書き込めません：{e}")
            _last_warn = time.time()
        return False


def write_flag(state, level, change, abnormal):
    """1秒ごとにフラグファイルを書く。異常を検出 → flag=1 にして HOLD_SECONDS 秒間保ち、その後 0 に戻す。現在の flag を返す"""
    global last_abnormal_at, current_flag, superseded
    now = time.time()
    if args.owner_pid is not None:
        owner = _owner_in_file()
        if owner is not None and owner != args.owner_pid:   # main が再起動された。新しい main が新しい監視プロセスを起動する
            superseded = True
            print("[環境モニター] 主プログラムが再起動されたため、この監視プロセスは終了します")
            raise SystemExit(0)
    if abnormal:
        last_abnormal_at = now
    flag = 1 if (last_abnormal_at is not None and now - last_abnormal_at <= HOLD_SECONDS) else 0
    if flag != current_flag:
        print(f"[環境モニター] flag {current_flag} -> {flag}（{state}）")
        current_flag = flag
    _save({
        "flag": flag,                 # 0 = 正常、1 = 異常音を検出（主プログラムはこれを見て増減値を 70% にする）
        "state": state,
        "level_dbfs": _r(level),
        "change_db": _r(change),
        "updated_at": now,            # ハートビート：最後に書き込んだ時刻（主プログラムはこれで監視プロセスが生きているか判断する）
        "flagged_at": last_abnormal_at,
        "owner_pid": args.owner_pid,
    })
    return flag


def reset_flag(state="監視停止"):
    """終了時に flag を 0 に戻す（フラグファイルがすでに新しい main のものになっている場合は触らない）"""
    if superseded:
        return
    owner = _owner_in_file()
    if args.owner_pid is not None and owner is not None and owner != args.owner_pid:
        return
    _save({"flag": 0, "state": state, "level_dbfs": None, "change_db": None, "updated_at": None,
           "flagged_at": None, "owner_pid": args.owner_pid})


print(f"[環境モニター] フラグファイル: {FLAG_FILE}")
write_flag("起動", None, None, False)          # 起動と同時にフラグファイルを作成する（flag=0）

if args.test_flag:
    try:
        print(f"[環境モニター] テストモード：flag=1 を {HOLD_SECONDS:g} 秒間保ちます")
        end = time.time() + HOLD_SECONDS
        while time.time() < end:
            write_flag("テスト：異常音の模擬", 0.0, 99.0, True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        reset_flag()
    print("[環境モニター] テスト終了。flag を 0 に戻しました")
    sys.exit(0)

try:
    import sounddevice as sd
except Exception as e:                           # PortAudio が無い場合など：コンソールに表示するだけで、主プログラムには影響しない
    print(f"[環境モニター] sounddevice を読み込めません（{e}）。環境モニターは起動しませんでした（flag は 0 のままです）。")
    sys.exit(1)


def rms_dbfs(audio):
    audio = audio.astype(np.float64)
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(max(rms, 1e-12))

# 直近 30 秒の1秒ごとの音量を保存し、基準（ベースライン）の算出に使う
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
    "flag"
])

print("[環境モニター] 環境音の監視を開始しました。Ctrl+C で停止します。最初の 30 秒はベースラインの作成に使うので、普段どおりの環境を保ってください。")

try:
    with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
    ) as stream:

        while True:
            # 約 1 秒分の音声を集め、結合して RMS を計算する
            chunks = []
            for _ in range(10):
                audio, overflowed = stream.read(BLOCK_SIZE)
                chunks.append(audio)

            audio_1s = np.concatenate(chunks, axis=0)
            level = rms_dbfs(audio_1s)

            timestamp = datetime.now().isoformat(
                timespec="seconds"
            )

            # 現在より前の記録だけを基準として使う
            if len(baseline_history) >= 10:
                baseline = float(np.median(baseline_history))
                change = level - baseline

                # 直近の環境ベースラインに対する急上昇を検出する
                sudden = change >= SUDDEN_CHANGE_DB

                if change >= HIGH_DB:
                    state = "音が著しく上昇"
                elif change >= ELEVATED_DB:
                    state = "音がやや上昇"
                else:
                    state = "音はベースライン付近"

                if sudden:
                    state = "突然の増音を検出"

                flag = write_flag(state, level, change, change >= RISK_CHANGE_DB)

                say(
                    f"{timestamp} | "
                    f"音量 {level:6.1f} dBFS | "
                    f"ベースライン {baseline:6.1f} | "
                    f"変化 {change:+5.1f} dB | "
                    f"{state} | flag {flag}"
                )

                writer.writerow([
                    timestamp,
                    round(level, 2),
                    round(baseline, 2),
                    round(change, 2),
                    state,
                    sudden,
                    flag
                ])
                csv_file.flush()

            else:
                baseline = None
                change = None
                sudden = False
                state = "ベースライン作成中"
                flag = write_flag(state, level, None, False)

                say(
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
                    flag
                ])
                csv_file.flush()

            baseline_history.append(level)
            previous_history.append(level)

except KeyboardInterrupt:
    print("\n[環境モニター] 監視を終了しました。データは environment_log.csv に保存されています")

except Exception as e:
    print(f"\n[環境モニター] エラーが発生しました：{e}")

finally:
    csv_file.close()
    reset_flag()
