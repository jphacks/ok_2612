import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agents
import selector

BASE = Path(__file__).parent
ESCAPE_FILE = BASE / "data" / "escapekeyword.json"   # 「ちょっと違うかも」を押したキーワード
SWIPE_FILE = BASE / "data" / "swipes.jsonl"   # 1行に1レコード（JSON Lines）
SWIPE_FILE.parent.mkdir(exist_ok=True)

# ---- 職業ストーリーは data/careers.json に置く。内容を変えるときにこのファイルを触る必要はない ----
CAREERS_FILE = BASE / "data" / "careers.json"
with CAREERS_FILE.open(encoding="utf-8") as f:
    CAREERS = json.load(f)
CAREER_MAP = {c["id"]: c for c in CAREERS}

@asynccontextmanager
async def lifespan(app):
    start_environment_monitor()      # 環境モニターを main と一緒に起動する（失敗してもコンソールに表示するだけで、主プログラムには影響しない）
    try:
        yield
    finally:
        stop_environment_monitor()


app = FastAPI(title="Career Path Explorer", lifespan=lifespan)


# ---- 態度スコアのルール（バックエンドだけで定義。フロントは「何をしたか」だけを送る）----
SCORES = {
    "swipe_like": 5,       # Tinder 画面：気になる
    "swipe_nope": -5,      # Tinder 画面：興味なし
    "detail_exit": 3,      # 詳細ページを開いたが、態度を示さずに閉じた
    "detail_like": 8,      # 詳細ページ：気になる
    "detail_neutral": 0,   # 詳細ページ：普通（ルール未指定のため暫定で 0）
    "detail_nope": -8,     # 詳細ページ：興味なし
}


# ---- 環境フラグ: data/environment_flag.json ----
# environment_monitor.py(main と一緒に起動する別プロセス)が異常な音量変化を検出すると flag を 1 にする。
# flag が 1 の間に行われた「好き/嫌い」は、増減値を元の 70% にする。
# モニターの起動失敗やフラグファイルの読み込み失敗は、コンソールに出すだけで主プログラムは止めない。
ENV_FLAG_FILE = BASE / "data" / "environment_flag.json"
ENV_FACTOR = 0.7
ENV_STALE_SEC = 10          # updated_at がこれより古いまま flag=1 なら「モニター停止」とみなして無視する
_env_log_at = {}
_monitor_proc = None
_monitor_stopping = False


def _env_log(msg, key=None, every=30):
    """コンソールに出すだけ(同じ種類のメッセージは every 秒に1回まで)。表示に失敗しても止まらない"""
    key = key or msg
    now = time.time()
    if now - _env_log_at.get(key, 0) >= every:
        _env_log_at[key] = now
        try:
            print(f"[環境モニター] {msg}", flush=True)
        except Exception:
            pass


def env_weight():
    """(増減値にかける係数, 状態名)。flag=1 なら 0.7、それ以外は 1.0。
    どんな問題があってもコンソール表示だけで必ず 1.0 を返し、主プログラムを止めない"""
    try:
        f = json.loads(ENV_FLAG_FILE.read_text(encoding="utf-8"))
        if f.get("flag") not in (1, True):
            return 1.0, None
        upd = f.get("updated_at")
        if upd is not None and time.time() - float(upd) > ENV_STALE_SEC:
            _env_log("flag=1 のまま %d 秒以上更新されていません（モニターが停止した可能性があります）。このフラグは無視し、通常の重みで処理します" % ENV_STALE_SEC, "stale")
            return 1.0, None
        state = f.get("state")
        return ENV_FACTOR, (state[:50] if isinstance(state, str) else None)
    except Exception as e:
        _env_log(f"{ENV_FLAG_FILE} を読み込めません（{type(e).__name__}: {e}）。通常の重み(1.0)で続行します", "read")
    return 1.0, None


def _write_flag_file(state):
    ENV_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _save_json(ENV_FLAG_FILE, {"flag": 0, "state": state, "updated_at": None, "flagged_at": None, "owner_pid": os.getpid()})


def _watch_monitor(p):
    code = p.wait()
    if not _monitor_stopping:
        _env_log(f"環境モニターが終了しました（終了コード {code}）。主プログラムはそのまま動作し、選択は通常の重みで記録します", "exit", every=0)


def start_environment_monitor():
    global _monitor_proc
    try:
        _write_flag_file("初期値")                      # 起動時にフラグファイルを作成する（flag の既定値は 0）
        _env_log(f"{ENV_FLAG_FILE} を作成しました（flag=0）", "init", every=0)
    except Exception as e:
        _env_log(f"フラグファイルを作成できませんでした：{e}。主プログラムの動作には影響しません", "init", every=0)
    if os.environ.get("ENV_MONITOR", "1") == "0":
        _env_log("ENV_MONITOR=0 が指定されたため、環境モニターは起動しません", "off", every=0)
        return
    script = BASE / "environment_monitor.py"
    try:
        if not script.exists():
            _env_log(f"{script} が見つからないため、環境モニターを起動しませんでした。主プログラムの動作には影響しません", "start", every=0)
            return
        (BASE / "data").mkdir(exist_ok=True)
        _monitor_proc = subprocess.Popen(
            [sys.executable, str(script), "--flag-file", str(ENV_FLAG_FILE), "--owner-pid", str(os.getpid()), "--quiet"],
            cwd=str(BASE / "data"))                     # environment_log.csv も data に書き出される
        _env_log(f"環境モニターを起動しました（pid {_monitor_proc.pid}）", "start", every=0)
        threading.Thread(target=_watch_monitor, args=(_monitor_proc,), daemon=True, name="env-monitor-watch").start()
    except Exception as e:
        _env_log(f"環境モニターの起動に失敗しました：{e}。主プログラムの動作には影響しません", "start", every=0)


def stop_environment_monitor():
    global _monitor_stopping
    _monitor_stopping = True
    p = _monitor_proc
    try:
        if p is not None and p.poll() is None:
            p.terminate()
            p.wait(timeout=3)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    try:
        _write_flag_file("停止")                        # 終了時に flag を 0 に戻す
    except Exception:
        pass


@app.get("/api/environment")
def environment_status():
    """診断用: 環境フラグをこのサーバーがどう見ているか。ブラウザで /api/environment を開いて確認できる"""
    factor, state = env_weight()
    info = {"flag_file": str(ENV_FLAG_FILE), "exists": ENV_FLAG_FILE.exists(), "flag": None, "factor_now": factor, "state": state,
            "monitor_process_running": _monitor_proc is not None and _monitor_proc.poll() is None, "monitor_alive": None}
    try:
        f = json.loads(ENV_FLAG_FILE.read_text(encoding="utf-8"))
        info["flag"] = f.get("flag")
        upd = f.get("updated_at")
        info["monitor_alive"] = upd is not None and (time.time() - float(upd)) <= 5      # モニターは毎秒書き込む
    except Exception:
        pass
    return info


class Swipe(BaseModel):
    career_id: int
    action: Literal["swipe_like", "swipe_nope", "detail_exit",
                    "detail_like", "detail_neutral", "detail_nope"]


def _read_kw(path):
    """{keyword: count}。ファイルが無い・壊れている場合は空"""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("keywords", {})
    except (OSError, ValueError, AttributeError):
        return {}


@app.get("/api/careers")
def list_careers():
    return selector.pick_batch(CAREERS, _read_kw(KEYWORD_FILE), _read_kw(ESCAPE_FILE))


@app.post("/api/swipes")
def save_swipe(swipe: Swipe):
    career = CAREER_MAP.get(swipe.career_id)
    if not career:
        raise HTTPException(404, "career not found")
    w, env_state = env_weight()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": swipe.action,
        "score": round(SCORES[swipe.action] * w, 2),     # flag=1(異常な音)の間は ×0.7
        "base_score": SCORES[swipe.action],
        "env_weight": w,
        "env_state": env_state,
        "career_id": career["id"],
        "title": career["title"],
        "abstract": career["abstract"],
        "story": career["story"],
        "feature": career.get("feature", []),
        "schedule": career.get("schedule", []),
    }
    if w < 1:
        _env_log(f"{env_state}：今回の選択の重み ×{w}（{swipe.action}：{record['base_score']} → {record['score']}）", "applied", every=0)
    with SWIPE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "score": record["score"]}


@app.get("/api/swipes")
def read_swipes():
    if not SWIPE_FILE.exists():
        return []
    with SWIPE_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- ユーザー自身が選んだ好みのキーワード：data/userdefkeywords.json ----
KEYWORD_FILE = BASE / "data" / "userdefkeywords.json"
_kw_lock = threading.Lock()


class KeywordPick(BaseModel):
    career_id: int
    keywords: list[str]


def _save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_keywords():
    if KEYWORD_FILE.exists():
        return json.loads(KEYWORD_FILE.read_text(encoding="utf-8"))
    return {"keywords": {}, "history": []}


@app.post("/api/keywords")
def save_keywords(pick: KeywordPick):
    career = CAREER_MAP.get(pick.career_id)
    if not career:
        raise HTTPException(404, "career not found")
    allowed = career.get("feature", [])
    chosen = [k for k in dict.fromkeys(pick.keywords) if k in allowed]  # 重複を除き、その職業の feature にあるものだけを受け付ける
    if not chosen:
        return {"ok": True, "saved": []}
    w, _ = env_weight()
    with _kw_lock:
        data = _load_keywords()
        for k in chosen:
            data["keywords"][k] = round(data["keywords"].get(k, 0) + w, 2)
        data["history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "career_id": career["id"],
            "title": career["title"],
            "keywords": chosen,
        })
        _save_json(KEYWORD_FILE, data)
    return {"ok": True, "saved": chosen}


@app.get("/api/keywords")
def read_keywords():
    with _kw_lock:
        return _load_keywords()


# ---- 職業探索マップの吹き出し: 「この傾向をもっと探る」/「ちょっと違うかも」----
class KeywordAction(BaseModel):
    keyword: str
    action: Literal["explore", "escape"]


@app.post("/api/explore/keyword")
def explore_keyword(req: KeywordAction):
    kw = req.keyword.strip()
    if not agents.CANDIDATES_FILE.exists():
        raise HTTPException(400, "先に職業探索マップを開いてください。")
    doc = json.loads(agents.CANDIDATES_FILE.read_text(encoding="utf-8"))
    known = {k["keyword"] for k in [doc.get("main_keyword")] + doc.get("other_positive", [])
             + doc.get("negative", []) + doc.get("explore", []) if k}
    if kw not in known:
        raise HTTPException(404, "keyword not found")
    now = datetime.now(timezone.utc).isoformat()
    w, _ = env_weight()
    with _kw_lock:
        if req.action == "explore":                       # 好みの頻度に +3
            data = _load_keywords()
            data["keywords"][kw] = round(data["keywords"].get(kw, 0) + 3 * w, 2)
            data["history"].append({"timestamp": now, "source": "explore_map", "keywords": [kw], "increment": round(3 * w, 2)})
            _save_json(KEYWORD_FILE, data)
        else:                                             # 苦手として escapekeyword.json に記録
            data = json.loads(ESCAPE_FILE.read_text(encoding="utf-8")) if ESCAPE_FILE.exists() else {"keywords": {}, "history": []}
            data["keywords"][kw] = round(data["keywords"].get(kw, 0) + w, 2)
            data["history"].append({"timestamp": now, "source": "explore_map", "keywords": [kw], "increment": w})
            _save_json(ESCAPE_FILE, data)
    return {"ok": True, "keyword": kw, "action": req.action}


# ---- 設定画面: 専攻・態度・好き/興味なしキーワード（LLM不要。userdefkeywords.json / escapekeyword.json に直接反映）----
PROFILE_FILE = BASE / "data" / "profile.json"
W_LIKE, W_DISLIKE = 3, 1          # 申告したキーワードを各ファイルの回数に加算する重み
MAX_KW, MAX_LEN = 10, 30


class Profile(BaseModel):
    major: str = ""
    major_attitude: Literal["", "like", "dislike"] = ""
    like_keywords: list[str] = []
    dislike_keywords: list[str] = []


def _norm_list(items, label):
    out = []
    for s in items:
        s = " ".join(s.split())
        if not s:
            continue
        if len(s) > MAX_LEN:
            raise HTTPException(400, f"キーワードは{MAX_LEN}文字以内にしてください。")
        if s not in out:
            out.append(s)
    if len(out) > MAX_KW:
        raise HTTPException(400, f"{label}は{MAX_KW}個までです。")
    return out


def _read_profile():
    try:
        return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"major": "", "major_attitude": "", "like_keywords": [], "dislike_keywords": [],
                "applied": {"like": [], "dislike": []}}


def _shift(path, old, new, weight, now):
    """前回の申告分(old)を取り消してから今回の申告分(new)を加算する。編集しても二重に積み上がらない"""
    if set(old) == set(new):
        return
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"keywords": {}, "history": []}
    kws = data.setdefault("keywords", {})
    for k in old:
        left = round(kws.get(k, 0) - weight, 2)
        if left > 0:
            kws[k] = left
        else:
            kws.pop(k, None)
    for k in new:
        kws[k] = kws.get(k, 0) + weight
    data.setdefault("history", []).append({"timestamp": now, "source": "settings", "keywords": new, "increment": weight})
    _save_json(path, data)


@app.get("/api/profile")
def get_profile():
    p = _read_profile()
    return {k: p[k] for k in ("major", "major_attitude", "like_keywords", "dislike_keywords")}


@app.put("/api/profile")
def put_profile(p: Profile):
    major = " ".join(p.major.split())
    if len(major) > MAX_LEN:
        raise HTTPException(400, f"専攻は{MAX_LEN}文字以内にしてください。")
    attitude = p.major_attitude if major else ""
    if major and not attitude:
        raise HTTPException(400, "専攻への態度（好き / 苦手）を選んでください。")
    likes = _norm_list(p.like_keywords, "好き・気になるキーワード")
    dislikes = _norm_list(p.dislike_keywords, "興味がないキーワード")
    eff_like = likes + ([major] if attitude == "like" and major not in likes else [])      # 専攻が「好き」→ 積極キーワード
    eff_dis = dislikes + ([major] if attitude == "dislike" and major not in dislikes else [])   # 「苦手」→ 消極キーワード
    clash = [k for k in eff_like if k in eff_dis]
    if clash:
        raise HTTPException(400, f"「{clash[0]}」が「好き」と「興味なし」の両方に入っています。どちらか一方にしてください。")
    now = datetime.now(timezone.utc).isoformat()
    with _kw_lock:
        old = _read_profile().get("applied", {"like": [], "dislike": []})
        _shift(KEYWORD_FILE, old.get("like", []), eff_like, W_LIKE, now)
        _shift(ESCAPE_FILE, old.get("dislike", []), eff_dis, W_DISLIKE, now)
        _save_json(PROFILE_FILE, {"major": major, "major_attitude": attitude, "like_keywords": likes, "dislike_keywords": dislikes,
                                  "applied": {"like": eff_like, "dislike": eff_dis}, "updated_at": now})
    return {"major": major, "major_attitude": attitude, "like_keywords": likes, "dislike_keywords": dislikes}


# ---- 職業探索マップ：multi-agent 分析 ----
_explore_lock = threading.Lock()


@app.post("/api/explore/run")
def explore_run():
    if not _explore_lock.acquire(blocking=False):
        raise HTTPException(409, "分析中です。しばらくお待ちください。")
    try:
        return agents.run_pipeline()
    except agents.PipelineError as e:
        raise HTTPException(e.status, str(e))
    finally:
        _explore_lock.release()


@app.get("/api/explore/result")
def explore_result():
    if not agents.CANDIDATES_FILE.exists():
        raise HTTPException(404, "まだ分析結果がありません。")
    return json.loads(agents.CANDIDATES_FILE.read_text(encoding="utf-8"))


# ---- キャリアビジョン：職種推薦 + スキル開発（Agent 4/5）----
_vision_lock = threading.Lock()


@app.post("/api/vision/run")
def vision_run():
    if not _vision_lock.acquire(blocking=False):
        raise HTTPException(409, "作成中です。しばらくお待ちください。")
    try:
        return agents.run_vision()
    except agents.PipelineError as e:
        raise HTTPException(e.status, str(e))
    finally:
        _vision_lock.release()


app.mount("/images", StaticFiles(directory=BASE / "images"), name="images")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
