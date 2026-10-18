import json
import threading
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
SWIPE_FILE = BASE / "data" / "swipes.jsonl"   # 每行一条记录（JSON Lines）
SWIPE_FILE.parent.mkdir(exist_ok=True)

# ---- 职业故事存放在 data/careers.json，改内容不需要动这个文件 ----
CAREERS_FILE = BASE / "data" / "careers.json"
with CAREERS_FILE.open(encoding="utf-8") as f:
    CAREERS = json.load(f)
CAREER_MAP = {c["id"]: c for c in CAREERS}

app = FastAPI(title="Career Path Explorer")


# ---- 态度分数规则（只在后端定义，前端只上报“做了什么操作”）----
SCORES = {
    "swipe_like": 5,       # Tinder 画面：感兴趣
    "swipe_nope": -5,      # Tinder 画面：不感兴趣
    "detail_exit": 3,      # 打开详情页，但没表态就退出
    "detail_like": 8,      # 详情页：感兴趣
    "detail_neutral": 0,   # 详情页：一般（规则未指定，暂定 0）
    "detail_nope": -8,     # 详情页：不感兴趣
}


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
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": swipe.action,
        "score": SCORES[swipe.action],
        "career_id": career["id"],
        "title": career["title"],
        "abstract": career["abstract"],
        "story": career["story"],
        "feature": career.get("feature", []),
        "schedule": career.get("schedule", []),
    }
    with SWIPE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "score": record["score"]}


@app.get("/api/swipes")
def read_swipes():
    if not SWIPE_FILE.exists():
        return []
    with SWIPE_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- 用户自己选择的偏好关键词：data/userdefkeywords.json ----
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
    chosen = [k for k in dict.fromkeys(pick.keywords) if k in allowed]  # 去重，并只接受该职业的 feature
    if not chosen:
        return {"ok": True, "saved": []}
    with _kw_lock:
        data = _load_keywords()
        for k in chosen:
            data["keywords"][k] = data["keywords"].get(k, 0) + 1
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
    with _kw_lock:
        if req.action == "explore":                       # 好みの頻度に +3
            data = _load_keywords()
            data["keywords"][kw] = data["keywords"].get(kw, 0) + 3
            data["history"].append({"timestamp": now, "source": "explore_map", "keywords": [kw], "increment": 3})
            _save_json(KEYWORD_FILE, data)
        else:                                             # 苦手として escapekeyword.json に記録
            data = json.loads(ESCAPE_FILE.read_text(encoding="utf-8")) if ESCAPE_FILE.exists() else {"keywords": {}, "history": []}
            data["keywords"][kw] = data["keywords"].get(kw, 0) + 1
            data["history"].append({"timestamp": now, "source": "explore_map", "keywords": [kw]})
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
        left = kws.get(k, 0) - weight
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
