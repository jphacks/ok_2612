import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
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


@app.get("/api/careers")
def list_careers():
    return CAREERS


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
        tmp = KEYWORD_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(KEYWORD_FILE)
    return {"ok": True, "saved": chosen}


@app.get("/api/keywords")
def read_keywords():
    with _kw_lock:
        return _load_keywords()


app.mount("/images", StaticFiles(directory=BASE / "images"), name="images")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
