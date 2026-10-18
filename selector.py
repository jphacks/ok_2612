"""カード選択: 1バッチ最大10件。
- userdefkeywords.json(好み) / escapekeyword.json(苦手) がどちらも無い、または空 → ランダム
- ある場合: 好みに沿うもの 70% + 好みとは一見無関係なもの 30%、苦手に近い職業は回避 → 最後にシャッフル
キーワードと職業(feature / abstract)の近さは、文字2-gram の一致度で計算する(LLM・API キーは不要)。"""
import random
import re

BATCH = 10
LIKE_RATIO = 0.7
AVOID_AT = 0.3          # 苦手キーワードとの近さがこれ以上の職業は回避する
_PUNCT = re.compile(r"[\s・、。，．,.!?！？「」『』()（）]")


def _clean(s):
    return _PUNCT.sub("", str(s or ""))


def _grams(s):
    s = _clean(s)
    if len(s) < 2:
        return {s} if s else set()
    g = {s[i:i + 2] for i in range(len(s) - 1)}
    content = {x for x in g if not all("\u3041" <= c <= "\u309f" for c in x)}   # ひらがなだけの2-gram(「する」等)は除く
    return content or g


def _sim(a, b):
    """キーワード同士の近さ 0〜1(完全一致=1、包含=0.8、それ以外は2-gramのDice係数)"""
    sa, sb = _clean(a), _clean(b)
    A, B = _grams(sa), _grams(sb)
    if not A or not B:
        return 0.0
    if sa == sb:
        return 1.0
    dice = 2 * len(A & B) / (len(A) + len(B))
    contained = len(sa) >= 2 and len(sb) >= 2 and (sa in sb or sb in sa)
    return max(dice, 0.8 if contained else 0.0)


def _rel(keyword, career):
    """キーワードと職業の近さ: feature との類似度、または abstract への含まれ具合(×0.6)の大きい方"""
    best = max((_sim(keyword, f) for f in career.get("feature") or []), default=0.0)
    K, T = _grams(keyword), _grams(career.get("abstract"))
    cov = len(K & T) / len(K) if K and T else 0.0
    return max(best, 0.6 * cov)


def pick_batch(careers, likes, dislikes, n=BATCH, rng=random):
    pool = list(careers)
    if not pool:
        return []
    bad = [k for k in (dislikes or {}) if _clean(k)]
    if bad:                                                   # 苦手は回避(全滅する場合だけは回避を諦める)
        pool = [c for c in pool if max(_rel(k, c) for k in bad) < AVOID_AT] or pool
    lw = [(k, float(v)) for k, v in (likes or {}).items() if _clean(k) and isinstance(v, (int, float)) and v > 0]
    tw = sum(w for _, w in lw)

    def like_rel(c):
        return sum(w * _rel(k, c) for k, w in lw) / tw if tw else 0.0

    scored = [(like_rel(c), rng.random(), c) for c in pool]   # 同点はランダム → 好みが無ければ実質ランダム
    target = min(n, len(pool))
    n_like = round(target * LIKE_RATIO)
    n_un = target - n_like
    scored.sort(key=lambda t: (-t[0], t[1]))
    liked, rest = scored[:n_like], scored[n_like:]
    rest.sort(key=lambda t: (t[0], t[1]))                     # 好みとの近さが低い順
    unrelated = rng.sample(rest[:max(n_un, len(rest) // 2)], n_un)
    chosen = [t[2] for t in liked + unrelated]
    rng.shuffle(chosen)
    return chosen
