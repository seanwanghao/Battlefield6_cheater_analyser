#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# -----------------------------
# Benford + helpers
# -----------------------------
BENFORD_P = {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def first_digit(x: Any) -> Optional[int]:
    """Return first significant digit (1..9) or None."""
    try:
        if x is None or isinstance(x, bool):
            return None
        x = abs(float(x))
        if x == 0:
            return None
        s = f"{x:.12g}"
        if "e" in s or "E" in s:
            x = float(s)
            s = f"{x:.12f}".rstrip("0").rstrip(".")
        s = s.lstrip("0")
        if s.startswith("."):
            s = s[1:]
        m = re.match(r"([1-9])", s)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def benford_counts(values: List[Any]) -> Tuple[Dict[int, int], int]:
    counts = {d: 0 for d in range(1, 10)}
    n = 0
    for v in values:
        d = first_digit(v)
        if d is None:
            continue
        counts[d] += 1
        n += 1
    return counts, n


def benford_mad(counts: Dict[int, int], n: int) -> Optional[float]:
    if n <= 0:
        return None
    obs = {d: counts[d] / n for d in range(1, 10)}
    return sum(abs(obs[d] - BENFORD_P[d]) for d in range(1, 10)) / 9.0


def chi_square_stat(counts: Dict[int, int], n: int) -> Optional[float]:
    if n <= 0:
        return None
    stat = 0.0
    for d in range(1, 10):
        exp = BENFORD_P[d] * n
        if exp > 0:
            stat += (counts[d] - exp) ** 2 / exp
    return stat


def heaping_score(values: List[Any]) -> Dict[str, Any]:
    cleaned = []
    for v in values:
        try:
            cleaned.append(int(round(float(v))))
        except Exception:
            pass
    n = len(cleaned)
    if n == 0:
        return {"n": 0}
    end05 = sum(1 for x in cleaned if x % 10 in (0, 5)) / n
    end00 = sum(1 for x in cleaned if x % 100 == 0) / n
    return {"n": n, "share_end_0_or_5": end05, "share_end_00": end00}


def robust_zscores(series: pd.Series) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return 0.6745 * (s - med) / mad


def gentle_sleep(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))


def parse_float(s: str) -> Optional[float]:
    try:
        s = s.replace(",", "").strip()
        return float(s)
    except Exception:
        return None


def parse_int(s: str) -> Optional[int]:
    try:
        s = s.replace(",", "").strip()
        return int(float(s))
    except Exception:
        return None


# -----------------------------
# Mode normalize / classify
# -----------------------------
def normalize_mode_text(raw_mode: str) -> str:
    """
    Fix concatenation like '1d agoBR Duos':
    - Insert space after 'ago' if stuck to mode
    - Remove 'Xd ago' prefix
    """
    s = (raw_mode or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(ago)(?=[A-Za-z])", r"\1 ", s, flags=re.IGNORECASE)
    s = re.sub(
        r"^\s*\d+\s*(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days|w|wk|wks|mo|mos|month|months|y|yr|yrs)\s+ago\s+",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def is_aggregate_overview_mode(mode: str) -> bool:
    s = (mode or "").strip()
    if re.search(r"^\s*\d+\s*matches\b", s, flags=re.IGNORECASE):
        return True
    if re.search(r"\bmatches\b", s, flags=re.IGNORECASE):
        return True
    return False


def classify_mode_group(mode: Optional[str]) -> str:
    """
    Return:
      - battle_royale
      - non_battle_royale
      - unknown
    """
    if not mode:
        return "unknown"

    s = normalize_mode_text(mode).lower()

    if (
        "battle royale" in s
        or "battle-royale" in s
        or re.search(r"\bbr\s*(duos|solos|squads|quads|trios)?\b", s)
        or re.search(r"\bbr(duos|solos|squads|quads|trios)\b", s)
    ):
        return "battle_royale"

    non_br_keywords = [
        "breakthrough",
        "conquest",
        "rush",
        "tdm",
        "team deathmatch",
        "frontlines",
        "domination",
        "control",
    ]
    if any(k in s for k in non_br_keywords):
        return "non_battle_royale"

    return "unknown"


# -----------------------------
# Player ID extraction (NEW)
# -----------------------------
def _clean_player_id(s: str) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()

    # Remove very generic junk
    bad = {
        "matches", "match history", "profile", "overview",
        "battlefield", "battlefield 6", "bf6", "tracker", "tracker.gg",
        "battlefield 6 tracker",
    }
    if s.lower() in bad or len(s) < 2:
        return None

    # Strip common title suffixes
    s = re.split(r"\s-\sBattlefield 6 Tracker\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.split(r"\s-\sTracker Network\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.split(r"\s-\sTracker\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # If still generic, drop
    if s.lower() in bad or len(s) < 2:
        return None

    return s


def extract_player_id(page) -> str:
    """
    Best-effort extraction of the player's displayed ID.
    Priority order:
      1) document.title parsing: "<PLAYER>'s Battlefield 6 ..."
      2) og:title meta parsing with same rule
      3) header/name selectors fallback
    """

    def from_title_like(text: str) -> Optional[str]:
        if not text:
            return None
        t = re.sub(r"\s+", " ", text).strip()

        # Primary rule: everything before "'s" or "’s"
        # Example: "XX's Battlefield 6 Session History - Battlefield 6 Tracker"
        m = re.match(r"^\s*(.+?)\s*(?:'s|’s)\s+Battlefield\b", t, flags=re.IGNORECASE)
        if m:
            cand = _clean_player_id(m.group(1))
            return cand

        # Optional: some pages may use "DoomPL - Battlefield 6 ..."
        m2 = re.match(r"^\s*(.+?)\s*-\s*Battlefield\b", t, flags=re.IGNORECASE)
        if m2:
            cand = _clean_player_id(m2.group(1))
            return cand

        return None

    # 1) document.title first (your requirement)
    try:
        title = page.evaluate("() => document.title || ''")
        pid = from_title_like(title)
        if pid:
            return pid
    except Exception:
        pass

    # 2) og:title second
    try:
        og = page.evaluate("""
        () => {
          const el = document.querySelector('meta[property="og:title"]');
          return el ? (el.getAttribute('content') || '') : '';
        }
        """)
        pid = from_title_like(og)
        if pid:
            return pid
    except Exception:
        pass

    # 3) fallback: header/name selectors
    try:
        candidates = page.evaluate(r"""
        () => {
          const out = [];
          const sels = [
            'h1',
            '.trn-profile-header__name',
            '.trn-profile-header h1',
            '.profile-header h1',
            '.profile-header__name',
            '[class*="profile"] h1',
          ];
          for (const sel of sels) {
            const el = document.querySelector(sel);
            if (el && el.innerText) out.push(el.innerText.trim());
          }
          return out;
        }
        """)
        cleaned = []
        for c in (candidates or []):
            cc = _clean_player_id(str(c))
            if cc:
                cleaned.append(cc)
        if cleaned:
            # keep shortest reasonable
            cleaned = sorted(set(cleaned), key=lambda x: (len(x), x.lower()))
            return cleaned[0]
    except Exception:
        pass

    return "unknown"


# -----------------------------
# DOM match extraction
# -----------------------------
def extract_matches_from_dom(page, player_id: str) -> List[Dict[str, Any]]:
    """
    Extract each match card's first-line stats:
    mode + kpm/kd/kills/deaths/assists/spm/game_score
    Adds player_id to each row.
    """
    js = r"""
    () => {
      const candidates = Array.from(document.querySelectorAll('a, div'))
        .filter(el => {
          const t = (el.innerText || '');
          return t.includes('K/D') && t.includes('Kills') && t.includes('Deaths');
        });

      const out = [];
      const seen = new Set();
      for (const el of candidates) {
        const text = (el.innerText || '').trim();
        if (!text || text.length < 60) continue;
        const key = text.slice(0, 2000);
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(text);
      }
      return out;
    }
    """
    texts: List[str] = page.evaluate(js)

    rows: List[Dict[str, Any]] = []
    for t in texts:
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        if not lines:
            continue

        raw_mode = None
        for ln in lines[:6]:
            if "//" in ln:
                raw_mode = ln.split("//", 1)[1].strip()
                break

        if not raw_mode:
            mode_keywords = [
                "br",
                "duos",
                "solos",
                "squads",
                "quads",
                "trios",
                "breakthrough",
                "conquest",
                "rush",
                "tdm",
                "battle royale",
            ]
            for ln in lines[:10]:
                lnl = ln.lower()
                if any(k in lnl for k in mode_keywords):
                    raw_mode = ln.strip()
                    break

        if not raw_mode:
            continue

        mode = normalize_mode_text(raw_mode)
        if is_aggregate_overview_mode(mode):
            continue

        def _norm_label(s: str) -> str:
            return re.sub(r"[\s:]+", "", (s or "").strip().lower())

        def find_value_exact(label: str) -> Optional[str]:
            target = _norm_label(label)
            for i, ln in enumerate(lines):
                if _norm_label(ln) == target:
                    if i + 1 < len(lines):
                        m = re.search(r"([-+]?\d[\d,]*\.?\d*)", lines[i + 1])
                        if m:
                            return m.group(1)
                    m2 = re.search(r"([-+]?\d[\d,]*\.?\d*)", ln)
                    if m2:
                        return m2.group(1)
                    return None
            return None

        def find_value_exact_any(labels: List[str]) -> Optional[str]:
            for lab in labels:
                v = find_value_exact(lab)
                if v is not None:
                    return v
            return None

        kpm_s = find_value_exact_any(["Kills/Min", "Kills per Min", "KPM", "K/M"])
        kd_s = find_value_exact_any(["K/D", "KD"])
        kills_s = find_value_exact("Kills")
        deaths_s = find_value_exact("Deaths")
        assists_s = find_value_exact("Assists")
        spm_s = find_value_exact_any(["Score/Min", "Score per Min", "SPM"])
        gscore_s = find_value_exact("Game Score")

        result = None
        map_name = None
        for ln in lines[:8]:
            if ln.lower() in ("victory", "defeat", "win", "loss"):
                result = "Victory" if ln.lower() in ("victory", "win") else "Defeat"
        if result:
            for i, ln in enumerate(lines[:12]):
                if ln.lower() in ("victory", "defeat", "win", "loss"):
                    if i + 1 < len(lines[:12]):
                        cand = lines[i + 1].strip()
                        if all(x not in cand.lower() for x in ("kills", "score", "k/d", "deaths", "assists")):
                            map_name = cand
                    break

        row = {
            "player_id": player_id,
            "mode": mode,
            "mode_group": classify_mode_group(mode),
            "result": result,
            "map": map_name,
            "kpm": parse_float(kpm_s) if kpm_s else None,
            "kd_ratio": parse_float(kd_s) if kd_s else None,
            "kills": parse_int(kills_s) if kills_s else None,
            "deaths": parse_int(deaths_s) if deaths_s else None,
            "assists": parse_int(assists_s) if assists_s else None,
            "score_per_min": parse_float(spm_s) if spm_s else None,
            "game_score": parse_int(gscore_s) if gscore_s else None,
            "_raw_preview": "\n".join(lines[:12]),
        }
        rows.append(row)

    # dedupe
    seen = set()
    uniq = []
    for r in rows:
        key = (
            r.get("player_id"),
            r.get("mode"),
            r.get("kills"),
            r.get("deaths"),
            r.get("assists"),
            r.get("kpm"),
            r.get("kd_ratio"),
            r.get("score_per_min"),
            r.get("game_score"),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def scrape_with_load_more(
    url: str,
    max_clicks: int,
    outdir: str,
    headless: bool,
    click_selector: str,
    cookie_button_selector: Optional[str],
    slow: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(outdir, exist_ok=True)

    meta: Dict[str, Any] = {
        "url": url,
        "player_id": "unknown",
        "max_clicks": max_clicks,
        "click_selector": click_selector,
        "cookie_button_selector": cookie_button_selector,
        "clicks_done": 0,
        "rows_seen": 0,
        "notes": [],
        "mode_groups": {"battle_royale": 0, "non_battle_royale": 0, "unknown": 0},
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        gentle_sleep(1.0, 2.0)

        if cookie_button_selector:
            try:
                if page.locator(cookie_button_selector).first.is_visible(timeout=3000):
                    page.locator(cookie_button_selector).first.click(timeout=3000)
                    gentle_sleep(0.8, 1.5)
            except Exception:
                pass

        gentle_sleep(1.5, 2.5)

        # NEW: extract player_id as early as possible
        try:
            meta["player_id"] = extract_player_id(page)
        except Exception:
            meta["player_id"] = "unknown"

        for i in range(max_clicks):
            try:
                btn = page.locator(click_selector).first
                if not btn.is_visible(timeout=2500):
                    meta["notes"].append("Load more button is not visible anymore (likely end of list).")
                    break

                btn.click(timeout=5000)
                meta["clicks_done"] += 1
                gentle_sleep(1.2, 2.3) if slow else gentle_sleep(0.8, 1.6)
                if (i + 1) % 8 == 0:
                    gentle_sleep(2.0, 4.0)
            except PWTimeout:
                meta["notes"].append("Timeout when clicking Load more (blocked or end of list).")
                break
            except Exception as e:
                meta["notes"].append(f"Click error: {e}")
                break

        gentle_sleep(1.0, 1.8)
        matches = extract_matches_from_dom(page, meta["player_id"])

        # debug artifacts
        try:
            page.screenshot(path=os.path.join(outdir, "final_page.png"), full_page=True)
        except Exception:
            pass
        try:
            html = page.content()
            with open(os.path.join(outdir, "final_page.html"), "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass

        browser.close()

    df = pd.DataFrame(matches)
    meta["rows_seen"] = int(len(df))

    if not df.empty and "mode_group" in df.columns:
        meta["mode_groups"]["battle_royale"] = int((df["mode_group"] == "battle_royale").sum())
        meta["mode_groups"]["non_battle_royale"] = int((df["mode_group"] == "non_battle_royale").sum())
        meta["mode_groups"]["unknown"] = int((df["mode_group"] == "unknown").sum())

    return df, matches, meta


# -----------------------------
# Analysis
# -----------------------------
def _to_num(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _summary_stats(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna().astype(float)
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _coef_var(series: pd.Series) -> Optional[float]:
    s = series.dropna().astype(float)
    if len(s) < 2:
        return None
    m = float(s.mean())
    if m == 0:
        return None
    return float(s.std(ddof=0) / abs(m))


def _digit_span_ok(series: pd.Series) -> bool:
    s = series.dropna().astype(float)
    s = s[s != 0]
    if s.empty:
        return False
    mn = float(s.min())
    mx = float(s.max())
    if mn <= 0:
        return False
    return (mx / mn) >= 100.0


def _mad_to_severity(mad: Optional[float], lo: float = 0.006, hi: float = 0.020) -> float:
    if mad is None:
        return 0.0
    if mad <= lo:
        return 0.0
    if mad >= hi:
        return 1.0
    return (mad - lo) / (hi - lo)


def verdict_from_risk(score: float) -> str:
    if score < 30:
        return "LOW (no strong statistical anomaly)"
    if score < 60:
        return "MEDIUM (needs manual review: clips / accuracy / match mix)"
    return "HIGH (strong anomaly signals; recommend review)"


def analyze_group(df: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "records": int(len(df)),
        "sanity": {},
        "summary_stats": {},
        "benford": {},
        "heaping": {},
        "outliers": {},
        "fps_signals": {},
        "risk": {},
        "risk_breakdown": {},
        "notes": [],
    }
    if df.empty:
        rep["notes"].append("No rows collected for this group.")
        return rep

    count_cols = ["kills", "deaths", "assists", "game_score"]
    rate_cols = ["kpm", "kd_ratio", "score_per_min"]
    all_cols = count_cols + rate_cols
    _to_num(df, all_cols)

    def zero_share(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        s = df[col].dropna()
        if s.empty:
            return None
        return float((s == 0).mean())

    zs_k = zero_share("kills")
    zs_d = zero_share("deaths")
    rep["sanity"] = {
        "zero_share_kills": zs_k,
        "zero_share_deaths": zs_d,
        "zero_pollution_flag": bool(
            (zs_k is not None and zs_k >= 0.55) and (zs_d is not None and zs_d >= 0.55)
        ),
    }
    if rep["sanity"]["zero_pollution_flag"]:
        rep["notes"].append(
            "Kills+Deaths have a very high zero-share (>=55%). This usually indicates DOM label/value misalignment "
            "or collapsed rows, not real player data. Fix scraping first."
        )

    for c in all_cols:
        if c in df.columns:
            rep["summary_stats"][c] = _summary_stats(df[c])

    for c in count_cols:
        if c in df.columns:
            rep["heaping"][c] = heaping_score(df[c].dropna().tolist())

    for c in all_cols:
        if c not in df.columns:
            continue
        z = robust_zscores(df[c])
        if z.empty:
            continue
        absz = z.abs()
        rep["outliers"][c] = {
            "share_abs_z_ge_5": float((absz >= 5).mean()),
            "share_abs_z_ge_8": float((absz >= 8).mean()),
            "top10": [
                {"row": int(i), "value": float(df.loc[i, c]), "z": float(z.loc[i])}
                for i in absz.sort_values(ascending=False).head(10).index
            ],
        }

    benford_used_cols = []
    for c in count_cols:
        if c not in df.columns:
            continue
        vals = df[c].dropna().tolist()
        counts, n = benford_counts(vals)
        mad = benford_mad(counts, n)
        chi = chi_square_stat(counts, n)
        span_ok = _digit_span_ok(df[c]) and (n >= 200)
        rep["benford"][c] = {
            "n_used": n,
            "mad": mad,
            "chi_square": chi,
            "counts": counts,
            "risk_counted": bool(span_ok),
            "note": "Benford is applied to count-like fields only and counted into risk only when digit-span and sample size are sufficient.",
        }
        if span_ok and mad is not None:
            benford_used_cols.append(c)

    def share_ge(col: str, thr: float) -> Optional[float]:
        if col not in df.columns:
            return None
        s = df[col].dropna().astype(float)
        if s.empty:
            return None
        return float((s >= thr).mean())

    fps = {
        "share_kd_ge_10": share_ge("kd_ratio", 10.0),
        "share_kd_ge_20": share_ge("kd_ratio", 20.0),
        "share_kpm_ge_3": share_ge("kpm", 3.0),
        "share_kpm_ge_4": share_ge("kpm", 4.0),
        "share_kills_ge_60": share_ge("kills", 60.0),
        "share_kills_ge_80": share_ge("kills", 80.0),
        "share_deaths_le_1": None,
    }
    if "deaths" in df.columns:
        s = df["deaths"].dropna().astype(float)
        if not s.empty:
            fps["share_deaths_le_1"] = float((s <= 1).mean())

    fps["coef_var_kills"] = _coef_var(df["kills"]) if "kills" in df.columns else None
    fps["coef_var_kd_ratio"] = _coef_var(df["kd_ratio"]) if "kd_ratio" in df.columns else None
    fps["coef_var_kpm"] = _coef_var(df["kpm"]) if "kpm" in df.columns else None
    rep["fps_signals"] = fps

    def sev_share(x: Optional[float], lo: float, hi: float) -> float:
        if x is None:
            return 0.0
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return (x - lo) / (hi - lo)

    b_scores = []
    for c in benford_used_cols:
        mad = rep["benford"][c]["mad"]
        b_scores.append(_mad_to_severity(mad))
    benford_component = sum(b_scores) / len(b_scores) if b_scores else 0.0

    h_scores = []
    for c in count_cols:
        obj = rep["heaping"].get(c)
        if not obj or obj.get("n", 0) < 200:
            continue
        share05 = float(obj.get("share_end_0_or_5", 0.0))
        share00 = float(obj.get("share_end_00", 0.0))
        s1 = 0.0 if share05 <= 0.22 else (1.0 if share05 >= 0.40 else (share05 - 0.22) / 0.18)
        s2 = 0.0 if share00 <= 0.04 else (1.0 if share00 >= 0.12 else (share00 - 0.04) / 0.08)
        h_scores.append(min(1.0, (s1 + s2) / 2.0))
    heaping_component = sum(h_scores) / len(h_scores) if h_scores else 0.0

    o_scores = []
    for _, obj in rep["outliers"].items():
        share8 = float(obj.get("share_abs_z_ge_8", 0.0))
        share5 = float(obj.get("share_abs_z_ge_5", 0.0))
        s8 = 0.0 if share8 <= 0.00 else (1.0 if share8 >= 0.01 else share8 / 0.01)
        s5 = 0.0 if share5 <= 0.03 else (1.0 if share5 >= 0.10 else (share5 - 0.03) / 0.07)
        o_scores.append(min(1.0, (s8 + s5) / 2.0))
    outlier_component = sum(o_scores) / len(o_scores) if o_scores else 0.0

    fps_scores = []
    fps_scores.append(sev_share(fps.get("share_kd_ge_10"), 0.02, 0.12))
    fps_scores.append(sev_share(fps.get("share_kd_ge_20"), 0.00, 0.05))
    fps_scores.append(sev_share(fps.get("share_kpm_ge_3"), 0.03, 0.15))
    fps_scores.append(sev_share(fps.get("share_kpm_ge_4"), 0.00, 0.08))
    fps_scores.append(sev_share(fps.get("share_kills_ge_60"), 0.00, 0.06))
    fps_scores.append(sev_share(fps.get("share_kills_ge_80"), 0.00, 0.03))

    consistency = 0.0
    if fps.get("share_kd_ge_10") is not None and fps["share_kd_ge_10"] >= 0.06:
        cv_kd = fps.get("coef_var_kd_ratio")
        if cv_kd is not None:
            if cv_kd <= 0.25:
                consistency = 1.0
            elif cv_kd >= 0.60:
                consistency = 0.0
            else:
                consistency = (0.60 - cv_kd) / (0.60 - 0.25)
    fps_scores.append(consistency)
    fps_component = sum(fps_scores) / len(fps_scores) if fps_scores else 0.0

    if rep["sanity"]["zero_pollution_flag"]:
        rep["notes"].append("Data quality warning: scores may be unreliable because scraped stats look misaligned.")
        shrink = 0.35
        benford_component *= shrink
        heaping_component *= shrink
        outlier_component *= shrink
        fps_component *= shrink

    risk = 100.0 * (
        0.50 * fps_component
        + 0.25 * outlier_component
        + 0.15 * heaping_component
        + 0.10 * benford_component
    )

    rep["risk_breakdown"] = {
        "fps_component": round(fps_component, 4),
        "outlier_component": round(outlier_component, 4),
        "heaping_component": round(heaping_component, 4),
        "benford_component": round(benford_component, 4),
        "benford_counted_fields": benford_used_cols,
    }

    rep["risk"] = {
        "score_0_100": round(risk, 2),
        "interpretation": "0-30 low, 30-60 medium, 60-100 high anomaly (not proof).",
        "verdict_hint": verdict_from_risk(risk),
    }

    if len(df) < 200:
        rep["notes"].append("Sample size < 200: many statistics are unstable; collect more matches for better confidence.")

    return rep


# -----------------------------
# FPS scoring + probability + accuracy
# -----------------------------
def _clamp01(x: float) -> float:
    return 0.0 if x <= 0 else (1.0 if x >= 1 else x)


def _ramp(x: float, lo: float, hi: float) -> float:
    if x is None:
        return 0.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def fps_signals_score(signals: dict, mode_group: str) -> dict:
    def g(k, default=0.0):
        v = signals.get(k, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    is_br = (mode_group == "battle_royale")

    if is_br:
        sev_share_kd_ge_10 = _ramp(g("share_kd_ge_10"), 0.10, 0.25)
        sev_share_kd_ge_20 = _ramp(g("share_kd_ge_20"), 0.005, 0.015)
        sev_share_kpm_ge_3 = _ramp(g("share_kpm_ge_3"), 0.01, 0.05)
        sev_share_kpm_ge_4 = _ramp(g("share_kpm_ge_4"), 0.002, 0.02)
        sev_share_kills_ge_60 = _ramp(g("share_kills_ge_60"), 0.01, 0.05)
        sev_share_kills_ge_80 = _ramp(g("share_kills_ge_80"), 0.005, 0.02)
        sev_cv_kd = _ramp(0.35 - g("coef_var_kd_ratio"), 0.00, 0.20)
        sev_cv_kpm = _ramp(0.30 - g("coef_var_kpm"), 0.00, 0.20)
    else:
        sev_share_kd_ge_10 = _ramp(g("share_kd_ge_10"), 0.08, 0.20)
        sev_share_kd_ge_20 = _ramp(g("share_kd_ge_20"), 0.003, 0.010)
        sev_share_kpm_ge_3 = _ramp(g("share_kpm_ge_3"), 0.03, 0.12)
        sev_share_kpm_ge_4 = _ramp(g("share_kpm_ge_4"), 0.01, 0.06)
        sev_share_kills_ge_60 = _ramp(g("share_kills_ge_60"), 0.02, 0.08)
        sev_share_kills_ge_80 = _ramp(g("share_kills_ge_80"), 0.01, 0.05)
        sev_cv_kd = _ramp(0.30 - g("coef_var_kd_ratio"), 0.00, 0.18)
        sev_cv_kpm = _ramp(0.25 - g("coef_var_kpm"), 0.00, 0.18)

    components = {
        "KD≥10 share": _clamp01(sev_share_kd_ge_10),
        "KD≥20 share": _clamp01(sev_share_kd_ge_20),
        "KPM≥3 share": _clamp01(sev_share_kpm_ge_3),
        "KPM≥4 share": _clamp01(sev_share_kpm_ge_4),
        "Kills≥60 share": _clamp01(sev_share_kills_ge_60),
        "Kills≥80 share": _clamp01(sev_share_kills_ge_80),
        "Low var KD": _clamp01(sev_cv_kd),
        "Low var KPM": _clamp01(sev_cv_kpm),
    }

    weights = {
        "KD≥10 share": 0.22,
        "KD≥20 share": 0.18,
        "KPM≥3 share": 0.12,
        "KPM≥4 share": 0.08,
        "Kills≥60 share": 0.12,
        "Kills≥80 share": 0.10,
        "Low var KD": 0.10,
        "Low var KPM": 0.08,
    }

    score01 = 0.0
    for k, w in weights.items():
        score01 += components.get(k, 0.0) * w
    fps_score = round(score01 * 100.0, 2)

    if fps_score < 30:
        level = "LOW"
    elif fps_score < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    triggers = []
    for k, sev in sorted(components.items(), key=lambda x: x[1], reverse=True):
        if sev >= 0.5:
            triggers.append({"indicator": k, "severity_0_1": round(sev, 3)})

    return {
        "fps_score_0_100": fps_score,
        "level": level,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": weights,
        "triggers": triggers,
    }


def estimate_cheat_probability(
    stat_score: float,
    fps_score: float,
    records: int,
    triggers_count: int,
    zero_pollution: bool,
) -> dict:
    s_stat = float(stat_score) / 100.0
    s_fps = float(fps_score) / 100.0

    base = 0.65 * s_fps + 0.35 * s_stat
    trigger_bonus = min(0.15, triggers_count * 0.025)

    if records >= 800:
        sample_bonus = 0.10
    elif records >= 400:
        sample_bonus = 0.06
    elif records >= 200:
        sample_bonus = 0.03
    else:
        sample_bonus = 0.0

    pollution_penalty = 0.30 if zero_pollution else 0.0

    prob = base + trigger_bonus + sample_bonus - pollution_penalty
    prob = max(0.0, min(1.0, prob))
    prob_pct = round(prob * 100.0, 1)

    if prob_pct >= 80:
        conf = "VERY HIGH"
    elif prob_pct >= 65:
        conf = "HIGH"
    elif prob_pct >= 45:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    return {"probability_0_100": prob_pct, "confidence_level": conf, "probability_raw": prob}


def estimate_model_accuracy(
    records: int,
    fps_score: float,
    stat_score: float,
    triggers_count: int,
    zero_pollution: bool,
) -> dict:
    acc = 0.55

    if records >= 800:
        acc += 0.25
    elif records >= 400:
        acc += 0.18
    elif records >= 200:
        acc += 0.12
    elif records >= 100:
        acc += 0.06

    acc += min(0.12, (float(fps_score) / 100.0) * 0.12)
    acc += min(0.08, (float(stat_score) / 100.0) * 0.08)
    acc += min(0.08, triggers_count * 0.015)

    if zero_pollution:
        acc -= 0.35

    acc = max(0.0, min(0.98, acc))
    acc_pct = round(acc * 100.0, 1)

    if acc_pct >= 85:
        level = "VERY HIGH"
    elif acc_pct >= 75:
        level = "HIGH"
    elif acc_pct >= 60:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"accuracy_0_100": acc_pct, "confidence_level": level, "accuracy_raw": acc}


# -----------------------------
# Terminal formatting
# -----------------------------
def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{x * 100:.{digits}f}%"
    except Exception:
        return "n/a"


def _fmt_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def _bar(score: float, width: int = 22) -> str:
    score = max(0.0, min(100.0, float(score)))
    filled = int(round(width * score / 100.0))
    return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"


def _section(title: str) -> str:
    return f"{title}\n" + ("─" * len(title))


def print_overall_header(meta: dict, counts: dict, paths: dict) -> None:
    print("\n" + "=" * 72)
    print("BF6 ANTI-CHEAT FORENSIC SUMMARY (screening signals, NOT proof)")
    print("=" * 72)
    print(f"Player ID: {meta.get('player_id', 'unknown')}")
    print(f"URL: {meta.get('url')}")
    print(f"LoadMore clicks: {meta.get('clicks_done')} / max={meta.get('max_clicks')}")
    print(f"DOM rows extracted: {meta.get('rows_seen')}")
    print(
        "Group counts: "
        f"BR={counts.get('battle_royale', 0)}  "
        f"MP={counts.get('non_battle_royale', 0)}  "
        f"UNK={counts.get('unknown', 0)}  "
        f"TOTAL={counts.get('total', 0)}"
    )
    if meta.get("notes"):
        print("Scrape notes:", "; ".join(meta["notes"][:2]))

    print("\n" + _section("Artifacts"))
    print(f"CSV    : {paths.get('csv')}")
    print(f"REPORT : {paths.get('report')}")
    print(f"META   : {paths.get('meta')}")
    print("=" * 72 + "\n")


def compute_group_estimates(group_key: str, report: dict) -> Tuple[dict, dict, dict]:
    stat_score = float(report.get("risk", {}).get("score_0_100", 0.0))
    fps = report.get("fps_signals", {}) or {}
    fps_eval = fps_signals_score(fps, group_key)

    zero_flag = bool(report.get("sanity", {}).get("zero_pollution_flag"))
    records = int(report.get("records", 0))
    triggers_count = len(fps_eval.get("triggers", []))

    cheat_est = estimate_cheat_probability(
        stat_score=stat_score,
        fps_score=float(fps_eval.get("fps_score_0_100", 0.0)),
        records=records,
        triggers_count=triggers_count,
        zero_pollution=zero_flag,
    )

    acc_est = estimate_model_accuracy(
        records=records,
        fps_score=float(fps_eval.get("fps_score_0_100", 0.0)),
        stat_score=stat_score,
        triggers_count=triggers_count,
        zero_pollution=zero_flag,
    )

    return fps_eval, cheat_est, acc_est


def print_group_forensic_block(group_key: str, analysis: dict) -> None:
    group_title = "BATTLE ROYALE" if group_key == "battle_royale" else "MULTIPLAYER (NON-BR)"
    records = int(analysis.get("records", 0))

    stat_risk = float(analysis.get("risk", {}).get("score_0_100", 0.0))
    stat_level = analysis.get("risk", {}).get("verdict_hint") or verdict_from_risk(stat_risk)

    fps_eval, cheat_est, acc_est = compute_group_estimates(group_key, analysis)
    fps_score = float(fps_eval.get("fps_score_0_100", 0.0))
    fps_level = fps_eval.get("level", "LOW")

    fps = analysis.get("fps_signals", {}) or {}
    sanity = analysis.get("sanity", {}) or {}
    zero_flag = bool(sanity.get("zero_pollution_flag"))
    rb = analysis.get("risk_breakdown", {}) or {}

    print(_section(f"{group_title} — FORENSIC BLOCK"))
    print(f"Matches analyzed: {records}")
    print("")
    print("Scores")
    print(f"  Statistical anomaly score : {stat_risk:6.2f} / 100  {_bar(stat_risk)}  Level: {stat_level}")
    print(f"  FPS cheat-risk score      : {fps_score:6.2f} / 100  {_bar(fps_score)}  Level: {fps_level}")
    print("")
    print("Probability estimate")
    print(f"  Estimated cheat probability : {cheat_est['probability_0_100']:6.1f}%   Confidence: {cheat_est['confidence_level']}")
    print(f"  Estimated model accuracy    : {acc_est['accuracy_0_100']:6.1f}%   Confidence: {acc_est['confidence_level']}")
    print("")

    print("Top indicators (match-level shares / consistency)")
    print(f"  KD ≥ 10 share     : {_fmt_pct(fps.get('share_kd_ge_10'))}")
    print(f"  KD ≥ 20 share     : {_fmt_pct(fps.get('share_kd_ge_20'))}")
    print(f"  KPM ≥ 3 share     : {_fmt_pct(fps.get('share_kpm_ge_3'))}")
    print(f"  KPM ≥ 4 share     : {_fmt_pct(fps.get('share_kpm_ge_4'))}")
    print(f"  Kills ≥ 60 share  : {_fmt_pct(fps.get('share_kills_ge_60'))}")
    print(f"  Kills ≥ 80 share  : {_fmt_pct(fps.get('share_kills_ge_80'))}")
    print(f"  CV(KD ratio)      : {_fmt_float(fps.get('coef_var_kd_ratio'))}  (lower = more consistent)")
    print(f"  CV(KPM)           : {_fmt_float(fps.get('coef_var_kpm'))}  (lower = more consistent)")
    print("")

    if fps_eval.get("triggers"):
        print("Triggered rules (severity ≥ 0.5)")
        for t in fps_eval["triggers"][:6]:
            print(f"  - {t['indicator']:<14s}  severity={t['severity_0_1']}")
        print("")

    print("Risk breakdown (internal components)")
    print(
        "  fps_component     : "
        f"{rb.get('fps_component')}  |  "
        f"outliers_component: {rb.get('outlier_component')}  |  "
        f"heaping_component : {rb.get('heaping_component')}  |  "
        f"benford_component : {rb.get('benford_component')}"
    )
    print("")

    if zero_flag:
        print("Data sanity")
        print("  WARNING: Kills+Deaths have a high zero-share (>=55%).")
        print("  This usually indicates DOM label/value misalignment or collapsed rows.")
        print("  Recommendation: fix scraping first; scores may be unreliable.")
        print("")

    used = rb.get("benford_counted_fields") or []
    if used:
        b = analysis.get("benford", {}) or {}
        print("Benford MAD (count fields only; counted only when digit-span + n sufficient)")
        for f in used:
            obj = b.get(f, {}) or {}
            mad = obj.get("mad")
            n_used = obj.get("n_used")
            if mad is None or not n_used:
                continue
            print(f"  - {f:<10s} MAD={mad:.6f}  n={n_used}")
        print("")

    notes = analysis.get("notes", []) or []
    if notes:
        print("Notes")
        for n in notes[:6]:
            print(f"  - {n}")
        print("")

    print("Recommendation")
    if cheat_est["probability_0_100"] >= 80 or fps_level == "HIGH" or "HIGH" in str(stat_level):
        print("  → Strong anomaly signals. Recommend manual review: clips, accuracy, match distribution.")
    elif cheat_est["probability_0_100"] >= 55 or fps_level == "MEDIUM" or "MEDIUM" in str(stat_level):
        print("  → Some anomaly signals. Recommend additional sampling and selective manual checks.")
    else:
        print("  → No strong signal. If suspicious, collect more matches and re-run.")
    print("\n" + ("-" * 72) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="BF6 one-click: click load more, DOM-extract mode+stats, analyze battle_royale vs non_battle_royale"
    )
    ap.add_argument("--url", required=True, help="Matches page URL you provide")
    ap.add_argument("--max-clicks", type=int, default=100, help="How many times to click Load more")
    ap.add_argument("--outdir", default="out_loadmore", help="Output directory")
    ap.add_argument("--headless", action="store_true", help="Run without opening a browser window")
    ap.add_argument("--slow", action="store_true", help="Slower pacing")
    ap.add_argument("--click-selector", default="button:has-text('Load More')", help="Playwright selector for load more button")
    ap.add_argument("--cookie-selector", default=None, help="Optional selector for 'Accept cookies' button")
    args = ap.parse_args()

    df, matches_list, meta = scrape_with_load_more(
        url=args.url,
        max_clicks=args.max_clicks,
        outdir=args.outdir,
        headless=args.headless,
        click_selector=args.click_selector,
        cookie_button_selector=args.cookie_selector,
        slow=args.slow,
    )

    os.makedirs(args.outdir, exist_ok=True)

    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if df.empty:
        print("No records extracted (DOM parse got 0).")
        print("Meta:", meta_path)
        print("See debug artifacts:", os.path.join(args.outdir, "final_page.html"), "and final_page.png")
        return

    cols = [
        "player_id",
        "mode",
        "mode_group",
        "result",
        "map",
        "kpm",
        "kd_ratio",
        "kills",
        "deaths",
        "assists",
        "score_per_min",
        "game_score",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols + ["_raw_preview"]]

    csv_path = os.path.join(args.outdir, "matches.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    df_br = df[df["mode_group"] == "battle_royale"].copy()
    df_non = df[df["mode_group"] == "non_battle_royale"].copy()
    df_unk = df[df["mode_group"] == "unknown"].copy()

    report_br = analyze_group(df_br)
    report_non = analyze_group(df_non)

    fps_eval_br, cheat_br, acc_br = compute_group_estimates("battle_royale", report_br) if len(df_br) > 0 else ({}, {}, {})
    fps_eval_non, cheat_non, acc_non = compute_group_estimates("non_battle_royale", report_non) if len(df_non) > 0 else ({}, {}, {})

    report_path = os.path.join(args.outdir, "report.json")
    report_wrap = {
        "player_id": meta.get("player_id", "unknown"),
        "meta": meta,
        "counts": {
            "battle_royale": int(len(df_br)),
            "non_battle_royale": int(len(df_non)),
            "unknown": int(len(df_unk)),
            "total": int(len(df)),
        },
        "matches": matches_list,
        "analysis_by_group": {
            "battle_royale": {
                **report_br,
                "fps_eval": fps_eval_br,
                "probability_estimate": cheat_br,
                "accuracy_estimate": acc_br,
            },
            "non_battle_royale": {
                **report_non,
                "fps_eval": fps_eval_non,
                "probability_estimate": cheat_non,
                "accuracy_estimate": acc_non,
            },
        },
        "notes": [
            "No overall analysis: battle royale and multiplayer are analyzed separately.",
            "Benford is applied to count-like fields only and counted into risk only when digit-span and sample size are sufficient.",
            "High risk is NOT proof of cheating; it means statistically unusual patterns worth manual review.",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_wrap, f, ensure_ascii=False, indent=2)

    counts = report_wrap["counts"]
    paths = {"csv": csv_path, "report": report_path, "meta": meta_path}

    print_overall_header(meta=meta, counts=counts, paths=paths)

    if len(df_br) > 0:
        print_group_forensic_block("battle_royale", report_br)
    else:
        print(_section("BATTLE ROYALE — FORENSIC BLOCK"))
        print("No records extracted for this group.\n" + ("-" * 72) + "\n")

    if len(df_non) > 0:
        print_group_forensic_block("non_battle_royale", report_non)
    else:
        print(_section("MULTIPLAYER (NON-BR) — FORENSIC BLOCK"))
        print("No records extracted for this group.\n" + ("-" * 72) + "\n")

    print("DONE")
    print(f"Player ID: {meta.get('player_id', 'unknown')}")
    print(f"CSV    : {csv_path}")
    print(f"REPORT : {report_path}")
    print(f"META   : {meta_path}")


if __name__ == "__main__":
    main()