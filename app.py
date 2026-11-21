import asyncio, os, time, secrets, json, re
import urllib.request, urllib.error
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
from strategy import signal, haar_denoise, ema, rsi

# Un ou plusieurs mots de passe séparés par des virgules. Chaque mot de passe a son propre espace.
PASSWORDS = [p.strip() for p in os.environ.get("APP_PASSWORD", "").split(",") if p.strip()]
ASSETS = os.environ.get("ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,EURJPY_otc,AUDCAD_otc,EURGBP_otc").split(",")
MAX_ASSETS = int(os.environ.get("MAX_ASSETS", "12"))
app = FastAPI()
STATES = {}

def new_state():
    return {"running": False, "log": [], "stats": {"win": 0, "loss": 0, "draw": 0}, "balance": None, "step": 0, "task": None}

def auth(pw):
    if PASSWORDS and not any(secrets.compare_digest(pw or "", p) for p in PASSWORDS):
        raise HTTPException(401, "Mot de passe incorrect")
    return STATES.setdefault(pw if PASSWORDS else "", new_state())

class Cfg(BaseModel):
    ssid: str; timeframe: int = 60; amount: float = 1; martingales: int = 2; multiplier: float = 2.2
    max_stake: float = 50; max_losses: int = 5; min_score: int = 2; min_payout: int = 80

def log(st, m):
    st["log"].append(f"{time.strftime('%H:%M:%S')}  {m}"); del st["log"][:-80]

def clean_ssid(s):
    """Nettoie le SSID et renomme sessionToken en session (même valeur)."""
    s = (s or "").strip()
    for a, b in (("“", '"'), ("”", '"'), ("„", '"'), ("‘", "'"), ("’", "'")): s = s.replace(a, b)
    s = s.strip("'`\" \n\r\t")
    s = re.sub(r",?\s*(…|\.\.\.)\s*(?=[}\]])", "", s)
    if not re.search(r'"(session|sessionToken|uid)"\s*:', s):
        s = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', s)
    try:
        d = json.loads(s[2:]) if s.startswith("42[") else None
        if isinstance(d, list) and len(d) == 2 and isinstance(d[1], dict):
            o = d[1]
            if "sessionToken" in o and "session" not in o: o["session"] = o.pop("sessionToken")
            if isinstance(o.get("uid"), str) and o["uid"].isdigit(): o["uid"] = int(o["uid"])
            o.setdefault("isDemo", 1); o.setdefault("platform", 1)
            s = "42" + json.dumps([d[0], o], separators=(",", ":"))
    except Exception: pass
    return s

def parse_ssid(s):
    """Vérifie le SSID. Renvoie True (démo), False (réel) ou None (type inconnu)."""
    if "…" in s or "..." in s:
        raise ValueError("Ton SSID contient « … » : c'est un aperçu coupé. Copie le message complet (appui long, Copy).")
    if not s.startswith("42["):
        raise ValueError('SSID invalide : il doit commencer par 42["auth", ... Copie-le en entier.')
    if "auth" not in s[:14]:
        raise ValueError('Ce message n\'est pas un message "auth". Prends celui qui commence par 42["auth".')
    if re.search(r'["\']?isChart["\']?\s*:\s*(1|true)', s):
        raise ValueError("Ce message vient du socket des graphiques (isChart). Prends celui du socket de trading.")
    o = None
    try:
        d = json.loads(s[2:])
        if isinstance(d, list) and len(d) == 2 and isinstance(d[1], dict): o = d[1]
    except Exception: pass
    if o is not None:
        if not (o.get("session") or o.get("sessionToken")) or not o.get("uid"):
            raise ValueError("SSID incomplet : il manque la session ou l'uid.")
        return bool(o["isDemo"]) if "isDemo" in o else None
    ok_sess = re.search(r'["\']?(session|sessionToken)["\']?\s*:\s*["\'][^"\']+', s)
    ok_uid = re.search(r'["\']?uid["\']?\s*:\s*["\']?\d+', s)
    if not (ok_sess and ok_uid):
        raise ValueError("SSID illisible : il manque la session ou l'uid. Copie le message brut en entier, avec tous ses guillemets.")
    m = re.search(r'["\']?isDemo["\']?\s*:\s*(1|0|true|false)', s)
    return (m.group(1) in ("1", "true")) if m else None

def signal_aggr(candles):
    """QuantumFlux variante agressive : seuils plus bas, plus de trades. Retourne (action, score)."""
    if len(candles) < 40: return None, 0
    o, h, l, c = (haar_denoise([k[f] for k in candles], 3) for f in ("open", "high", "low", "close"))
    f, s = ema(c, 5), ema(c, 13)
    sma, sd = c[-20:].mean(), c[-20:].std()
    r = rsi(c); r0, r1 = r[-2], r[-1]
    def stoch_at(i):
        lo, hi = l[max(0, i - 13):i + 1].min(), h[max(0, i - 13):i + 1].max()
        return (c[i] - lo) / max(hi - lo, 1e-12) * 100
    ks = [stoch_at(i) for i in range(len(c) - 5, len(c))]
    st0, st1 = np.mean(ks[-4:-1]), np.mean(ks[-3:])
    up, dn = f[-1] > s[-1], f[-1] < s[-1]
    cl = c[-1]; bull, bear = cl > o[-1], cl < o[-1]
    xlo, xhi = cl < sma - sd * 1.5, cl > sma + sd * 1.5
    olo, ohi = cl < sma - sd * 0.8, cl > sma + sd * 0.8
    co = lambda a, b, lv: a <= lv < b; cu = lambda a, b, lv: a >= lv > b
    rules = [
        (xlo and r1 < 35 and st1 < 25 and bull, "CALL", 3), (xhi and r1 > 65 and st1 > 75 and bear, "PUT", 3),
        (up and co(r0, r1, 30) and bull, "CALL", 3), (dn and cu(r0, r1, 70) and bear, "PUT", 3),
        (olo and r1 < 35 and bull, "CALL", 2), (ohi and r1 > 65 and bear, "PUT", 2),
        (olo and up, "CALL", 2), (ohi and dn, "PUT", 2),
        (up and co(st0, st1, 20), "CALL", 2), (dn and cu(st0, st1, 80), "PUT", 2),
    ]
    for cond, act, sc in rules:
        if cond: return act, sc
    return None, 0

def pick_assets(payouts, cfg):
    """Prend les paires de devises les mieux payées (jusqu'à MAX_ASSETS), sinon la liste par défaut."""
    if isinstance(payouts, dict) and payouts:
        c = [(a, p) for a, p in payouts.items()
             if isinstance(p, (int, float)) and p >= cfg.min_payout and re.fullmatch(r"[A-Z]{6}(_otc)?", str(a))]
        c.sort(key=lambda x: -x[1])
        if c: return [a for a, _ in c[:MAX_ASSETS]]
    return ASSETS

async def scan_one(api, a, cfg, sem, fn):
    async with sem:
        try: candles = await asyncio.wait_for(api.get_candles(a, cfg.timeframe, cfg.timeframe * 80), 15)
        except Exception: return a, None, 0, False
    act, sc = fn(candles)
    return a, act, sc, True

async def best_signal(api, cfg):
    """Analyse toutes les paires en parallèle et garde le signal le plus fort."""
    pf = getattr(api, "payout", None) or getattr(api, "payouts", None)
    try: payouts = await asyncio.wait_for(pf(), 15) if pf else {}
    except Exception: payouts = {}
    fn = signal_aggr if cfg.min_score <= 2 else signal
    sem = asyncio.Semaphore(8)
    res = await asyncio.gather(*[scan_one(api, a, cfg, sem, fn) for a in pick_assets(payouts, cfg)])
    best, n = None, 0
    for a, act, sc, ok in res:
        if not ok: continue
        n += 1
        if act and sc >= cfg.min_score and (best is None or sc > best[2]): best = (a, act, sc)
    return best, n

def trade_result(res):
    """Renvoie 'win', 'loss' ou 'draw' à partir de la réponse de la librairie."""
    if isinstance(res, dict):
        r = str(res.get("result", "")).lower()
        if "win" in r: return "win"
        if "loss" in r or "lose" in r: return "loss"
        if "draw" in r or "tie" in r: return "draw"
        try:
            p = float(res.get("profit"))
            return "win" if p > 0 else "loss" if p < 0 else "draw"
        except Exception: pass
    s = str(res).lower()
    return "win" if "win" in s else "loss" if ("loss" in s or "lose" in s) else "draw"

async def run(cfg: Cfg, st):
    api = None
    try:
        try: from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
        except ImportError: from BinaryOptionsToolsV2 import PocketOptionAsync
        try:
            from BinaryOptionsToolsV2.tracing import start_logs
            start_logs(path="/tmp/po_logs", level="INFO", terminal=True)
        except Exception: pass
        demo = parse_ssid(cfg.ssid)
        log(st, "Connexion à Pocket Option…")
        api = PocketOptionAsync(cfg.ssid)
        wa = getattr(api, "wait_for_assets", None)
        if wa:
            try: await asyncio.wait_for(wa(15), 25)
            except Exception: pass
        else: await asyncio.sleep(5)
        try: st["balance"] = await asyncio.wait_for(api.balance(), 25)
        except asyncio.TimeoutError:
            raise RuntimeError("Connexion impossible : session morte ou refusée. Reconnecte-toi à Pocket Option et recopie un SSID tout neuf.")
        except Exception as e:
            raise RuntimeError(f"Connexion impossible : {type(e).__name__}: {e}")
        mode = "DÉMO" if demo is True else "RÉEL ⚠️" if demo is False else "type inconnu, vérifie ton solde"
        log(st, f"Connecté, compte {mode}, solde {st['balance']}")
        st["step"] = 0; loss_streak = 0; last_hb = 0; last_bal = time.time()
        while st["running"]:
            if time.time() - last_bal > 20:
                st["balance"] = await asyncio.wait_for(api.balance(), 25); last_bal = time.time()
            sig, n = await best_signal(api, cfg)
            if not sig:
                if time.time() - last_hb > 60: log(st, f"Aucun signal pour l'instant ({n} paires analysées)"); last_hb = time.time()
                await asyncio.sleep(1); continue
            asset, act, sc = sig
            stake = round(min(cfg.amount * cfg.multiplier ** st["step"], cfg.max_stake), 2)
            log(st, f"{act} {asset} {stake:.2f} (force {sc}, martingale {st['step']}/{cfg.martingales})")
            fn = api.buy if act == "CALL" else api.sell
            r0 = await asyncio.wait_for(fn(asset=asset, amount=stake, time=cfg.timeframe), 30)
            trade_id = r0[0] if isinstance(r0, (tuple, list)) else r0
            res = await asyncio.wait_for(api.check_win(trade_id), cfg.timeframe + 60)
            out = trade_result(res)
            if out == "win": st["stats"]["win"] += 1; st["step"] = 0; loss_streak = 0; log(st, "✅ Gagné")
            elif out == "loss":
                st["stats"]["loss"] += 1; loss_streak += 1; log(st, "❌ Perdu")
                st["step"] = st["step"] + 1 if st["step"] < cfg.martingales else 0
                if loss_streak >= cfg.max_losses: log(st, "Arrêt : trop de pertes d'affilée"); st["running"] = False
            else: st["stats"]["draw"] += 1; log(st, f"↔ Égalité ({str(res)[:80]})")
            last_bal = 0
            await asyncio.sleep(1)
    except asyncio.CancelledError: pass
    except asyncio.TimeoutError: log(st, "Connexion perdue avec Pocket Option (session coupée). Reprends un SSID tout neuf.")
    except Exception as e: log(st, f"Erreur : {e}")
    finally:
        st["running"] = False
        if api is not None:
            for name in ("shutdown", "disconnect", "close"):
                f = getattr(api, name, None)
                if f:
                    try:
                        r = f()
                        if asyncio.iscoroutine(r): await asyncio.wait_for(r, 5)
                        break
                    except Exception: pass
        log(st, "Bot arrêté")

def _probe(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return f"{r.status} {r.read(60).decode('utf-8', 'ignore')!r}"
    except urllib.error.HTTPError as e: return f"HTTP {e.code} (serveur: {e.headers.get('server', '?')})"
    except Exception as e: return f"{type(e).__name__}: {e}"

@app.get("/api/diag")
async def diag():
    urls = ["https://pocketoption.com/en/", "https://demo-api-eu.po.market/socket.io/?EIO=4&transport=polling"]
    res = await asyncio.gather(*[asyncio.to_thread(_probe, u) for u in urls])
    return dict(zip(urls, res))

@app.post("/api/start")
async def start(cfg: Cfg, x_pw: str = Header(None)):
    st = auth(x_pw)
    if st["running"]: raise HTTPException(400, "Déjà en marche")
    cfg.ssid = clean_ssid(cfg.ssid)
    try: parse_ssid(cfg.ssid)
    except ValueError as e: raise HTTPException(400, str(e))
    st.update(running=True, stats={"win": 0, "loss": 0, "draw": 0}, balance=None)
    st["task"] = asyncio.create_task(run(cfg, st)); return {"ok": True}

@app.post("/api/stop")
async def stop(x_pw: str = Header(None)):
    st = auth(x_pw); st["running"] = False
    if st["task"]: st["task"].cancel()
    return {"ok": True}

@app.get("/api/status")
async def status(x_pw: str = Header(None)):
    st = auth(x_pw); return {k: st[k] for k in ("running", "log", "stats", "balance", "step")}

@app.get("/")
async def index(): return FileResponse("static/index.html")
