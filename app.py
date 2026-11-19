import asyncio, os, time, secrets, json, re
import urllib.request, urllib.error
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
from strategy import signal

PASSWORD = os.environ.get("APP_PASSWORD", "")
ASSETS = os.environ.get("ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,EURJPY_otc,AUDCAD_otc,EURGBP_otc").split(",")
app = FastAPI()
S = {"running": False, "log": [], "stats": {"win": 0, "loss": 0, "draw": 0}, "balance": None, "step": 0, "task": None}

class Cfg(BaseModel):
    ssid: str; timeframe: int = 60; amount: float = 1; martingales: int = 2; multiplier: float = 2.2
    max_stake: float = 50; max_losses: int = 5; min_score: int = 2; min_payout: int = 80

def log(m):
    S["log"].append(f"{time.strftime('%H:%M:%S')}  {m}"); del S["log"][:-80]

def auth(pw):
    if PASSWORD and not secrets.compare_digest(pw or "", PASSWORD): raise HTTPException(401, "Mot de passe incorrect")

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
        raise ValueError("Ton SSID contient « … » : c'est un aperçu coupé. Copie le message complet (clic droit, Copy message).")
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

async def best_signal(api, cfg):
    """Scanne toutes les paires, garde le signal le plus fort (payout suffisant)."""
    pf = getattr(api, "payout", None) or getattr(api, "payouts", None)
    try: payouts = await asyncio.wait_for(pf(), 15) if pf else {}
    except Exception: payouts = {}
    best, n = None, 0
    for a in ASSETS:
        p = payouts.get(a) if isinstance(payouts, dict) else None
        if p is not None and p < cfg.min_payout: continue
        try: candles = await asyncio.wait_for(api.get_candles(a, cfg.timeframe, cfg.timeframe * 80), 20)
        except Exception as e: log(f"{a}: bougies indisponibles ({type(e).__name__}: {e})"); continue
        n += 1
        act, sc = signal(candles)
        if act and sc >= cfg.min_score and (best is None or sc > best[2]): best = (a, act, sc)
    return best, n

async def run(cfg: Cfg):
    try:
        try: from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
        except ImportError: from BinaryOptionsToolsV2 import PocketOptionAsync
        try:
            from BinaryOptionsToolsV2.tracing import start_logs
            start_logs(path="/tmp/po_logs", level="INFO", terminal=True)
        except Exception: pass
        demo = parse_ssid(cfg.ssid)
        log("Connexion à Pocket Option…")
        api = PocketOptionAsync(cfg.ssid)
        wa = getattr(api, "wait_for_assets", None)
        if wa:
            try: await asyncio.wait_for(wa(15), 25)
            except Exception: pass
        else: await asyncio.sleep(5)
        try: S["balance"] = await asyncio.wait_for(api.balance(), 25)
        except asyncio.TimeoutError:
            raise RuntimeError("Connexion impossible : Pocket Option n'a pas répondu en 25 s (token refusé, ou adresse du serveur bloquée).")
        except Exception as e:
            raise RuntimeError(f"Connexion impossible : {type(e).__name__}: {e}")
        mode = "DÉMO" if demo is True else "RÉEL ⚠️" if demo is False else "type inconnu, vérifie ton solde"
        log(f"Connecté, compte {mode}, solde {S['balance']}")
        S["step"] = 0; loss_streak = 0; last_hb = 0
        while S["running"]:
            S["balance"] = await asyncio.wait_for(api.balance(), 25)
            sig, n = await best_signal(api, cfg)
            if not sig:
                if time.time() - last_hb > 60: log(f"Aucun signal pour l'instant ({n} paires analysées)"); last_hb = time.time()
                await asyncio.sleep(max(5, cfg.timeframe // 4)); continue
            asset, act, sc = sig
            stake = min(cfg.amount * cfg.multiplier ** S["step"], cfg.max_stake)
            log(f"{act} {asset} {stake:.2f} (force {sc}, martingale {S['step']}/{cfg.martingales})")
            fn = api.buy if act == "CALL" else api.sell
            r0 = await asyncio.wait_for(fn(asset=asset, amount=stake, time=cfg.timeframe), 30)
            trade_id = r0[0] if isinstance(r0, (tuple, list)) else r0
            res = await asyncio.wait_for(api.check_win(trade_id), cfg.timeframe + 60)
            r = str(res.get("result", res)).lower() if isinstance(res, dict) else str(res).lower()
            if "win" in r: S["stats"]["win"] += 1; S["step"] = 0; loss_streak = 0; log("✅ Gagné")
            elif "loss" in r or "lose" in r:
                S["stats"]["loss"] += 1; loss_streak += 1; log("❌ Perdu")
                S["step"] = S["step"] + 1 if S["step"] < cfg.martingales else 0
                if loss_streak >= cfg.max_losses: log("Arrêt : trop de pertes d'affilée"); S["running"] = False
            else: S["stats"]["draw"] += 1; log("↔ Égalité")
            await asyncio.sleep(2)
    except asyncio.CancelledError: pass
    except Exception as e: log(f"Erreur : {e}")
    finally:
        S["running"] = False; log("Bot arrêté")

def _probe(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return f"{r.status} {r.read(60).decode('utf-8', 'ignore')!r}"
    except urllib.error.HTTPError as e: return f"HTTP {e.code} (serveur: {e.headers.get('server', '?')})"
    except Exception as e: return f"{type(e).__name__}: {e}"

@app.get("/api/diag")
async def diag():
    urls = ["https://pocketoption.com/en/",
            "https://api-eu.po.market/socket.io/?EIO=4&transport=polling",
            "https://api-sc.po.market/socket.io/?EIO=4&transport=polling",
            "https://demo-api-eu.po.market/socket.io/?EIO=4&transport=polling"]
    res = await asyncio.gather(*[asyncio.to_thread(_probe, u) for u in urls])
    return dict(zip(urls, res))

@app.post("/api/start")
async def start(cfg: Cfg, x_pw: str = Header(None)):
    auth(x_pw)
    if S["running"]: raise HTTPException(400, "Déjà en marche")
    cfg.ssid = clean_ssid(cfg.ssid)
    try: parse_ssid(cfg.ssid)
    except ValueError as e: raise HTTPException(400, str(e))
    S.update(running=True, stats={"win": 0, "loss": 0, "draw": 0}, balance=None); S["task"] = asyncio.create_task(run(cfg)); return {"ok": True}

@app.post("/api/stop")
async def stop(x_pw: str = Header(None)):
    auth(x_pw); S["running"] = False
    if S["task"]: S["task"].cancel()
    return {"ok": True}

@app.get("/api/status")
async def status(x_pw: str = Header(None)):
    auth(x_pw); return {k: S[k] for k in ("running", "log", "stats", "balance", "step")}

@app.get("/")
async def index(): return FileResponse("static/index.html")
