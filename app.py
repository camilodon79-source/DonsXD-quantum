import asyncio, os, time, secrets, json
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

def parse_ssid(s):
    """Vérifie la forme 42["auth",{...}] et renvoie True si compte démo."""
    s = (s or "").strip()
    if not s.startswith('42["auth"'):
        raise ValueError('SSID invalide : il doit commencer par 42["auth", ... Copie-le en entier.')
    try: d = json.loads(s[2:])
    except Exception: raise ValueError('SSID mal formé : copie le message brut en entier (clic droit, Copy message), avec tous les guillemets.')
    if not (isinstance(d, list) and len(d) == 2 and d[0] == "auth" and isinstance(d[1], dict)):
        raise ValueError('Format attendu : 42["auth",{...}]')
    o = d[1]
    if o.get("isChart"):
        raise ValueError("Ce message vient du socket des graphiques (isChart). Prends celui du socket de trading.")
    if not (o.get("session") or o.get("sessionToken")) or not o.get("uid"):
        raise ValueError("SSID incomplet : il manque la session ou l'uid.")
    return bool(o.get("isDemo"))

async def best_signal(api, cfg):
    """Scanne toutes les paires, garde le signal le plus fort (payout suffisant)."""
    try: payouts = await asyncio.wait_for(api.payout(), 15)
    except Exception: payouts = {}
    best, n = None, 0
    for a in ASSETS:
        p = payouts.get(a) if isinstance(payouts, dict) else None
        if p is not None and p < cfg.min_payout: continue
        try: candles = await asyncio.wait_for(api.get_candles(a, cfg.timeframe, cfg.timeframe * 80), 20)
        except Exception as e: log(f"{a}: bougies indisponibles ({e})"); continue
        n += 1
        act, sc = signal(candles)
        if act and sc >= cfg.min_score and (best is None or sc > best[2]): best = (a, act, sc)
    return best, n

async def run(cfg: Cfg):
    try:
        try: from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
        except ImportError: from BinaryOptionsToolsV2 import PocketOptionAsync
        demo = parse_ssid(cfg.ssid)
        log("Connexion à Pocket Option…")
        api = PocketOptionAsync(cfg.ssid)
        wa = getattr(api, "wait_for_assets", None)
        if wa:
            try: await asyncio.wait_for(wa(15), 25)
            except Exception: pass
        else: await asyncio.sleep(5)
        try: S["balance"] = await asyncio.wait_for(api.balance(), 25)
        except Exception: raise RuntimeError("Connexion impossible : SSID expiré ou invalide. Récupère un SSID tout neuf.")
        log(f"Connecté, compte {'DÉMO' if demo else 'RÉEL ⚠️'}, solde {S['balance']}")
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
            _, res = await asyncio.wait_for(fn(asset=asset, amount=stake, time=cfg.timeframe, check_win=True), cfg.timeframe + 60)
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

@app.post("/api/start")
async def start(cfg: Cfg, x_pw: str = Header(None)):
    auth(x_pw)
    if S["running"]: raise HTTPException(400, "Déjà en marche")
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
