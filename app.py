import asyncio, os, time, secrets
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

async def best_signal(api, cfg):
    """Scanne toutes les paires, garde le signal le plus fort (payout suffisant)."""
    try: payouts = await api.payout()
    except Exception: payouts = {}
    best = None
    for a in ASSETS:
        p = payouts.get(a) if isinstance(payouts, dict) else None
        if p is not None and p < cfg.min_payout: continue
        try: candles = await api.get_candles(a, cfg.timeframe, cfg.timeframe * 80)
        except Exception as e: log(f"{a}: bougies indisponibles ({e})"); continue
        act, sc = signal(candles)
        if act and sc >= cfg.min_score and (best is None or sc > best[2]): best = (a, act, sc)
    return best

async def run(cfg: Cfg):
    from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
    api = PocketOptionAsync(cfg.ssid); await asyncio.sleep(5)
    S["step"] = 0; loss_streak = 0; log("Connecté, scan des paires…")
    while S["running"]:
        try:
            S["balance"] = await api.balance()
            sig = await best_signal(api, cfg)
            if not sig: await asyncio.sleep(max(5, cfg.timeframe // 4)); continue
            asset, act, sc = sig
            stake = min(cfg.amount * cfg.multiplier ** S["step"], cfg.max_stake)
            log(f"{act} {asset} {stake:.2f} (force {sc}, martingale {S['step']}/{cfg.martingales})")
            fn = api.buy if act == "CALL" else api.sell
            _, res = await fn(asset=asset, amount=stake, time=cfg.timeframe, check_win=True)
            r = str(res.get("result", res)).lower() if isinstance(res, dict) else str(res).lower()
            if "win" in r: S["stats"]["win"] += 1; S["step"] = 0; loss_streak = 0; log("✅ Gagné")
            elif "loss" in r or "lose" in r:
                S["stats"]["loss"] += 1; loss_streak += 1; log("❌ Perdu")
                S["step"] = S["step"] + 1 if S["step"] < cfg.martingales else 0
                if loss_streak >= cfg.max_losses: log("Arrêt : trop de pertes d'affilée"); S["running"] = False
            else: S["stats"]["draw"] += 1; log("↔ Égalité")
            await asyncio.sleep(2)
        except asyncio.CancelledError: break
        except Exception as e: log(f"Erreur : {e}"); await asyncio.sleep(10)
    S["running"] = False; log("Bot arrêté")

@app.post("/api/start")
async def start(cfg: Cfg, x_pw: str = Header(None)):
    auth(x_pw)
    if S["running"]: raise HTTPException(400, "Déjà en marche")
    S.update(running=True, stats={"win": 0, "loss": 0, "draw": 0}); S["task"] = asyncio.create_task(run(cfg)); return {"ok": True}

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
