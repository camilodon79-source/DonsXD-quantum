import asyncio, os, time, secrets, json, re
import urllib.request, urllib.error
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from strategy import signal

# Un ou plusieurs mots de passe séparés par des virgules. Chaque mot de passe a son propre espace.
PASSWORDS = [p.strip() for p in os.environ.get("APP_PASSWORD", "").split(",") if p.strip()]
ASSETS = os.environ.get("ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,EURJPY_otc,AUDCAD_otc,EURGBP_otc").split(",")
MAX_ASSETS = int(os.environ.get("MAX_ASSETS", "12"))
MIN_RATE = float(os.environ.get("MIN_RATE", "45"))      # arrêt auto si le taux de réussite passe sous ce % (0 = désactivé)
MIN_TRADES = int(os.environ.get("MIN_TRADES", "30"))    # ... après au moins ce nombre de trades
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
    max_stake: float = 50; max_losses: int = 5; min_score: int = 2; min_payout: int = 85
    max_loss: float = float(os.environ.get("MAX_LOSS", "0"))        # perte max de la session (0 = désactivé)
    max_minutes: int = int(os.environ.get("MAX_MINUTES", "0"))      # durée max en minutes (0 = désactivé)
    max_profit: float = float(os.environ.get("MAX_PROFIT", "0"))    # objectif de gain de la session (0 = désactivé)

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

def pick_assets(payouts, cfg):
    """Prend les paires de devises les mieux payées (jusqu'à MAX_ASSETS), sinon la liste par défaut."""
    if isinstance(payouts, dict) and payouts:
        c = [(a, p) for a, p in payouts.items()
             if isinstance(p, (int, float)) and p >= cfg.min_payout and re.fullmatch(r"[A-Z]{6}(_otc)?", str(a))]
        c.sort(key=lambda x: -x[1])
        if c: return [a for a, _ in c[:MAX_ASSETS]]
    return ASSETS

def vol_ok(candles):
    """Évite les marchés trop plats (bruit) ou en pic brutal : l'ATR récent doit rester proche de la normale."""
    h = np.array([k["high"] for k in candles]); l = np.array([k["low"] for k in candles]); c = np.array([k["close"] for k in candles])
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    ref = float(np.median(tr[-50:]))
    return ref > 0 and 0.7 <= float(tr[-14:].mean()) / ref <= 2.5

async def scan_one(api, a, cfg, sem, fn, errs):
    off = cfg.timeframe * 50
    async with sem:
        try: candles = await asyncio.wait_for(api.get_candles(a, cfg.timeframe, off), 12 + off / 250)
        except Exception as e:
            errs.append(type(e).__name__); return a, None, 0, False
    if len(candles) > 41: candles = candles[:-1]      # on ignore la bougie en cours (pas encore fermée)
    if not vol_ok(candles): return a, None, 0, True
    act, sc = fn(candles)
    return a, act, sc, True

async def best_signal(api, cfg):
    """Analyse les paires et garde le signal le plus fort. Moins de paires et moins en parallèle sur les longs timeframes."""
    pf = getattr(api, "payout", None) or getattr(api, "payouts", None)
    try: payouts = await asyncio.wait_for(pf(), 15) if pf else {}
    except Exception: payouts = {}
    fn = signal
    tf = cfg.timeframe
    limit = MAX_ASSETS if tf <= 15 else min(MAX_ASSETS, 8) if tf <= 60 else min(MAX_ASSETS, 5)
    sem = asyncio.Semaphore(8 if tf <= 5 else 4 if tf <= 30 else 1)
    errs = []
    res = await asyncio.gather(*[scan_one(api, a, cfg, sem, fn, errs) for a in pick_assets(payouts, cfg)[:limit]])
    best, n = None, 0
    for a, act, sc, ok in res:
        if not ok: continue
        n += 1
        pay = payouts.get(a, 0) if isinstance(payouts, dict) else 0
        if act and sc >= cfg.min_score and (best is None or (sc, pay) > (best[2], best[3])): best = (a, act, sc, pay)
    return best, n, errs

def rate(st):
    w, l = st["stats"]["win"], st["stats"]["loss"]
    return f"{w}/{w + l} = {round(100 * w / (w + l))}%" if w + l else "0/0"

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
        start_bal = float(st["balance"]) if st["balance"] is not None else None; t0 = time.time()
        if cfg.max_loss: log(st, f"Protection : arrêt si la perte dépasse {cfg.max_loss}")
        if cfg.max_minutes: log(st, f"Protection : arrêt après {cfg.max_minutes} min")
        if cfg.max_profit: log(st, f"Objectif : arrêt si le gain atteint {cfg.max_profit}")
        while st["running"]:
            if cfg.max_minutes and time.time() - t0 > cfg.max_minutes * 60:
                log(st, f"Arrêt : durée maximale atteinte ({cfg.max_minutes} min)"); break
            if time.time() - last_bal > 20:
                st["balance"] = await asyncio.wait_for(api.balance(), 25); last_bal = time.time()
            t_s = time.time()
            sig, n, errs = await best_signal(api, cfg)
            if not sig:
                if time.time() - last_hb > 60:
                    log(st, f"Aucun signal pour l'instant ({n} paires analysées)" + (f" — erreurs : {errs[0]} x{len(errs)}" if errs else "")); last_hb = time.time()
                await asyncio.sleep(1 if n else 5); continue
            if cfg.timeframe <= 30 and time.time() - t_s > max(cfg.timeframe, 4): continue  # analyse trop lente : signal périmé
            asset, act, sc, _ = sig
            stake = round(min(cfg.amount * cfg.multiplier ** st["step"], cfg.max_stake), 2)
            log(st, f"{act} {asset} {stake:.2f} (force {sc}, martingale {st['step']}/{cfg.martingales})")
            fn = api.buy if act == "CALL" else api.sell
            r0 = await asyncio.wait_for(fn(asset=asset, amount=stake, time=cfg.timeframe), 30)
            trade_id = r0[0] if isinstance(r0, (tuple, list)) else r0
            res = await asyncio.wait_for(api.check_win(trade_id), cfg.timeframe + 60)
            out = trade_result(res); lost_cycle = False
            if out == "win": st["stats"]["win"] += 1; st["step"] = 0; loss_streak = 0; log(st, "✅ Gagné")
            elif out == "loss":
                st["stats"]["loss"] += 1; loss_streak += 1; log(st, "❌ Perdu")
                maxed = st["step"] >= cfg.martingales
                lost_cycle = maxed and cfg.martingales > 0
                st["step"] = 0 if maxed else st["step"] + 1
                if loss_streak >= cfg.max_losses: log(st, "Arrêt : trop de pertes d'affilée"); st["running"] = False
            else: st["stats"]["draw"] += 1; log(st, f"↔ Égalité ({str(res)[:80]})")
            log(st, f"Bilan : {rate(st)} de réussite")
            w, l = st["stats"]["win"], st["stats"]["loss"]
            if MIN_RATE and w + l >= MIN_TRADES and 100 * w / (w + l) < MIN_RATE:
                log(st, f"Arrêt : taux de réussite trop bas ({rate(st)})"); break
            if lost_cycle:
                log(st, "Cycle de martingale perdu : pause de 5 min")
                for _ in range(300):
                    if not st["running"]: break
                    await asyncio.sleep(1)
            try: st["balance"] = await asyncio.wait_for(api.balance(), 15); last_bal = time.time()
            except Exception: last_bal = 0
            if cfg.max_loss and start_bal is not None and st["balance"] is not None and start_bal - float(st["balance"]) >= cfg.max_loss:
                log(st, f"Arrêt : perte maximale atteinte (-{start_bal - float(st['balance']):.2f})"); break
            if cfg.max_profit and start_bal is not None and st["balance"] is not None and float(st["balance"]) - start_bal >= cfg.max_profit:
                log(st, f"Arrêt : objectif de gain atteint (+{float(st['balance']) - start_bal:.2f})"); break
            await asyncio.sleep(min(cfg.timeframe, 60))  # pause d'une bougie : pas de trades en rafale
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

SOUND_JS = r"""<script>
(function(){
var ctx=null,muted=localStorage.getItem('mute')==='1',q=[],busy=false,lastWarn=0;
function ac(){if(!ctx){var C=window.AudioContext||window.webkitAudioContext;if(!C)return null;ctx=new C();}if(ctx.state==='suspended')ctx.resume();return ctx;}
function tone(f,t0,d,type,v){var c=ac();if(!c||muted)return;var o=c.createOscillator(),g=c.createGain();o.type=type||'sine';o.frequency.value=f;o.connect(g);g.connect(c.destination);var t=c.currentTime+t0;g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(v||0.18,t+0.02);g.gain.exponentialRampToValueAtTime(0.0001,t+d);o.start(t);o.stop(t+d+0.05);}
var S={
connect:function(){tone(523,0,.12);tone(784,.13,.22);},
open:function(){tone(660,0,.08,'square',.07);},
mart:function(){tone(392,0,.1,'triangle');tone(392,.15,.1,'triangle');},
win:function(){tone(523,0,.1);tone(659,.11,.1);tone(784,.22,.3);},
loss:function(){tone(220,0,.25,'sawtooth',.13);tone(165,.2,.35,'sawtooth',.13);},
warn:function(){tone(330,0,.18,'square',.1);tone(247,.2,.3,'square',.1);},
stop:function(){tone(392,0,.12);tone(262,.14,.26);}
};
function enqueue(f){q.push(f);if(busy)return;busy=true;(function next(){var g=q.shift();if(!g){busy=false;return;}g();setTimeout(next,450);})();}
function sfx(t){
var m=String(t).replace(/^\d\d:\d\d:\d\d\s+/,'');
if(/^Connecté/.test(m))enqueue(S.connect);
else if(/^(CALL|PUT) /.test(m)){var x=m.match(/martingale (\d+)\//);enqueue(x&&+x[1]>0?S.mart:S.open);}
else if(m.indexOf('Gagné')>-1)enqueue(S.win);
else if(m.indexOf('Perdu')>-1)enqueue(S.loss);
else if(/^(Arrêt|Erreur|Connexion impossible|Connexion perdue|Cycle de martingale)/.test(m)){lastWarn=Date.now();enqueue(S.warn);}
else if(m==='Bot arrêté'){if(Date.now()-lastWarn>3000)enqueue(S.stop);}
}
var t0=0;
try{
var _add=add;
add=function(t){_add(t);var n=Date.now();if(!t0)t0=n;if(n-t0>1500)sfx(t);};
}catch(e){}
var b=document.createElement('button');
function paint(){b.textContent=muted?'🔇':'🔊';b.setAttribute('aria-label',muted?'Activer les sons':'Couper les sons');}
b.style.cssText='margin-left:auto;background:none;border:0;font-size:1.2rem;cursor:pointer;padding:4px 8px;color:inherit';
paint();
b.onclick=function(){muted=!muted;localStorage.setItem('mute',muted?'1':'0');paint();if(!muted){ac();S.connect();}};
var h=document.querySelector('header'),d=document.getElementById('dot');
if(h&&d){d.style.marginLeft='8px';h.insertBefore(b,d);}
document.addEventListener('pointerdown',function(){ac();});
})();
</script>"""

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
async def index():
    with open("static/index.html", encoding="utf-8") as f: html = f.read()
    return HTMLResponse(html.replace("</body>", SOUND_JS + "</body>"))
