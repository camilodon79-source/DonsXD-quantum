import asyncio, os, time, secrets, json, re
import urllib.request, urllib.error
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from strategy import signal, ema, rsi

# Un ou plusieurs mots de passe séparés par des virgules. Chaque mot de passe a son propre espace.
PASSWORDS = [p.strip() for p in os.environ.get("APP_PASSWORD", "").split(",") if p.strip()]
ASSETS = os.environ.get("ASSETS", "EURUSD_otc,GBPUSD_otc,USDJPY_otc,AUDUSD_otc,EURJPY_otc,AUDCAD_otc,EURGBP_otc").split(",")
MAX_ASSETS = int(os.environ.get("MAX_ASSETS", "60"))      # nombre max de paires éligibles surveillées
TREND_MAX = float(os.environ.get("TREND_MAX", "1.5"))     # refuse un trade contre une tendance courte plus forte que ça
MIN_RATE = float(os.environ.get("MIN_RATE", "45"))      # arrêt auto si le taux de réussite passe sous ce % (0 = désactivé)
MIN_TRADES = int(os.environ.get("MIN_TRADES", "30"))    # ... après au moins ce nombre de trades
app = FastAPI()
STATES = {}

def new_state():
    return {"running": False, "log": [], "stats": {"win": 0, "loss": 0, "draw": 0}, "balance": None, "step": 0, "task": None, "journal": []}

def auth(pw):
    if PASSWORDS and not any(secrets.compare_digest(pw or "", p) for p in PASSWORDS):
        raise HTTPException(401, "Mot de passe incorrect")
    return STATES.setdefault(pw if PASSWORDS else "", new_state())

class Cfg(BaseModel):
    ssid: str; timeframe: int = 60; amount: float = 1; martingales: int = 2; multiplier: float = 2.2
    max_stake: float = 50; max_losses: int = 5; min_score: int = 2; min_payout: int = 90
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


def ctime(c):
    """Horodatage (secondes) d'une bougie, ou None si introuvable."""
    for k in ("time", "timestamp", "t", "open_time"):
        v = c.get(k) if isinstance(c, dict) else None
        if v is None: continue
        try:
            if hasattr(v, "timestamp"): return int(v.timestamp())
            if isinstance(v, str):
                try: v = float(v)
                except ValueError: return int(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp())
            v = float(v); return int(v / 1000) if v > 1e12 else int(v)
        except Exception: pass
    return None

class Mem:
    """Mémoire des bougies : on charge l'historique une fois, puis on ne redemande que les dernières bougies."""
    def __init__(self): self.ok = None; self.note = None; self.store = {}; self.fail = {}; self.skip = {}; self.rot = 0

def vol_ok(candles):
    """Évite les marchés trop plats (bruit) ou en pic brutal : l'ATR récent doit rester proche de la normale."""
    h = np.array([k["high"] for k in candles], float); l = np.array([k["low"] for k in candles], float); c = np.array([k["close"] for k in candles], float)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    ref = float(np.median(tr[-50:]))
    return ref > 0 and 0.7 <= float(tr[-14:].mean()) / ref <= 2.5

def feats(c):
    """Conditions du marché au moment du signal (pour le filtre et le journal)."""
    cl = np.array([k["close"] for k in c], float); op = np.array([k["open"] for k in c], float)
    h = np.array([k["high"] for k in c], float); l = np.array([k["low"] for k in c], float)
    pc = np.r_[cl[0], cl[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = float(tr[-14:].mean()); ref = float(np.median(tr[-50:]))
    trend = float((ema(cl, 8)[-1] - ema(cl, 34)[-1]) / max(atr, 1e-12))
    return {"rsi": round(float(rsi(cl)[-1]), 1), "trend": round(trend, 2), "vol": round(atr / ref if ref > 0 else 0, 2),
            "dir": 1 if cl[-1] > op[-1] else -1 if cl[-1] < op[-1] else 0}

def accept(act, f):
    """Filtres anti-échec : la dernière bougie doit aller dans le sens du trade, et pas de trade contre une forte tendance."""
    d = 1 if act == "CALL" else -1
    return f["dir"] == d and not (f["trend"] * d < -TREND_MAX)

async def get_payouts(api):
    """Payout de toutes les paires (payouts() = toutes ; payout(asset) = une seule)."""
    for name in ("payouts", "payout"):
        f = getattr(api, name, None)
        if not f: continue
        try: r = await asyncio.wait_for(f(), 15)
        except Exception: continue
        if isinstance(r, dict): items = list(r.items())
        elif isinstance(r, (list, tuple)): items = [(x[0], x[1]) for x in r if isinstance(x, (list, tuple)) and len(x) >= 2]
        else: continue
        d = {}
        for k, v in items:
            try: d[str(k)] = float(v)
            except Exception: pass
        if d:
            if max(d.values()) <= 1.5: d = {k: v * 100 for k, v in d.items()}
            return d
    return {}

def eligible(payouts, cfg, mem):
    now = time.time()
    c = sorted(((a, p) for a, p in payouts.items() if p >= cfg.min_payout and mem.skip.get(a, 0) < now), key=lambda x: -x[1])
    return [a for a, _ in c[:MAX_ASSETS]]

async def get_closed(api, a, cfg, mem, errs):
    """Bougies terminées d'une paire (50 max). Utilise la mémoire si les bougies sont horodatées."""
    tf = cfg.timeframe; d = mem.store.get(a); inc = bool(mem.ok and d)
    off = 4 if inc else 53
    try: raw = await asyncio.wait_for(api.get_candles(a, tf, off), 12 + off * tf / 250)
    except Exception as e:
        errs.append(type(e).__name__)
        n = mem.fail[a] = mem.fail.get(a, 0) + 1
        if n >= 3: mem.skip[a] = time.time() + 600; mem.fail[a] = 0   # paire qui ne répond pas : on la met de côté 10 min
        return None
    mem.fail[a] = 0
    now = time.time(); ts = [ctime(c) for c in raw]
    if mem.ok is None and not inc and len(raw) >= 20:
        steps = [y - x for x, y in zip(ts, ts[1:]) if x is not None and y is not None]
        mem.ok = len(steps) >= 0.9 * (len(raw) - 1) and sum(1 for s_ in steps if s_ == tf) >= 0.9 * len(steps)
        mem.note = "activée" if mem.ok else "désactivée (pas d'horodatage exploitable)"
    if not mem.ok:
        return raw[:-1] if len(raw) > 42 else None
    d = d if d is not None else {}
    for c, t in zip(raw, ts):
        if t is not None: d[t] = c
    for t in sorted(d)[:-80]: del d[t]
    mem.store[a] = d
    last = [t for t in sorted(d) if t + tf <= now + 1][-45:]
    if len(last) < 42 or any(y - x != tf for x, y in zip(last, last[1:])):
        mem.store.pop(a, None); return None             # trou dans l'historique : on recharge tout au prochain tour
    if last[-1] < int(now // tf) * tf - tf: return None  # données en retard
    return [d[t] for t in last]

async def scan(api, cfg, mem, assets, errs):
    sem = asyncio.Semaphore(6 if cfg.timeframe <= 15 else 4)
    async def one(a):
        async with sem:
            c = await get_closed(api, a, cfg, mem, errs)
        if c is None: return a, None, 0, None, False
        if not vol_ok(c): return a, None, 0, None, True
        act, sc = signal(c)
        if not act: return a, None, 0, None, True
        f = feats(c)
        return (a, act, sc, f, True) if accept(act, f) else (a, None, 0, None, True)
    return await asyncio.gather(*[one(a) for a in assets])

def journal_report(st):
    j = [r for r in st["journal"] if r.get("res") in ("win", "loss")]
    if not j: return "Journal vide : aucun trade terminé pour l'instant."
    def wr(rows):
        w = sum(1 for r in rows if r["res"] == "win"); return f"{w}/{len(rows)} = {round(100 * w / len(rows))}%"
    pay = sum(r["pay"] for r in j) / len(j)
    out = [f"BILAN GLOBAL : {wr(j)}   (pour ne pas perdre il faut environ {round(100 / (1 + pay / 100))}%)"]
    def group(title, keyf, minn=1):
        g = {}
        for r in j: g.setdefault(keyf(r), []).append(r)
        out.append("\n" + title)
        for k in sorted(g, key=str):
            if len(g[k]) >= minn: out.append(f"  {k} : {wr(g[k])}")
    sens = lambda r: r["trend"] * (1 if r["act"] == "CALL" else -1)
    group("Par force du signal", lambda r: r["sc"])
    group("Par tendance", lambda r: "dans le sens" if sens(r) > 0.5 else "contre" if sens(r) < -0.5 else "neutre")
    group("Par volatilité", lambda r: "calme (<1)" if r["vol"] < 1 else "normale (1 à 1,5)" if r["vol"] < 1.5 else "agitée (>1,5)")
    group("Par étage de martingale", lambda r: r["step"])
    group("Par paire (5 trades minimum)", lambda r: r["a"], 5)
    return "\n".join(out)

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
    api = None; mem = Mem()
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
        tf = cfg.timeframe
        payouts = await get_payouts(api); last_pay = time.time()
        elig = eligible(payouts, cfg, mem) if payouts else list(ASSETS)
        log(st, f"{len(elig)} paires à {cfg.min_payout}% ou plus" if payouts else "Payouts indisponibles : liste de secours (7 paires)")
        lim = {5: 6, 10: 8, 15: 10, 30: 14}.get(tf, 16)
        st["step"] = 0; loss_streak = 0; last_hb = 0; noted = False; slow_n = 0
        start_bal = float(st["balance"]) if st["balance"] is not None else None; t0 = time.time()
        if cfg.max_loss: log(st, f"Protection : arrêt si la perte dépasse {cfg.max_loss}")
        if cfg.max_minutes: log(st, f"Protection : arrêt après {cfg.max_minutes} min")
        if cfg.max_profit: log(st, f"Objectif : arrêt si le gain atteint {cfg.max_profit}")
        while st["running"]:
            if cfg.max_minutes and time.time() - t0 > cfg.max_minutes * 60:
                log(st, f"Arrêt : durée maximale atteinte ({cfg.max_minutes} min)"); break
            target = (time.time() // tf + 1) * tf + 0.4          # on analyse juste après la fermeture de chaque bougie
            while st["running"] and time.time() < target: await asyncio.sleep(min(1, max(0.05, target - time.time())))
            if not st["running"]: break
            if time.time() - last_pay > 120:
                p2 = await get_payouts(api); last_pay = time.time()
                if p2: payouts = p2; elig = eligible(payouts, cfg, mem)
            if not elig:
                if time.time() - last_hb > 60: log(st, f"Aucune paire à {cfg.min_payout}% ou plus pour l'instant"); last_hb = time.time()
                continue
            k = min(lim, len(elig))
            chosen = [elig[(mem.rot + i) % len(elig)] for i in range(k)]; mem.rot = (mem.rot + k) % len(elig)
            t_s = time.time(); errs = []
            res = await scan(api, cfg, mem, chosen, errs)
            dur = time.time() - t_s
            if dur < 0.3 * tf and lim < len(elig): lim += 3         # assez rapide : on surveille plus de paires par tour
            elif dur > 0.55 * tf: lim = max(3, lim - 3)             # trop lent : on en surveille moins par tour
            if mem.note and not noted: log(st, f"Mémoire des bougies : {mem.note}"); noted = True
            n = sum(1 for r in res if r[4])
            cands = [r for r in res if r[1] and r[2] >= cfg.min_score]
            if not cands:
                if time.time() - last_hb > 60:
                    log(st, f"Aucun signal ({n}/{k} paires lues ce tour, {len(elig)} à {cfg.min_payout}%+, analyse {dur:.1f}s)" + (f" — erreurs : {errs[0]} x{len(errs)}" if errs else "")); last_hb = time.time()
                continue
            if dur > (max(3.0, 0.6 * tf) if tf <= 30 else min(25.0, 0.3 * tf)):
                slow_n += 1
                if slow_n % 5 == 1: log(st, f"Signal ignoré : analyse trop lente ({dur:.1f}s)")
                continue
            asset, act, sc, f, _ = max(cands, key=lambda r: (r[2], payouts.get(r[0], 0)))
            stake = round(min(cfg.amount * cfg.multiplier ** st["step"], cfg.max_stake), 2)
            log(st, f"{act} {asset} {stake:.2f} (force {sc}, martingale {st['step']}/{cfg.martingales})")
            rec = {"t": int(time.time()), "a": asset, "tf": tf, "act": act, "sc": sc, "stake": stake, "step": st["step"], "pay": payouts.get(asset, 0), **f}
            fn = api.buy if act == "CALL" else api.sell
            r0 = await asyncio.wait_for(fn(asset=asset, amount=stake, time=tf), 30)
            trade_id = r0[0] if isinstance(r0, (tuple, list)) else r0
            res_ = await asyncio.wait_for(api.check_win(trade_id), tf + 60)
            out = trade_result(res_); lost_cycle = False
            rec["res"] = out; st["journal"].append(rec); del st["journal"][:-1000]
            if out == "win": st["stats"]["win"] += 1; st["step"] = 0; loss_streak = 0; log(st, "✅ Gagné")
            elif out == "loss":
                st["stats"]["loss"] += 1; loss_streak += 1; log(st, "❌ Perdu")
                maxed = st["step"] >= cfg.martingales
                lost_cycle = maxed and cfg.martingales > 0
                st["step"] = 0 if maxed else st["step"] + 1
                if loss_streak >= cfg.max_losses: log(st, "Arrêt : trop de pertes d'affilée"); st["running"] = False
            else: st["stats"]["draw"] += 1; log(st, f"↔ Égalité ({str(res_)[:80]})")
            log(st, f"Bilan : {rate(st)} de réussite")
            w, l = st["stats"]["win"], st["stats"]["loss"]
            if MIN_RATE and w + l >= MIN_TRADES and 100 * w / (w + l) < MIN_RATE:
                log(st, f"Arrêt : taux de réussite trop bas ({rate(st)})"); break
            if lost_cycle:
                log(st, "Cycle de martingale perdu : pause de 5 min")
                for _ in range(300):
                    if not st["running"]: break
                    await asyncio.sleep(1)
            try: st["balance"] = await asyncio.wait_for(api.balance(), 15)
            except Exception: pass
            if cfg.max_loss and start_bal is not None and st["balance"] is not None and start_bal - float(st["balance"]) >= cfg.max_loss:
                log(st, f"Arrêt : perte maximale atteinte (-{start_bal - float(st['balance']):.2f})"); break
            if cfg.max_profit and start_bal is not None and st["balance"] is not None and float(st["balance"]) - start_bal >= cfg.max_profit:
                log(st, f"Arrêt : objectif de gain atteint (+{float(st['balance']) - start_bal:.2f})"); break
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
b.style.cssText='background:none;border:0;font-size:1.2rem;cursor:pointer;padding:4px 8px;color:inherit';
var j=document.createElement('button');j.textContent='📊';j.setAttribute('aria-label','Journal des trades');j.style.cssText='margin-left:auto;background:none;border:0;font-size:1.2rem;cursor:pointer;padding:4px 8px;color:inherit';j.onclick=function(){var p=document.getElementById('pw');window.open('/api/journal?pw='+encodeURIComponent(p?p.value:''),'_blank');};
paint();
b.onclick=function(){muted=!muted;localStorage.setItem('mute',muted?'1':'0');paint();if(!muted){ac();S.connect();}};
var h=document.querySelector('header'),d=document.getElementById('dot');
if(h&&d){d.style.marginLeft='8px';h.insertBefore(j,d);h.insertBefore(b,d);}
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

@app.get("/api/journal")
async def journal(pw: str = "", x_pw: str = Header(None)):
    st = auth(x_pw or pw)
    return PlainTextResponse(journal_report(st))

@app.get("/")
async def index():
    with open("static/index.html", encoding="utf-8") as f: html = f.read()
    html = html.replace('<button data-v="5">5s</button>', '<button data-v="5">5s</button><button data-v="10">10s</button>')
    return HTMLResponse(html.replace("</body>", SOUND_JS + "</body>"))

# FIN DU FICHIER
