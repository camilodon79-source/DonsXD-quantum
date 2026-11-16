"""QuantumFlux (port de quantum_flux.chtl). Entrée : bougies [{open,high,low,close}], plus ancienne -> récente."""
import numpy as np

def haar_denoise(x, level=4):
    x = np.asarray(x, float); n = len(x)
    m = 2 ** level; pad = (-n) % m
    y = np.concatenate([x, np.full(pad, x[-1])])
    det = []
    for _ in range(level):
        a = (y[0::2] + y[1::2]) / np.sqrt(2); det.append((y[0::2] - y[1::2]) / np.sqrt(2)); y = a
    sig = np.median(np.abs(det[0])) / 0.6745 + 1e-12
    thr = sig * np.sqrt(2 * np.log(len(x)))
    for i in range(level - 1, -1, -1):
        d = np.sign(det[i]) * np.maximum(np.abs(det[i]) - thr, 0)
        z = np.empty(len(y) * 2); z[0::2] = (y + d) / np.sqrt(2); z[1::2] = (y - d) / np.sqrt(2); y = z
    return y[:n]

def ema(x, p):
    k = 2 / (p + 1); out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)): out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out

def rsi(c, p=14):
    d = np.diff(c, prepend=c[0]); g = np.where(d > 0, d, 0); l = np.where(d < 0, -d, 0)
    ag = ema(g, 2 * p - 1); al = ema(l, 2 * p - 1)
    return 100 - 100 / (1 + ag / (al + 1e-12))

def signal(candles):
    """Retourne (action, score) ; action in CALL/PUT/None ; score 3 = fort, 2 = moyen."""
    if len(candles) < 40: return None, 0
    o, h, l, c = (haar_denoise([k[f] for k in candles]) for f in ("open", "high", "low", "close"))
    f, s, t = ema(c, 5), ema(c, 13), ema(c, 21)
    sma, sd = c[-20:].mean(), c[-20:].std()
    r = rsi(c); r0, r1 = r[-2], r[-1]
    up, dn = f[-1] > s[-1], f[-1] < s[-1]
    aup, adn = s[-1] > t[-1], s[-1] < t[-1]
    cl = c[-1]; bull, bear = cl > o[-1], cl < o[-1]
    rng = h[-1] - l[-1]; nh = cl > h[-1] - rng * .2; nl = cl < l[-1] + rng * .2
    xlo, xhi = cl < sma - sd * 2, cl > sma + sd * 2
    olo, ohi = cl < sma - sd * 1.2, cl > sma + sd * 1.2
    co = lambda lv: r0 <= lv < r1; cu = lambda lv: r0 >= lv > r1
    rules = [
        (xlo and r1 < 30 and bull and nh, "CALL", 3), (xhi and r1 > 70 and bear and nl, "PUT", 3),
        (up and aup and co(30) and bull, "CALL", 3), (dn and adn and cu(70) and bear, "PUT", 3),
        (olo and r1 < 45 and bull, "CALL", 2), (ohi and r1 > 55 and bear, "PUT", 2),
        (up and olo and r1 < 45, "CALL", 2), (dn and ohi and r1 > 55, "PUT", 2),
        (up and co(35), "CALL", 2), (dn and cu(65), "PUT", 2),
    ]
    for cond, act, sc in rules:
        if cond: return act, sc
    return None, 0
