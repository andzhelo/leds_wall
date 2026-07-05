#!/usr/bin/env python3
"""
niji_web.py - phone-friendly web control panel for the Niji LED panel.

Runs the DDP streaming itself (importing niji.py) in a background thread, so
tapping an effect switches instantly. Open the printed URL on your phone.

    python3 niji_web.py            # serves on http://<this-machine>:8080

No external dependencies (pure standard library + niji.py).
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import niji

PORT = 8080

# Effects shown in the UI (skip the calibrate/diag diagnostics), with emoji.
UI_EFFECTS = [
    ("clock", "🕐 Clock"), ("text", "🔤 Text"),
    ("fire", "🔥 Fire"), ("fireworks", "🎆 Fireworks"),
    ("lightning", "⚡ Lightning"), ("plasma", "🌈 Plasma"),
    ("aurora", "🌌 Aurora"), ("warp", "🚀 Warp"),
    ("helix", "🧬 Helix"), ("vortex", "🌀 Vortex"),
    ("metaballs", "🫧 Lava"), ("ripple", "💧 Ripple"),
    ("rain", "🟩 Matrix"), ("meteors", "☄️ Meteors"),
    ("life", "♟️ Life"), ("wipe", "🎨 Rainbow"),
]


# ----------------------------------------------------------------------------
# Streaming controller
# ----------------------------------------------------------------------------
class Controller:
    def __init__(self):
        self.ddp = niji.DDP(vflip=niji.VFLIP, hflip=niji.HFLIP)
        self.lock = threading.Lock()
        self.effect = None            # None = not streaming (WLED reverts)
        self.bri = 0.4
        self.fps = 30
        self.text = niji.CFG["text"]
        self._gen = None
        self._cur = (None, None)      # (effect, text) the generator was built for
        threading.Thread(target=self._loop, daemon=True).start()

    def snapshot(self):
        with self.lock:
            return {"effect": self.effect, "brightness": round(self.bri * 100),
                    "text": self.text, "fps": self.fps}

    def update(self, effect=None, brightness=None, text=None, fps=None,
               has_effect=False):
        with self.lock:
            if has_effect:
                self.effect = effect
            if brightness is not None:
                self.bri = max(0.0, min(1.0, brightness / 100.0))
            if text is not None:
                self.text = text
            if fps is not None:
                self.fps = max(5, min(60, int(fps)))

    def _loop(self):
        next_t = time.time()
        last_off = False
        while True:
            with self.lock:
                eff, bri, fps, txt = self.effect, self.bri, self.fps, self.text
            if eff is None:
                if not last_off:                       # blank once, then idle
                    try:
                        self.ddp.send(niji.np.zeros((niji.H, niji.W, 3)))
                    except OSError:
                        pass
                    last_off = True
                self._gen, self._cur = None, (None, None)
                time.sleep(0.1)
                next_t = time.time()
                continue
            last_off = False
            if self._cur != (eff, txt):                # (re)build generator
                niji.CFG["text"] = txt
                self._gen = niji.EFFECTS[eff](fps)
                self._cur = (eff, txt)
            try:
                frame = next(self._gen) * bri
                self.ddp.send(frame)
            except (StopIteration, OSError):
                self._gen = None
            next_t += 1.0 / fps
            time.sleep(max(0.0, next_t - time.time()))


CTRL = Controller()


# ----------------------------------------------------------------------------
# Web page
# ----------------------------------------------------------------------------
def page():
    buttons = "\n".join(
        f'<button class="fx" data-fx="{name}" onclick="pick(this)">{label}</button>'
        for name, label in UI_EFFECTS)
    return """<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Niji</title>
<style>
  :root{ color-scheme:dark; }
  *{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body{ margin:0; font-family:-apple-system,system-ui,sans-serif; background:#0c0c10;
        color:#eee; padding:16px 14px 40px; }
  h1{ font-size:20px; font-weight:600; margin:4px 0 2px; letter-spacing:.5px; }
  .sub{ color:#888; font-size:13px; margin-bottom:18px; }
  .grid{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .fx{ background:#1b1b22; color:#eee; border:1px solid #2a2a33; border-radius:14px;
       padding:18px 10px; font-size:16px; font-weight:500; cursor:pointer;
       transition:transform .05s, background .15s, border-color .15s; }
  .fx:active{ transform:scale(.96); }
  .fx.on{ background:#2563ff22; border-color:#3b82f6; box-shadow:0 0 0 1px #3b82f6 inset; }
  .row{ margin:20px 2px 0; }
  label{ font-size:13px; color:#aaa; display:block; margin-bottom:8px; }
  input[type=range]{ width:100%; height:34px; accent-color:#3b82f6; }
  .txt{ display:flex; gap:8px; margin-top:6px; }
  input[type=text]{ flex:1; background:#16161c; border:1px solid #2a2a33; color:#eee;
       border-radius:12px; padding:13px; font-size:16px; }
  .send,.off{ border:none; border-radius:12px; padding:13px 16px; font-size:15px;
       font-weight:600; cursor:pointer; }
  .send{ background:#3b82f6; color:#fff; }
  .off{ background:#33212a; color:#ff6b8a; width:100%; margin-top:22px; padding:16px; }
  .val{ float:right; color:#3b82f6; font-variant-numeric:tabular-nums; }
</style></head><body>
<h1>Niji</h1>
<div class="sub">tap an effect</div>
<div class="grid">""" + buttons + """</div>

<div class="row">
  <label>Brightness <span class="val" id="bv">40%</span></label>
  <input type="range" id="bri" min="0" max="100" value="40"
         oninput="bv.textContent=this.value+'%'" onchange="setBri(this.value)">
</div>

<div class="row">
  <label>Text message (for 🔤 Text)</label>
  <div class="txt">
    <input type="text" id="msg" value="HELLO WORLD" maxlength="120">
    <button class="send" onclick="sendText()">Send</button>
  </div>
</div>

<button class="off" onclick="turnOff()">Stop (WLED resumes)</button>

<script>
let cur=null;
function api(body){ return fetch('/api/set',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>r.json()).then(render); }
function render(s){
  cur=s.effect;
  document.querySelectorAll('.fx').forEach(b=>
    b.classList.toggle('on', b.dataset.fx===s.effect));
  bri.value=s.brightness; bv.textContent=s.brightness+'%';
  if(s.text) msg.value=s.text;
}
function pick(b){ api({effect:b.dataset.fx}); }
function setBri(v){ api({brightness:+v}); }
function sendText(){ api({effect:'text', text:msg.value}); }
function turnOff(){ api({effect:null}); }
fetch('/api/state').then(r=>r.json()).then(render);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, page(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(CTRL.snapshot()))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/api/set":
            return self._send(404, "{}")
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or "{}")
        except ValueError:
            return self._send(400, "{}")
        CTRL.update(effect=body.get("effect"),
                    has_effect=("effect" in body),
                    brightness=body.get("brightness"),
                    text=body.get("text"),
                    fps=body.get("fps"))
        self._send(200, json.dumps(CTRL.snapshot()))

    def log_message(self, *a):
        pass            # quiet


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((niji.HOST, 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    ip = lan_ip()
    print("\n  Niji control panel running:")
    print(f"    on this machine : http://localhost:{PORT}")
    print(f"    on your phone   : http://{ip}:{PORT}   (same Wi-Fi)\n")
    print("  Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
