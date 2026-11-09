"""Local web UI: upload a video -> convert to NetHack layer view -> download.

Run:
  runtime\\python\\python.exe server.py
then open http://localhost:8765 in your browser.
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run import process_video

try:
    import cv2
    import numpy as np
    import library as libmod
    from face import FaceEngine
    _LIB_OK = True
except Exception:
    _LIB_OK = False


def _face_engine():
    return FaceEngine(
        os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx"),
        os.path.join(HERE, "models", "face_recognition_sface_2021dec.onnx"))

PORT = 8765
tasks = {}            # task_id -> dict(state)
_lock = threading.Lock()

# output goes next to the exe (frozen) or next to this file (dev)
if getattr(sys, "frozen", False):
    OUTPUT_DIR = os.path.join(os.path.dirname(sys.executable), "output")
else:
    OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>NetHack Layer View 转换</title>
<style>
  body{font-family:system-ui,Segoe UI,sans-serif;background:#0b1220;color:#d6e2ee;
       max-width:760px;margin:32px auto;padding:0 16px}
  h1{font-size:20px;color:#7fb6ff}
  .tabs{display:flex;gap:8px;margin-top:14px}
  .tabs button{background:#121d33;color:#9fb4d0;border:1px solid #243656;
       padding:8px 18px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
  .tabs button.active{background:#1a2a47;color:#7fb6ff;border-bottom-color:#1a2a47}
  .card{background:#121d33;border:1px solid #243656;border-radius:0 10px 10px 10px;
       padding:24px;margin-top:-1px}
  label{display:block;margin:12px 0 6px;color:#9fb4d0;font-size:13px}
  input[type=file],select,input[type=text]{width:100%;padding:8px;background:#0b1220;
       color:#d6e2ee;border:1px solid #2f4470;border-radius:6px;box-sizing:border-box}
  button.primary{margin-top:18px;padding:10px 22px;background:#2a6df4;color:#fff;
       border:0;border-radius:6px;font-size:15px;cursor:pointer}
  button.primary:disabled{background:#3a4660;cursor:not-allowed}
  button.mini{padding:4px 10px;background:#2a6df4;color:#fff;border:0;
       border-radius:5px;font-size:12px;cursor:pointer;margin-left:6px}
  button.mini.gray{background:#3a4660}
  .bar{height:18px;background:#0b1220;border:1px solid #2f4470;border-radius:9px;
       overflow:hidden;margin-top:14px}
  .bar>div{height:100%;width:0;background:linear-gradient(90deg,#2a6df4,#7fb6ff);transition:width .3s}
  #status{margin-top:10px;font-size:13px;color:#9fb4d0;min-height:18px}
  a{color:#7fb6ff}
  .row{display:flex;gap:16px} .row>div{flex:1}
  h3{font-size:15px;color:#7fb6ff;margin:20px 0 8px}
  .item{display:flex;align-items:center;gap:10px;padding:8px;background:#0b1220;
       border:1px solid #243656;border-radius:8px;margin-top:8px;font-size:13px}
  .item img{width:56px;height:56px;object-fit:cover;border-radius:6px;background:#000}
  .item .grow{flex:1}
  .chk{display:inline-flex;align-items:center;gap:6px;margin:6px 14px 0 0;
       font-size:13px;color:#d6e2ee}
  .hidden{display:none}
</style>
</head>
<body>
<h1>NetHack Layer View</h1>
<div class="tabs">
  <button id="tab1" class="active" onclick="showTab(1)">视频转换</button>
  <button id="tab2" onclick="showTab(2)">人脸库 / 设备库 / 异常记录</button>
</div>

<div class="card" id="page1">
  <label>选择视频文件</label>
  <input type="file" id="f" accept="video/*">
  <div class="row">
    <div>
      <label>视角</label>
      <select id="profile"><option value="game" selected>game（蓝红图层）</option>
      <option value="neon">neon（霓虹）</option></select>
    </div>
    <div>
      <label>输出模式</label>
      <select id="mode"><option value="compare" selected>左右对比</option>
      <option value="single">仅图层</option></select>
    </div>
  </div>
  <button class="primary" id="go" disabled>开始转换</button>
  <div class="bar"><div id="fill"></div></div>
  <div id="status">就绪</div>
  <button class="primary" id="openbtn" style="display:none;margin-top:10px">打开输出文件夹</button>
</div>

<div class="card hidden" id="page2">
  <h3>人脸库（录入内部/外部人员）</h3>
  <div id="facelist"></div>
  <div class="row">
    <div><label>姓名</label><input type="text" id="fname" placeholder="张三"></div>
    <div><label>身份</label>
      <select id="frole"><option value="内部" selected>内部（合规·蓝）</option>
      <option value="外部">外部（异常·红）</option></select></div>
  </div>
  <label>正脸照片</label>
  <input type="file" id="fphoto" accept="image/*">
  <button class="primary" onclick="enroll()">录入人脸</button>

  <h3>电子设备白名单（勾选=合规标蓝，其余标红并记异常）</h3>
  <div id="devlist"></div>
  <button class="primary" onclick="saveDevs()">保存白名单</button>

  <h3>异常记录（系统自动捕捉的未知人员/设备）</h3>
  <div id="anomlist"></div>
  <div id="libmsg" style="margin-top:10px;font-size:13px;color:#9fb4d0"></div>
</div>

<script>
const f=document.getElementById('f'), go=document.getElementById('go');
const openbtn=document.getElementById('openbtn'), status=document.getElementById('status');
function showTab(n){
  document.getElementById('page1').className='card'+(n===1?'':' hidden');
  document.getElementById('page2').className='card'+(n===2?'':' hidden');
  document.getElementById('tab1').className=n===1?'active':'';
  document.getElementById('tab2').className=n===2?'active':'';
  if(n===2) loadLib();
}
f.onchange=()=>go.disabled=!f.files.length;
go.onclick=async()=>{
  go.disabled=true; openbtn.style.display='none';
  const fd=new FormData();
  fd.append('file', f.files[0]);
  fd.append('profile', document.getElementById('profile').value);
  fd.append('compare', document.getElementById('mode').value==='compare'?'1':'0');
  const r=await fetch('/convert',{method:'POST',body:fd});
  const j=await r.json();
  if(j.error){status.textContent='错误: '+j.error;go.disabled=false;return;}
  poll(j.task);
};
function poll(tid){
  const iv=setInterval(async()=>{
    const r=await fetch('/progress?task='+tid);
    const j=await r.json();
    document.getElementById('fill').style.width=j.pct+'%';
    status.textContent=j.status;
    if(j.status==='done'){
      clearInterval(iv);
      go.disabled=false;
      status.textContent='完成！已保存到: '+(j.out||'');
      openbtn.style.display='inline-block';
      openbtn.onclick=()=>{
        if(window.pywebview&&window.pywebview.api) window.pywebview.api.open_folder(j.out);
        else status.textContent='请手动打开: '+(j.out||'');
      };
    } else if(j.status.startsWith('error')){
      clearInterval(iv); go.disabled=false;
    }
  }, 500);
}

// ---------- library management ----------
const DEV_TYPES=[[62,'显示器'],[63,'笔记本'],[64,'鼠标'],[65,'遥控器'],[66,'键盘'],
                 [67,'手机'],[68,'微波炉'],[69,'烤箱'],[70,'烤面包机'],[72,'冰箱']];
async function loadLib(){
  const j=await (await fetch('/library')).json();
  const fl=document.getElementById('facelist');
  fl.innerHTML=j.faces.length?'':'<div style="color:#5a7a9a;font-size:13px">（空）</div>';
  for(const p of j.faces){
    const d=document.createElement('div');d.className='item';
    d.innerHTML='<img src="/libimg?f='+p.image+'"><div class="grow">'+
      p.name+' · '+p.role+'</div>'+
      '<button class="mini gray" onclick="delFace('+p.id+')">删除</button>';
    fl.appendChild(d);
  }
  const dl=document.getElementById('devlist');
  dl.innerHTML='';
  for(const [id,nm] of DEV_TYPES){
    const c=document.createElement('label');c.className='chk';
    c.innerHTML='<input type="checkbox" value="'+id+'"'+
      (j.devices.allowed.includes(id)?' checked':'')+'> '+nm;
    dl.appendChild(c);
  }
  const al=document.getElementById('anomlist');
  al.innerHTML=j.anomalies.length?'':'<div style="color:#5a7a9a;font-size:13px">（无异常记录）</div>';
  for(const a of j.anomalies){
    const d=document.createElement('div');d.className='item';
    let btns='';
    if(a.status!=='pending') btns='<span style="color:#5a7a9a">'+a.status+'</span>';
    else if(a.kind==='face') btns=
      '<button class="mini" onclick="anomFace('+a.id+',\'内部\')">标为内部</button>'+
      '<button class="mini" onclick="anomFace('+a.id+',\'外部\')">标为外部</button>'+
      '<button class="mini gray" onclick="anomAct('+a.id+',\'ignore\')">忽略</button>';
    else btns=
      '<button class="mini" onclick="anomAct('+a.id+',\'device_ok\')">加入白名单</button>'+
      '<button class="mini gray" onclick="anomAct('+a.id+',\'ignore\')">忽略</button>';
    d.innerHTML='<img src="/libimg?f='+a.image+'"><div class="grow">'+
      a.label+'<br><span style="color:#5a7a9a">'+a.ts+'</span></div><div>'+btns+'</div>';
    al.appendChild(d);
  }
}
async function enroll(){
  const name=document.getElementById('fname').value.trim();
  const photo=document.getElementById('fphoto').files[0];
  if(!name||!photo){libmsg('请填写姓名并选择照片');return;}
  const fd=new FormData();
  fd.append('name',name);fd.append('role',document.getElementById('frole').value);
  fd.append('photo',photo);
  const j=await (await fetch('/enroll_face',{method:'POST',body:fd})).json();
  libmsg(j.error||'已录入: '+name); loadLib();
}
async function delFace(id){await fetch('/face_delete',{method:'POST',body:JSON.stringify({id})});loadLib();}
async function saveDevs(){
  const ids=[...document.querySelectorAll('#devlist input:checked')].map(c=>+c.value);
  const j=await (await fetch('/device_allow',{method:'POST',body:JSON.stringify({allowed:ids})})).json();
  libmsg(j.ok?'白名单已保存':'保存失败');
}
async function anomFace(id,role){
  const name=prompt('该人员的姓名：'); if(!name) return;
  const j=await (await fetch('/anomaly_action',{method:'POST',
    body:JSON.stringify({id,action:role==='内部'?'face_in':'face_out',name})})).json();
  libmsg(j.error||j.ok||''); loadLib();
}
async function anomAct(id,action){
  await fetch('/anomaly_action',{method:'POST',body:JSON.stringify({id,action})});
  loadLib();
}
function libmsg(m){document.getElementById('libmsg').textContent=m||'';}
</script>
</body>
</html>
"""


def _parse_multipart(body, ctype):
    """minimal multipart/form-data parse -> dict of name->(filename, bytes)."""
    import re
    m = re.search(r"boundary=(.+)$", ctype)
    if not m:
        return {}
    boundary = ("--" + m.group(1).strip().strip('"')).encode()
    parts = body.split(boundary)
    result = {}
    for part in parts:
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not head:
            continue
        head_s = head.decode("utf-8", "replace")
        name_m = re.search(r'name="([^"]+)"', head_s)
        fname_m = re.search(r'filename="([^"]*)"', head_s)
        if not name_m:
            continue
        name = name_m.group(1)
        data = data.rstrip(b"\r\n")
        result[name] = (fname_m.group(1) if fname_m else None, data)
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/progress":
            tid = self.path.split("task=", 1)[-1]
            with _lock:
                t = tasks.get(tid, {})
            self._send_json(t)
        elif path == "/library":
            lib = libmod.load()
            self._send_json({
                "faces": [{"id": f.get("id"), "name": f["name"],
                           "role": f.get("role", "内部"),
                           "image": f.get("image", "")} for f in lib["faces"]],
                "devices": lib["devices"],
                "anomalies": libmod.list_anomalies(),
            })
        elif path == "/libimg":
            fname = self.path.split("f=", 1)[-1]
            fname = os.path.basename(fname)     # no path traversal
            fp = os.path.join(libmod.IMG_DIR, fname)
            if not os.path.exists(fp):
                self._send_json({"error": "not found"}, 404)
                return
            with open(fp, "rb") as fimg:
                data = fimg.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/download":
            tid = self.path.split("task=", 1)[-1]
            with _lock:
                t = tasks.get(tid, {})
            out = t.get("out")
            if not out or not os.path.exists(out):
                self._send_json({"error": "file not found"}, 404)
                return
            with open(out, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{os.path.basename(out)}"')
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_enroll(self, body, ctype):
        fields = _parse_multipart(body, ctype)
        name = (fields.get("name") or (b"", None))[1]
        role = (fields.get("role") or ("内部", None))[1]
        photo = fields.get("photo")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        if isinstance(role, bytes):
            role = role.decode("utf-8", "ignore")
        if not name or not photo or not photo[1]:
            self._send_json({"error": "缺少姓名或照片"})
            return
        arr = np.frombuffer(photo[1], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self._send_json({"error": "照片无法解析"})
            return
        eng = _face_engine()
        if not eng.ok:
            self._send_json({"error": "人脸模型缺失"})
            return
        aligned = eng.find_face(img)
        if aligned is None:
            self._send_json({"error": "照片中未检测到正脸，请换一张"})
            return
        emb = eng.embed(aligned)
        libmod.add_face(name, role, emb, aligned)
        self._send_json({"ok": True})

    def _handle_anomaly(self, body):
        try:
            j = json.loads(body)
        except Exception:
            self._send_json({"error": "bad json"}, 400)
            return
        aid, action = j.get("id"), j.get("action")
        if action == "ignore":
            libmod.set_anomaly_status(aid, "已忽略")
            self._send_json({"ok": True})
            return
        if action == "device_ok":
            # add this device class to the whitelist
            cls = None
            for e in libmod.list_anomalies():
                if e.get("id") == aid:
                    cls = e.get("cls")
            if cls is None:
                self._send_json({"error": "记录缺少类别"})
                return
            allow = libmod.allowed_devices()
            allow.add(int(cls))
            libmod.set_allowed_devices(sorted(allow))
            libmod.set_anomaly_status(aid, "已加白名单")
            self._send_json({"ok": True})
            return
        if action in ("face_in", "face_out"):
            name = (j.get("name") or "").strip()
            img_path = libmod.get_anomaly_image(aid)
            if not name or not img_path or not os.path.exists(img_path):
                self._send_json({"error": "缺少姓名或图片"})
                return
            img = cv2.imread(img_path)
            eng = _face_engine()
            aligned = eng.find_face(img) if eng.ok else None
            # stored anomaly crops are already aligned faces; if detection
            # fails on the small crop, embed it directly
            if aligned is None:
                aligned = cv2.resize(img, (112, 112))
            emb = eng.embed(aligned)
            libmod.add_face(name, "内部" if action == "face_in" else "外部",
                            emb, aligned)
            libmod.set_anomaly_status(aid, f"已录入:{name}")
            self._send_json({"ok": f"已录入 {name}"})
            return
        self._send_json({"error": "unknown action"}, 400)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")

        if self.path.startswith("/enroll_face"):
            self._handle_enroll(body, ctype)
            return
        if self.path.startswith("/face_delete"):
            try:
                libmod.delete_face(json.loads(body).get("id"))
            except Exception:
                pass
            self._send_json({"ok": True})
            return
        if self.path.startswith("/device_allow"):
            try:
                libmod.set_allowed_devices(json.loads(body).get("allowed", []))
            except Exception:
                pass
            self._send_json({"ok": True})
            return
        if self.path.startswith("/anomaly_action"):
            self._handle_anomaly(body)
            return
        if self.path != "/convert":
            self._send_json({"error": "not found"}, 404)
            return
        fields = _parse_multipart(body, ctype)
        file_info = fields.get("file")
        if not file_info or not file_info[1]:
            self._send_json({"error": "no file"})
            return
        fname, data = file_info
        profile = (fields.get("profile") or (b"game", None))[1]
        if isinstance(profile, bytes):
            profile = profile.decode()
        compare = (fields.get("compare") or (b"1", None))[1]
        if isinstance(compare, bytes):
            compare = compare.decode() == "1"

        task_id = str(int(time.time() * 1000))
        tmp = tempfile.mkdtemp(prefix="nethack_")
        base = os.path.splitext(os.path.basename(fname) or "video")[0]
        src = os.path.join(tmp, os.path.basename(fname) or "input.mp4")
        out = os.path.join(OUTPUT_DIR, f"{base}_nethack.mp4")
        with open(src, "wb") as f:
            f.write(data)
        with _lock:
            tasks[task_id] = {"pct": 0, "status": "queued", "out": out}

        def work():
            def prog(frame, total):
                pct = int(100 * frame / max(1, total))
                with _lock:
                    tasks[task_id]["pct"] = pct
                    tasks[task_id]["status"] = f"处理中 {frame}/{total} ({pct}%)"

            def log(msg):
                with _lock:
                    tasks[task_id]["status"] = msg

            try:
                n = process_video(src, out=out, compare=compare,
                                  profile=profile, progress=prog, log=log)
                with _lock:
                    tasks[task_id]["pct"] = 100
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["frames"] = n
            except Exception as e:
                with _lock:
                    tasks[task_id]["status"] = f"error: {e}"

        threading.Thread(target=work, daemon=True).start()
        self._send_json({"task": task_id})


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"NetHack converter running at http://localhost:{PORT}")
    print("(press Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
