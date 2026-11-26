import sys
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
Web_dir=Path(__file__).resolve().parents[1]
if(str(Web_dir) not in sys.path): sys.path.insert(0,str(Web_dir))
from backend.app import app
from backend.cfg import rst_cfg
def test_idx_and_hlt():
	with TestClient(app) as cli:
		r1=cli.get("/")
		assert r1.status_code==200
		assert "BlindNav Care" in r1.text
		r2=cli.get("/api/health")
		assert r2.status_code==200
		assert r2.json()["ok"] is True
def test_cfg_api():
	with TestClient(app) as cli:
		try:
			r1=cli.get("/api/config")
			assert r1.status_code==200
			cfg=r1.json()
			assert "ultrasonic" in cfg and "gps" in cfg
			r2=cli.put("/api/config",json={"ultrasonic":{"emergency_distance_m":1.25},"gps":{"geofence_radius_m":300.0}})
			assert r2.status_code==200
			assert r2.json()["config"]["ultrasonic"]["emergency_distance_m"]==1.25
			assert r2.json()["config"]["gps"]["geofence_radius_m"]==300.0
			r3=cli.put("/api/config",json={"risk_scoring":{"w_distance":0.9,"w_class":0.9,"w_heading":0.1,"w_ttc":0.1}})
			assert r3.status_code==400
		finally:
			rst_cfg()
def test_sim_and_snp():
	with TestClient(app) as cli:
		r1=cli.post("/api/scenario",json={"scenario":"emergency"})
		assert r1.status_code==200
		r2=cli.get("/api/snapshot.jpg")
		assert r2.status_code==200
		assert r2.headers["content-type"]=="image/jpeg"
		assert r2.content[:2]==b"\xff\xd8"
		r3=cli.get("/api/state")
		assert r3.status_code==200
		dat=r3.json()["telemetry"]
		assert "gps" in dat and "ultrasonic" in dat and "obstacles" in dat
		cli.post("/api/scenario",json={"scenario":"normal"})
def test_tts_and_ptt():
	with TestClient(app) as cli:
		r1=cli.post("/api/tts",json={"text":"Dừng lại có xe"})
		assert r1.status_code==200
		pcm=(np.sin(np.linspace(0,20,1600))*8000).astype(np.int16).tobytes()
		with cli.websocket_connect("/ws/ptt") as ws:
			ws.send_text('{"type":"ptt_start"}')
			ws.send_bytes(pcm)
			vu=ws.receive_json()
			assert vu["type"]=="vu"
			ws.send_text('{"type":"ptt_stop"}')
			svd=ws.receive_json()
			assert svd["type"]=="ptt_saved"
			assert svd["bytes"]==len(pcm)
		r2=cli.get("/api/ptt/last.wav")
		assert r2.status_code==200
		assert r2.content[:4]==b"RIFF"
def test_edg_ing():
	with TestClient(app) as cli:
		r1=cli.post("/api/edge/telemetry",json={"device_name":"pi-test","emergency":True,"ultrasonic":{"distance_m":0.55}})
		assert r1.status_code==200
		r2=cli.post("/api/edge/frame",content=b"\xff\xd8\xff\xe0test\xff\xd9",headers={"content-type":"image/jpeg"})
		assert r2.status_code==200
def test_ws_tel():
	with TestClient(app) as cli:
		with cli.websocket_connect("/ws/tel") as ws:
			msg=ws.receive_json()
			assert msg["type"]=="telemetry"
			assert "gps" in msg["data"]
def test_edg_missing_hw_errors():
	from backend.cfg import load_cfg
	from backend.edge import Edg_st,get_edg,upd_edg
	cfg=load_cfg()
	Edg_st["last_ts"]=0.0
	st=get_edg(cfg)
	assert st["connected"] is False
	assert st["camera"]["connected"] is False
	assert "chưa kết nối" in st["camera"]["error"].lower()
	assert st["ultrasonic"]["connected"] is False
	assert st["ultrasonic"]["distance_m"] is None
	assert st["gps"]["connected"] is False
	assert st["gps"]["lat"] is None and st["gps"]["lng"] is None
	assert "gps" in st["gps"]["error"].lower()
	assert st["system"]["battery_pct"] is None
	st2=upd_edg({"camera":{"connected":True,"error":""},"ultrasonic":{"connected":False,"distance_m":None,"error":"Chưa kết nối cảm biến siêu âm"}},cfg)
	assert st2["connected"] is True
	assert st2["camera"]["connected"] is True
	assert st2["ultrasonic"]["distance_m"] is None
