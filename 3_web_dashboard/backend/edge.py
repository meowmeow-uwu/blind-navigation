import io
import json
import time
import wave
import asyncio
import threading
import httpx
Edg_lck=threading.Lock()
Pi_ws=set()
Dead_ws=set()
Pcm_buf=bytearray()
Wav_buf=b""
Edg_jpg=b""
Edg_ts=0.0
Mqtt_cli=None
Edg_st={
	"connected":False,
	"last_ts":0.0,
	"mode":"edge_live",
	"device_name":"blind-nav-pi4",
	"emergency":False,
	"geofence_breach":False,
	"ultrasonic":{"distance_m":2.5,"beam_angle_deg":30.0,"emergency_distance_m":1.0,"status":"safe"},
	"gps":{"lat":16.0738,"lng":108.1499,"heading_deg":0.0,"speed_mps":0.0,"satellites":0,"hdop":9.9,"home_lat":16.0738,"home_lng":108.1499,"geofence_radius_m":250.0,"home_distance_m":0.0,"trail":[]},
	"system":{"battery_pct":100.0,"cpu_temp_c":45.0,"fps":0.0,"ptt_active":False,"ptt_level":0.0,"last_tts":""},
	"obstacles":[]
}
def upd_edg(pay:dict,cfg:dict)->dict:
	with Edg_lck:
		Edg_st["connected"]=True
		Edg_st["last_ts"]=time.time()
		for k in ("device_name","emergency","geofence_breach","ultrasonic","gps","system","obstacles"):
			if(k in pay):
				if(isinstance(Edg_st.get(k),dict) and isinstance(pay[k],dict)):
					Edg_st[k].update(pay[k])
				else:
					Edg_st[k]=pay[k]
		if("model" in cfg): Edg_st["model"]=cfg["model"]
		return dict(Edg_st)
def get_edg(cfg:dict)->dict:
	with Edg_lck:
		if(time.time()-Edg_st["last_ts"]>6.0): Edg_st["connected"]=False
		Edg_st["model"]=cfg.get("model",{})
		return dict(Edg_st)
def set_efm(jpg:bytes)->None:
	global Edg_jpg,Edg_ts
	with Edg_lck:
		Edg_jpg=jpg
		Edg_ts=time.time()
		Edg_st["last_ts"]=Edg_ts
		Edg_st["connected"]=True
def get_efm()->bytes:
	with Edg_lck:
		if(time.time()-Edg_ts>4.0): return b""
		return Edg_jpg
async def pull_frm(cfg:dict)->bytes:
	edg=cfg.get("edge",{})
	url=f"http://{edg.get('edge_host','192.168.1.50')}:{int(edg.get('edge_port',8080))}/video_feed"
	try:
		async with httpx.AsyncClient(timeout=2.0) as cli:
			async with cli.stream("GET",url) as res:
				if(res.status_code!=200): return b""
				buf=bytearray()
				async for chnk in res.aiter_bytes():
					buf.extend(chnk)
					s=buf.find(b"\xff\xd8")
					e=buf.find(b"\xff\xd9")
					if(s!=-1 and e>s): return bytes(buf[s:e+2])
					if(len(buf)>512000): break
	except Exception:
		return b""
	return b""
def beg_pcm()->None:
	with Edg_lck:
		Pcm_buf.clear()
def add_pcm(chnk:bytes)->None:
	with Edg_lck:
		Pcm_buf.extend(chnk)
		if(len(Pcm_buf)>32000*60): del Pcm_buf[:len(Pcm_buf)-32000*60]
def end_pcm()->bytes:
	global Wav_buf
	with Edg_lck:
		if(not Pcm_buf): return Wav_buf
		bio=io.BytesIO()
		with wave.open(bio,"wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)
			wf.setframerate(16000)
			wf.writeframes(bytes(Pcm_buf))
		Wav_buf=bio.getvalue()
		return Wav_buf
def get_wav()->bytes:
	with Edg_lck:
		return Wav_buf
async def pull_edg(cfg:dict)->bool:
	edg=cfg.get("edge",{})
	host=edg.get("edge_host","192.168.1.50")
	port=int(edg.get("edge_port",8080))
	base=f"http://{host}:{port}"
	try:
		async with httpx.AsyncClient(timeout=1.8) as cli:
			r=await cli.get(f"{base}/alert")
			if(r.status_code==200):
				pay=r.json()
				with Edg_lck:
					Edg_st["connected"]=True
					Edg_st["last_ts"]=time.time()
					Edg_st["emergency"]=pay.get("mode")=="emergency"
					if("distance" in pay):
						d=float(pay["distance"])
						Edg_st["ultrasonic"]["distance_m"]=d
						Edg_st["ultrasonic"]["status"]="emergency" if d<float(cfg.get("ultrasonic",{}).get("emergency_distance_m",1.0)) else "safe"
				return True
	except Exception:
		return False
	return False
async def fwd_pcm(chnk:bytes)->None:
	Dead_ws.clear()
	for ws in list(Pi_ws):
		try:
			await ws.send_bytes(chnk)
		except Exception:
			Dead_ws.add(ws)
	for ws in Dead_ws:
		Pi_ws.discard(ws)
async def fwd_cmd(cmd:dict)->None:
	Dead_ws.clear()
	msg=json.dumps(cmd,ensure_ascii=False)
	for ws in list(Pi_ws):
		try:
			await ws.send_text(msg)
		except Exception:
			Dead_ws.add(ws)
	for ws in Dead_ws:
		Pi_ws.discard(ws)
def start_mq(cfg:dict)->None:
	global Mqtt_cli
	edg=cfg.get("edge",{})
	if(not edg.get("mqtt_enabled")): return
	try:
		import paho.mqtt.client as mqtt
		cli=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
		def on_msg(_c,_u,msg)->None:
			try:
				pay=json.loads(msg.payload.decode("utf-8"))
				if(isinstance(pay,dict)): upd_edg(pay,cfg)
			except Exception:
				pass
		cli.on_message=on_msg
		cli.connect(str(edg.get("mqtt_host","localhost")),int(edg.get("mqtt_port",1883)),30)
		cli.subscribe("blindnav/telemetry")
		cli.loop_start()
		Mqtt_cli=cli
	except Exception:
		Mqtt_cli=None
def stop_mq()->None:
	global Mqtt_cli
	if(Mqtt_cli is not None):
		try:
			Mqtt_cli.loop_stop()
			Mqtt_cli.disconnect()
		except Exception:
			pass
		Mqtt_cli=None
def pub_mq(tpc:str,pay:dict)->None:
	if(Mqtt_cli is not None):
		try:
			Mqtt_cli.publish(tpc,json.dumps(pay,ensure_ascii=False))
		except Exception:
			pass
