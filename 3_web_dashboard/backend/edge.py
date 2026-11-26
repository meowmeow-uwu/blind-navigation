import copy
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
	"device_name":"raspi",
	"emergency":False,
	"geofence_breach":False,
	"camera":{"connected":False,"error":"Camera chưa kết nối"},
	"ultrasonic":{"connected":False,"distance_m":None,"beam_angle_deg":30.0,"emergency_distance_m":1.0,"status":"disconnected","error":"Chưa kết nối cảm biến siêu âm"},
	"gps":{"connected":False,"lat":None,"lng":None,"heading_deg":None,"speed_mps":None,"satellites":0,"hdop":None,"home_lat":16.0738,"home_lng":108.1499,"geofence_radius_m":250.0,"home_distance_m":None,"trail":[],"error":"Chưa có GPS"},
	"system":{"battery_pct":None,"cpu_temp_c":None,"fps":0.0,"ptt_active":False,"ptt_level":0.0,"last_tts":""},
	"obstacles":[]
}
def upd_edg(pay:dict,cfg:dict)->dict:
	with Edg_lck:
		Edg_st["connected"]=True
		Edg_st["last_ts"]=time.time()
		for k in ("device_name","emergency","geofence_breach","camera","ultrasonic","gps","system","obstacles"):
			if(k in pay):
				if(isinstance(Edg_st.get(k),dict) and isinstance(pay[k],dict)):
					Edg_st[k].update(pay[k])
				else:
					Edg_st[k]=pay[k]
		if("ultrasonic" in pay and isinstance(pay["ultrasonic"],dict) and "connected" not in pay["ultrasonic"]):
			has_d=Edg_st["ultrasonic"].get("distance_m") is not None
			Edg_st["ultrasonic"]["connected"]=has_d
			if(has_d):
				Edg_st["ultrasonic"]["error"]=""
				d_val=float(Edg_st["ultrasonic"]["distance_m"])
				emg_d=float(cfg.get("ultrasonic",{}).get("emergency_distance_m",1.0))
				Edg_st["ultrasonic"]["status"]="emergency" if d_val<emg_d else ("caution" if d_val<2.2 else "safe")
		if("gps" in pay and isinstance(pay["gps"],dict) and "connected" not in pay["gps"]):
			has_g=Edg_st["gps"].get("lat") is not None and Edg_st["gps"].get("lng") is not None
			Edg_st["gps"]["connected"]=has_g
			if(has_g): Edg_st["gps"]["error"]=""
		if("model" in cfg): Edg_st["model"]=cfg["model"]
		return copy.deepcopy(Edg_st)
def get_edg(cfg:dict)->dict:
	with Edg_lck:
		gps_c=cfg.get("gps",{})
		us_c=cfg.get("ultrasonic",{})
		Edg_st["gps"]["home_lat"]=float(gps_c.get("home_lat",16.0738))
		Edg_st["gps"]["home_lng"]=float(gps_c.get("home_lng",108.1499))
		Edg_st["gps"]["geofence_radius_m"]=float(gps_c.get("geofence_radius_m",250.0))
		Edg_st["ultrasonic"]["beam_angle_deg"]=float(us_c.get("beam_angle_deg",30.0))
		Edg_st["ultrasonic"]["emergency_distance_m"]=float(us_c.get("emergency_distance_m",1.0))
		if(time.time()-Edg_st["last_ts"]>6.0):
			Edg_st["connected"]=False
			Edg_st["emergency"]=False
			Edg_st["geofence_breach"]=False
			Edg_st["camera"]={"connected":False,"error":"Camera chưa kết nối (mất kết nối Raspberry Pi)"}
			Edg_st["ultrasonic"]["connected"]=False
			Edg_st["ultrasonic"]["distance_m"]=None
			Edg_st["ultrasonic"]["status"]="disconnected"
			Edg_st["ultrasonic"]["error"]="Chưa kết nối cảm biến siêu âm"
			Edg_st["gps"]["connected"]=False
			Edg_st["gps"]["lat"]=None
			Edg_st["gps"]["lng"]=None
			Edg_st["gps"]["heading_deg"]=None
			Edg_st["gps"]["speed_mps"]=None
			Edg_st["gps"]["satellites"]=0
			Edg_st["gps"]["hdop"]=None
			Edg_st["gps"]["home_distance_m"]=None
			Edg_st["gps"]["error"]="Chưa có GPS"
			Edg_st["system"]["battery_pct"]=None
			Edg_st["system"]["cpu_temp_c"]=None
			Edg_st["system"]["fps"]=0.0
			Edg_st["obstacles"]=[]
		Edg_st["model"]=cfg.get("model",{})
		Edg_st["device_name"]=cfg.get("system",{}).get("device_name",Edg_st.get("device_name","raspi"))
		return copy.deepcopy(Edg_st)
def set_efm(jpg:bytes)->None:
	global Edg_jpg,Edg_ts
	with Edg_lck:
		Edg_jpg=jpg
		Edg_ts=time.time()
		Edg_st["last_ts"]=Edg_ts
		Edg_st["connected"]=True
		Edg_st["camera"]={"connected":True,"error":""}
def get_efm()->bytes:
	with Edg_lck:
		if(time.time()-Edg_ts>4.0): return b""
		return Edg_jpg
async def pull_frm(cfg:dict)->bytes:
	edg=cfg.get("edge",{})
	url=f"http://{edg.get('edge_host','172.20.10.3')}:{int(edg.get('edge_port',8080))}/video_feed"
	try:
		async with httpx.AsyncClient(timeout=2.0) as cli:
			async with cli.stream("GET",url) as res:
				if(res.status_code!=200): return b""
				buf=bytearray()
				async for chnk in res.aiter_bytes():
					buf.extend(chnk)
					s=buf.find(b"\xff\xd8")
					if(s==-1):
						if(len(buf)>2): del buf[:-2]
						continue
					e=buf.find(b"\xff\xd9",s)
					if(e!=-1): return bytes(buf[s:e+2])
					if(len(buf)>512000): break
	except Exception:
		return b""
	return b""
async def stream_frm(cfg:dict):
	edg=cfg.get("edge",{})
	url=f"http://{edg.get('edge_host','172.20.10.3')}:{int(edg.get('edge_port',8080))}/video_feed"
	try:
		async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0,read=4.0,write=2.0,pool=2.0)) as cli:
			async with cli.stream("GET",url) as res:
				if(res.status_code!=200): return
				buf=bytearray()
				async for chnk in res.aiter_bytes():
					buf.extend(chnk)
					while(True):
						s=buf.find(b"\xff\xd8")
						if(s==-1):
							if(len(buf)>2): del buf[:-2]
							break
						e=buf.find(b"\xff\xd9",s)
						if(e==-1): break
						jpg=bytes(buf[s:e+2])
						del buf[:e+2]
						yield jpg
					if(len(buf)>512000): buf.clear()
	except Exception:
		return
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
	host=edg.get("edge_host","172.20.10.3")
	port=int(edg.get("edge_port",8080))
	base=f"http://{host}:{port}"
	try:
		async with httpx.AsyncClient(timeout=1.8) as cli:
			r_tel=await cli.get(f"{base}/telemetry")
			if(r_tel.status_code==200):
				pay_tel=r_tel.json()
				if(isinstance(pay_tel,dict) and ("ultrasonic" in pay_tel or "camera" in pay_tel or "gps" in pay_tel)):
					upd_edg(pay_tel,cfg)
					return True
			r=await cli.get(f"{base}/alert")
			if(r.status_code==200):
				pay=r.json()
				with Edg_lck:
					Edg_st["connected"]=True
					Edg_st["last_ts"]=time.time()
					Edg_st["emergency"]=pay.get("mode")=="emergency"
					if("distance" in pay and pay["distance"] is not None):
						d=float(pay["distance"])
						Edg_st["ultrasonic"]["connected"]=True
						Edg_st["ultrasonic"]["distance_m"]=d
						Edg_st["ultrasonic"]["error"]=""
						Edg_st["ultrasonic"]["status"]="emergency" if d<float(cfg.get("ultrasonic",{}).get("emergency_distance_m",1.0)) else ("caution" if d<2.2 else "safe")
					else:
						Edg_st["ultrasonic"]["connected"]=False
						Edg_st["ultrasonic"]["distance_m"]=None
						Edg_st["ultrasonic"]["status"]="disconnected"
						Edg_st["ultrasonic"]["error"]="Chưa có dữ liệu cảm biến siêu âm"
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
async def push_tts(cfg:dict,txt:str)->bool:
	edg=cfg.get("edge",{})
	url=f"http://{edg.get('edge_host','172.20.10.3')}:{int(edg.get('edge_port',8080))}/tts"
	try:
		async with httpx.AsyncClient(timeout=1.5) as cli:
			r=await cli.post(url,json={"text":txt})
			return r.status_code==200
	except Exception:
		return False
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
