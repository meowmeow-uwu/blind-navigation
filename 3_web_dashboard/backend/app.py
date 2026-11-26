import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI,HTTPException,Request,WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from backend.cfg import Cur_cfg,load_cfg,rst_cfg,save_cfg
from backend.edge import Pi_ws,add_pcm,beg_pcm,end_pcm,fwd_cmd,fwd_pcm,get_edg,get_efm,get_wav,pub_mq,pull_edg,pull_frm,push_tts,set_efm,start_mq,stop_mq,stream_frm,upd_edg
from backend.sim import get_jpg,set_ptt,set_scn,step_sim
Base_dir=Path(__file__).resolve().parents[1]
Front_dir=Base_dir/"frontend"
Tel_ws=set()
Dead_tel=set()
Logs=[]
Cur_st={}
Run_flg={"on":True,"emg":False,"gf":False,"emg_ts":0.0,"gf_ts":0.0}
def add_log(lvl:str,src:str,msg:str)->None:
	Logs.insert(0,{"id":int(time.time()*1000),"ts":time.strftime("%H:%M:%S"),"level":lvl,"source":src,"message":msg})
	while(len(Logs)>80):
		Logs.pop()
def mk_off()->bytes:
	frm=np.zeros((480,640,3),dtype=np.uint8)
	frm[:]=(22,26,36)
	cv2.rectangle(frm,(18,18),(622,462),(0,0,200),2)
	cv2.putText(frm,"CAMERA CHUA KET NOI",(135,215),cv2.FONT_HERSHEY_SIMPLEX,1.0,(90,110,255),2)
	cv2.putText(frm,"Chua nhan duoc tin hieu tu Raspberry Pi (port 8080)",(88,258),cv2.FONT_HERSHEY_SIMPLEX,0.52,(190,205,225),1)
	ok,buf=cv2.imencode(".jpg",frm,[int(cv2.IMWRITE_JPEG_QUALITY),75])
	return buf.tobytes() if ok else b""
Off_jpg=mk_off()
async def bcast_st(st:dict)->None:
	Dead_tel.clear()
	msg=json.dumps({"type":"telemetry","data":st,"logs":Logs[:25]},ensure_ascii=False)
	for ws in list(Tel_ws):
		try:
			await ws.send_text(msg)
		except Exception:
			Dead_tel.add(ws)
	for ws in Dead_tel:
		Tel_ws.discard(ws)
async def run_tel()->None:
	cnt=0
	while(Run_flg["on"]):
		cfg=load_cfg()
		mode=cfg.get("system",{}).get("mode","simulation")
		if(mode=="edge_live"):
			if(cnt%5==0): await pull_edg(cfg)
			st=get_edg(cfg)
		else:
			st=step_sim(cfg)
		Cur_st.clear()
		Cur_st.update(st)
		now=time.time()
		emg=bool(st.get("emergency"))
		if(emg and not Run_flg["emg"] and now-Run_flg["emg_ts"]>2.5):
			d=st.get("ultrasonic",{}).get("distance_m")
			d_txt=f"{float(d):.2f} m" if d is not None else "chưa rõ cự ly"
			add_log("emergency","Siêu âm / AI",f"Cảnh báo khẩn cấp! Vật cản ở khoảng cách {d_txt}")
			Run_flg["emg_ts"]=now
		Run_flg["emg"]=emg
		gf=bool(st.get("geofence_breach"))
		if(gf and not Run_flg["gf"] and now-Run_flg["gf_ts"]>5.0):
			hd=st.get("gps",{}).get("home_distance_m")
			hd_txt=f"{float(hd):.0f} m" if hd is not None else "--"
			add_log("warning","GPS",f"Người dùng đã đi ra ngoài vùng an toàn ({hd_txt})")
			Run_flg["gf_ts"]=now
		Run_flg["gf"]=gf
		if(Tel_ws): await bcast_st(st)
		cnt+=1
		await asyncio.sleep(0.1)
@asynccontextmanager
async def life(_app:FastAPI):
	cfg=load_cfg()
	Run_flg["on"]=True
	if(cfg.get("system",{}).get("mode","simulation")=="edge_live"):
		Cur_st.clear()
		Cur_st.update(get_edg(cfg))
	else:
		Cur_st.clear()
		Cur_st.update(step_sim(cfg))
	add_log("info","Hệ thống","Bảng điều khiển người thân BlindNav Care đã sẵn sàng")
	start_mq(cfg)
	tsk=asyncio.create_task(run_tel())
	try:
		yield
	finally:
		Run_flg["on"]=False
		tsk.cancel()
		stop_mq()
app=FastAPI(title="BlindNav Care Dashboard",lifespan=life)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
app.mount("/static",StaticFiles(directory=str(Front_dir)),name="static")
@app.get("/")
async def idx():
	return FileResponse(str(Front_dir/"index.html"))
@app.get("/api/health")
async def health():
	cfg=load_cfg()
	return {"ok":True,"mode":cfg.get("system",{}).get("mode","simulation"),"clients":len(Tel_ws),"edge_audio_clients":len(Pi_ws)}
@app.get("/api/state")
async def get_st():
	if(not Cur_st):
		cfg=load_cfg()
		if(cfg.get("system",{}).get("mode","simulation")=="edge_live"):
			Cur_st.update(get_edg(cfg))
		else:
			Cur_st.update(step_sim(cfg))
	return {"telemetry":dict(Cur_st),"logs":Logs[:30]}
@app.get("/api/config")
async def get_cfg():
	return load_cfg()
@app.put("/api/config")
async def put_cfg(req:Request):
	raw=await req.json()
	if(not isinstance(raw,dict)): raise HTTPException(status_code=400,detail="Invalid JSON payload")
	cfg,synced,err=save_cfg(raw)
	if(err): raise HTTPException(status_code=400,detail=err)
	pub_mq("blindnav/config",cfg)
	await fwd_cmd({"type":"config","data":cfg})
	add_log("info","Cấu hình","Đã lưu và áp dụng thông số hệ thống" + (" (đã đồng bộ 2_edge_device)" if synced else ""))
	return {"ok":True,"config":cfg,"synced":synced}
@app.post("/api/config/reset")
async def post_rst():
	cfg=rst_cfg()
	pub_mq("blindnav/config",cfg)
	await fwd_cmd({"type":"config","data":cfg})
	add_log("info","Cấu hình","Đã khôi phục thông số mặc định")
	return {"ok":True,"config":cfg}
@app.post("/api/scenario")
async def post_scn(req:Request):
	raw=await req.json()
	scn=str(raw.get("scenario","normal"))
	if(scn not in ("normal","emergency","invisible","geofence")): raise HTTPException(status_code=400,detail="Unknown scenario")
	set_scn(scn)
	add_log("info","Mô phỏng",f"Chuyển kịch bản kiểm thử: {scn}")
	return {"ok":True,"scenario":scn}
@app.post("/api/tts")
async def post_tts(req:Request):
	raw=await req.json()
	txt=str(raw.get("text","")).strip()
	if(not txt): raise HTTPException(status_code=400,detail="Empty text")
	set_ptt(False,0.0,txt)
	await fwd_cmd({"type":"tts","text":txt})
	cfg=Cur_cfg if Cur_cfg else load_cfg()
	if(not Pi_ws and cfg.get("system",{}).get("mode")=="edge_live"): await push_tts(cfg,txt)
	pub_mq("blindnav/tts",{"text":txt})
	add_log("voice","Loa thiết bị",f"Đã gửi lời nhắc thoại: \"{txt}\"")
	return {"ok":True,"text":txt}
@app.get("/api/logs")
async def get_log():
	return {"logs":list(Logs)}
@app.delete("/api/logs")
async def clr_log():
	Logs.clear()
	return {"ok":True}
@app.get("/api/ptt/last.wav")
async def get_ptt():
	wav=get_wav()
	if(not wav): raise HTTPException(status_code=404,detail="Chưa có đoạn ghi âm nào")
	return Response(content=wav,media_type="audio/wav",headers={"Cache-Control":"no-store"})
@app.post("/api/edge/telemetry")
async def post_edg(req:Request):
	raw=await req.json()
	if(not isinstance(raw,dict)): raise HTTPException(status_code=400,detail="Invalid telemetry")
	cfg=load_cfg()
	st=upd_edg(raw,cfg)
	if(cfg.get("system",{}).get("mode")=="edge_live"):
		Cur_st.clear()
		Cur_st.update(st)
	return {"ok":True}
@app.post("/api/edge/frame")
async def post_frm(req:Request):
	raw=await req.body()
	if(not raw): raise HTTPException(status_code=400,detail="Empty frame")
	set_efm(raw)
	return {"ok":True,"bytes":len(raw)}
@app.post("/api/edge/test")
async def test_edg():
	cfg=load_cfg()
	ok=await pull_edg(cfg)
	edg=cfg.get("edge",{})
	return {"ok":ok,"url":f"http://{edg.get('edge_host')}:{edg.get('edge_port')}/alert"}
def cur_jpg()->bytes:
	cfg=load_cfg()
	if(cfg.get("system",{}).get("mode")=="edge_live"):
		efm=get_efm()
		return efm if efm else Off_jpg
	jpg=get_jpg()
	return jpg if jpg else Off_jpg
@app.get("/api/snapshot.jpg")
async def snap_jpg():
	cfg=load_cfg()
	if(cfg.get("system",{}).get("mode")=="edge_live"):
		jpg=get_efm() or await pull_frm(cfg) or Off_jpg
		return Response(content=jpg,media_type="image/jpeg",headers={"Cache-Control":"no-store"})
	return Response(content=cur_jpg(),media_type="image/jpeg",headers={"Cache-Control":"no-store"})
@app.get("/api/video_feed")
async def vid_mjp():
	async def gen():
		while(True):
			cfg=Cur_cfg if Cur_cfg else load_cfg()
			if(cfg.get("system",{}).get("mode")=="edge_live"):
				jpg=get_efm()
				if(jpg):
					yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpg+b"\r\n"
					await asyncio.sleep(0.08)
					continue
				got=False
				async for frm_b in stream_frm(cfg):
					got=True
					yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frm_b+b"\r\n"
					if((Cur_cfg or cfg).get("system",{}).get("mode")!="edge_live" or get_efm()): break
				if(not got):
					yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+Off_jpg+b"\r\n"
					await asyncio.sleep(0.5)
			else:
				jpg=cur_jpg()
				if(jpg): yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpg+b"\r\n"
				await asyncio.sleep(0.08)
	return StreamingResponse(gen(),media_type="multipart/x-mixed-replace; boundary=frame",headers={"Cache-Control":"no-store, no-cache, must-revalidate"})
@app.websocket("/ws/tel")
async def ws_tel(ws:WebSocket):
	await ws.accept()
	Tel_ws.add(ws)
	try:
		if(Cur_st): await ws.send_text(json.dumps({"type":"telemetry","data":dict(Cur_st),"logs":Logs[:25]},ensure_ascii=False))
		while(True):
			msg=await ws.receive_text()
			if(msg=="ping"): await ws.send_text('{"type":"pong"}')
	except WebSocketDisconnect:
		Tel_ws.discard(ws)
	except Exception:
		Tel_ws.discard(ws)
@app.websocket("/ws/ptt")
async def ws_ptt(ws:WebSocket):
	await ws.accept()
	t_beg=0.0
	tot_b=0
	try:
		while(True):
			pkt=await ws.receive()
			if(pkt.get("type")=="websocket.disconnect"): break
			if("text" in pkt and pkt["text"] is not None):
				try:
					obj=json.loads(pkt["text"])
				except Exception:
					continue
				tp=obj.get("type")
				if(tp=="ptt_start"):
					t_beg=time.time()
					tot_b=0
					beg_pcm()
					set_ptt(True,10.0)
					await fwd_cmd({"type":"ptt_start"})
				elif(tp=="ptt_stop"):
					end_pcm()
					set_ptt(False,0.0)
					await fwd_cmd({"type":"ptt_stop"})
					dur=max(0.1,time.time()-t_beg) if t_beg>0 else 0.0
					if(tot_b>0): add_log("voice","Đàm thoại PTT",f"Người thân vừa nói qua loa ({dur:.1f}s)")
					t_beg=0.0
					await ws.send_text(json.dumps({"type":"ptt_saved","bytes":tot_b,"duration_s":round(dur,1)}))
				elif(tp=="tts" and obj.get("text")):
					txt=str(obj["text"]).strip()
					set_ptt(False,0.0,txt)
					await fwd_cmd({"type":"tts","text":txt})
					cfg=Cur_cfg if Cur_cfg else load_cfg()
					if(not Pi_ws and cfg.get("system",{}).get("mode")=="edge_live"): await push_tts(cfg,txt)
					add_log("voice","Loa thiết bị",f"Đã gửi lời nhắc thoại: \"{txt}\"")
			elif("bytes" in pkt and pkt["bytes"] is not None):
				raw=pkt["bytes"]
				tot_b+=len(raw)
				add_pcm(raw)
				arr=np.frombuffer(raw,dtype=np.int16)
				lvl=float(min(100.0,np.sqrt(np.mean(arr.astype(np.float32)**2))/220.0)) if arr.size>0 else 0.0
				set_ptt(True,lvl)
				await fwd_pcm(raw)
				await ws.send_text(json.dumps({"type":"vu","level":round(lvl,1)}))
	except WebSocketDisconnect:
		pass
	finally:
		if(t_beg>0):
			end_pcm()
			set_ptt(False,0.0)
			await fwd_cmd({"type":"ptt_stop"})
@app.websocket("/ws/edge_audio")
async def ws_edg(ws:WebSocket):
	await ws.accept()
	Pi_ws.add(ws)
	try:
		cfg=Cur_cfg if Cur_cfg else load_cfg()
		await ws.send_text(json.dumps({"type":"config","data":cfg},ensure_ascii=False))
		while(True):
			await ws.receive_text()
	except WebSocketDisconnect:
		Pi_ws.discard(ws)
	except Exception:
		Pi_ws.discard(ws)
