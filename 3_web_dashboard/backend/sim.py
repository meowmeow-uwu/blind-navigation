import math
import time
import threading
import cv2
import numpy as np
Sim_lck=threading.Lock()
Sim_st={
	"t0":time.time(),
	"scn":"normal",
	"lat":16.0738,
	"lng":108.1499,
	"hdg":42.0,
	"spd":1.15,
	"us_d":1.85,
	"bat":86.0,
	"tmp":54.2,
	"fps":14.8,
	"sat":11,
	"hdop":0.9,
	"ptt_on":False,
	"ptt_lvl":0.0,
	"last_tts":"",
	"jpg":b""
}
Sim_obs=[]
Pts_buf=[]
Txt_map={"Person":"NGUOI","Chair":"GHE","Stairs":"CAU THANG","Car":"O TO","Bike":"XE DAP","Electrical Pole":"COT DIEN","Traffic Cone":"NON GT","vật cản":"VAT CAN"}
Col_map={"Person":(60,210,80),"Chair":(0,200,255),"Stairs":(0,110,255),"Car":(0,60,235),"Bike":(220,180,40),"Electrical Pole":(40,180,255),"vật cản":(200,70,220)}
def geo_dst(lat1:float,lng1:float,lat2:float,lng2:float)->float:
	r=6371000.0
	p1=math.radians(lat1)
	p2=math.radians(lat2)
	dp=math.radians(lat2-lat1)
	dl=math.radians(lng2-lng1)
	a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
	return 2*r*math.atan2(math.sqrt(a),math.sqrt(max(1e-12,1-a)))
def geo_off(lat:float,lng:float,hdg:float,dst:float,lane:str)->tuple[float,float]:
	ang=hdg
	if(lane=="left"): ang-=18.0
	elif(lane=="right"): ang+=18.0
	rad=math.radians(ang)
	dx=dst*math.sin(rad)
	dy=dst*math.cos(rad)
	n_lat=lat+(dy/111111.0)
	n_lng=lng+(dx/(111111.0*max(0.2,math.cos(math.radians(lat)))))
	return round(n_lat,6),round(n_lng,6)
def set_scn(scn:str)->None:
	with Sim_lck:
		Sim_st["scn"]=scn
		Sim_st["t0"]=time.time()
def set_ptt(on:bool,lvl:float=0.0,tts:str="")->None:
	with Sim_lck:
		Sim_st["ptt_on"]=on
		Sim_st["ptt_lvl"]=lvl
		if(tts): Sim_st["last_tts"]=tts
def make_obs(tid:int,cls:str,conf:float,bx:tuple[int,int,int,int],dst:float,lane:str,rsk:float,ttc:float,us_t:bool,emg:bool,lat:float,lng:float)->dict:
	return {
		"track_id":tid,
		"class_name":cls,
		"confidence":round(conf,2),
		"bbox":list(bx),
		"distance_m":round(dst,2),
		"lane":lane,
		"risk_score":round(rsk,3),
		"ttc_seconds":round(ttc,1),
		"is_ultrasonic_target":us_t,
		"is_emergency":emg,
		"lat":lat,
		"lng":lng
	}
def step_sim(cfg:dict)->dict:
	now=time.time()
	with Sim_lck:
		t=now-Sim_st["t0"]
		scn=Sim_st["scn"]
		gps_c=cfg.get("gps",{})
		us_c=cfg.get("ultrasonic",{})
		rsk_c=cfg.get("risk_scoring",{})
		h_lat=float(gps_c.get("home_lat",16.0738))
		h_lng=float(gps_c.get("home_lng",108.1499))
		rad_m=float(gps_c.get("geofence_radius_m",250.0))
		lim=int(gps_c.get("trail_limit",120))
		emg_d=float(us_c.get("emergency_distance_m",1.0))
		if(scn=="geofence"):
			ang=t*0.15
			off_m=rad_m+38.0+12.0*math.sin(t*0.4)
			Sim_st["lat"]=h_lat+(off_m*math.cos(ang))/111111.0
			Sim_st["lng"]=h_lng+(off_m*math.sin(ang))/(111111.0*math.cos(math.radians(h_lat)))
			Sim_st["hdg"]=(math.degrees(ang)+90.0)%360.0
			Sim_st["spd"]=1.28
		else:
			ang=t*0.12
			r_w=38.0
			Sim_st["lat"]=h_lat+(r_w*math.sin(ang))/111111.0
			Sim_st["lng"]=h_lng+(r_w*math.sin(ang*2.0)*0.65)/(111111.0*math.cos(math.radians(h_lat)))
			Sim_st["hdg"]=(45.0+math.sin(ang)*55.0)%360.0
			Sim_st["spd"]=1.05+0.25*math.sin(t*0.7)
		u_lat=round(Sim_st["lat"],6)
		u_lng=round(Sim_st["lng"],6)
		u_hdg=round(Sim_st["hdg"],1)
		if(not Pts_buf or geo_dst(Pts_buf[-1][0],Pts_buf[-1][1],u_lat,u_lng)>=1.5):
			Pts_buf.append([u_lat,u_lng])
		while(len(Pts_buf)>lim):
			Pts_buf.pop(0)
		Sim_obs.clear()
		if(scn=="emergency"):
			us_d=round(max(0.35,0.62+0.12*math.sin(t*2.4)),2)
			s=max(0.65,min(1.45,1.25/us_d))
			bw=int(95*s)
			bh=int(210*s)
			cx=320+int(18*math.sin(t*1.5))
			bx=(max(20,cx-bw//2),max(55,390-bh),min(620,cx+bw//2),430)
			o_lat,o_lng=geo_off(u_lat,u_lng,u_hdg,us_d,"center")
			Sim_obs.append(make_obs(101,"Person",0.91,bx,us_d,"center",0.94,max(0.4,us_d/1.1),True,us_d<emg_d,o_lat,o_lng))
			o2_lat,o2_lng=geo_off(u_lat,u_lng,u_hdg,2.2,"right")
			Sim_obs.append(make_obs(104,"Electrical Pole",0.84,(475,90,545,425),2.2,"right",0.58,4.1,False,False,o2_lat,o2_lng))
		elif(scn=="invisible"):
			us_d=round(0.68+0.08*math.sin(t*2.0),2)
			o_lat,o_lng=geo_off(u_lat,u_lng,u_hdg,us_d,"center")
			Sim_obs.append(make_obs(-1,"vật cản",1.0,(210,165,430,440),us_d,"center",0.95,1.3,True,us_d<emg_d,o_lat,o_lng))
		else:
			cyc=t%12.0
			d1=round(max(0.75,3.4-cyc*0.22),2)
			s1=max(0.45,min(1.3,1.35/d1))
			bw=int(88*s1)
			bh=int(190*s1)
			cx=325+int(24*math.sin(t*0.9))
			bx1=(max(40,cx-bw//2),max(60,390-bh),min(600,cx+bw//2),410)
			lane1="left" if cx<213 else ("right" if cx>426 else "center")
			w_d=float(rsk_c.get("w_distance",0.4))
			w_c=float(rsk_c.get("w_class",0.3))
			w_h=float(rsk_c.get("w_heading",0.2))
			w_t=float(rsk_c.get("w_ttc",0.1))
			r1=min(1.0,w_d*math.exp(-1.1*d1)+w_c*0.70+w_h*(0.95 if lane1=="center" else 0.45)+w_t*math.exp(-d1/2.0))
			o1_lat,o1_lng=geo_off(u_lat,u_lng,u_hdg,d1,lane1)
			Sim_obs.append(make_obs(12,"Person",0.89,bx1,d1,lane1,r1,max(0.5,d1/0.8),lane1=="center",d1<emg_d,o1_lat,o1_lng))
			d2=round(2.4+0.35*math.cos(t*0.6),2)
			o2_lat,o2_lng=geo_off(u_lat,u_lng,u_hdg,d2,"left")
			Sim_obs.append(make_obs(15,"Stairs",0.82,(58,255,188,430),d2,"left",0.68,5.2,False,False,o2_lat,o2_lng))
			us_d=d1 if lane1=="center" else round(2.85+0.25*math.sin(t*0.8),2)
		Sim_st["us_d"]=us_d
		Sim_st["bat"]=round(max(12.0,86.0-(t/120.0)),1)
		Sim_st["tmp"]=round(53.5+2.8*math.sin(t*0.15),1)
		Sim_st["fps"]=round(14.5+0.9*math.sin(t*1.1),1)
		is_emg=(us_d<emg_d) or any(o["is_emergency"] for o in Sim_obs)
		hm_dst=round(geo_dst(u_lat,u_lng,h_lat,h_lng),1)
		gf_out=hm_dst>rad_m
		frm=gen_frm(Sim_st,Sim_obs,is_emg,t,cfg)
		ok,buf=cv2.imencode(".jpg",frm,[int(cv2.IMWRITE_JPEG_QUALITY),72])
		if(ok): Sim_st["jpg"]=buf.tobytes()
		return {
			"ts":now,
			"mode":"simulation",
			"scenario":scn,
			"device_name":cfg.get("system",{}).get("device_name","blind-nav-pi4"),
			"model":cfg.get("model",{}),
			"emergency":is_emg,
			"geofence_breach":gf_out,
			"ultrasonic":{
				"distance_m":us_d,
				"beam_angle_deg":float(us_c.get("beam_angle_deg",30.0)),
				"emergency_distance_m":emg_d,
				"status":"emergency" if us_d<emg_d else ("caution" if us_d<2.2 else "safe")
			},
			"gps":{
				"lat":u_lat,
				"lng":u_lng,
				"heading_deg":u_hdg,
				"speed_mps":round(Sim_st["spd"],2),
				"satellites":Sim_st["sat"],
				"hdop":Sim_st["hdop"],
				"home_lat":h_lat,
				"home_lng":h_lng,
				"geofence_radius_m":rad_m,
				"home_distance_m":hm_dst,
				"trail":[list(p) for p in Pts_buf]
			},
			"system":{
				"battery_pct":Sim_st["bat"],
				"cpu_temp_c":Sim_st["tmp"],
				"fps":Sim_st["fps"],
				"ptt_active":Sim_st["ptt_on"],
				"ptt_level":round(Sim_st["ptt_lvl"],1),
				"last_tts":Sim_st["last_tts"]
			},
			"obstacles":[dict(o) for o in Sim_obs]
		}
def gen_frm(st:dict,obs:list,is_emg:bool,t:float,cfg:dict)->np.ndarray:
	w,h=640,480
	frm=np.zeros((h,w,3),dtype=np.uint8)
	frm[0:225,:]=(185,172,158)
	frm[225:,:]=(52,58,64)
	cv2.line(frm,(0,225),(w,225),(120,126,132),2)
	cv2.line(frm,(320,225),(35,h),(90,98,106),2)
	cv2.line(frm,(320,225),(605,h),(90,98,106),2)
	off=int((t*42)%36)
	for i in range(6):
		y=235+((i*40+off)%240)
		cv2.line(frm,(316,y),(324,y),(210,210,190),2)
	t3=w//3
	cv2.line(frm,(t3,34),(t3,h),(80,88,96),1)
	cv2.line(frm,(2*t3,34),(2*t3,h),(80,88,96),1)
	cv2.putText(frm,"TRAI",(t3//2-22,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,185,190),1)
	cv2.putText(frm,"GIUA",(w//2-22,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,240,255),1)
	cv2.putText(frm,"PHAI",(2*t3+t3//2-22,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,185,190),1)
	bw=int(t3*0.88)
	pts=np.array([[w//2,34],[w//2-bw,h],[w//2+bw,h]],np.int32)
	ov=frm.copy()
	cv2.fillPoly(ov,[pts],(235,210,30))
	cv2.addWeighted(ov,0.12,frm,0.88,0,frm)
	for idx,o in enumerate(obs):
		x1,y1,x2,y2=o["bbox"]
		if(o["track_id"]!=-1):
			cv2.rectangle(frm,(x1+6,y1+6),(x2-6,y2-6),(75,85,95),-1)
			if(o["class_name"]=="Person"):
				cx=(x1+x2)//2
				r=max(10,(x2-x1)//4)
				cv2.circle(frm,(cx,y1+r+8),r,(205,175,145),-1)
		col=(0,45,235) if o["is_emergency"] else Col_map.get(o["class_name"],(0,220,120))
		cv2.rectangle(frm,(x1,y1),(x2,y2),col,3 if idx==0 else 2)
		lbl=Txt_map.get(o["class_name"],o["class_name"].upper())
		tag=f"#{o['track_id']} {lbl} | {o['distance_m']:.2f}m | R:{o['risk_score']:.2f}"
		if(o["is_ultrasonic_target"]): tag+=" [*US]"
		sz=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)[0]
		cv2.rectangle(frm,(x1,max(34,y1-22)),(min(w-2,x1+sz[0]+8),max(34,y1)),col,-1)
		cv2.putText(frm,tag,(x1+4,max(48,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,255,255),1)
	hdr=(0,30,190) if is_emg else (20,28,38)
	cv2.rectangle(frm,(0,0),(w,34),hdr,-1)
	mdl=cfg.get("model",{}).get("name","yolov26n").upper()
	txt=f"BLINDNAV | {mdl} | FPS:{st['fps']:.1f} | US:{st['us_d']:.2f}m | BAT:{st['bat']:.0f}%"
	if(is_emg): txt+=" | KHAN CAP!"
	cv2.putText(frm,txt,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
	if(st["ptt_on"]):
		cv2.rectangle(frm,(140,h-48),(500,h-14),(30,120,235),-1)
		cv2.putText(frm,"LOA: NGUOI THAN DANG NOI...",(162,h-26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
	return frm
def get_jpg()->bytes:
	with Sim_lck:
		return Sim_st["jpg"]
