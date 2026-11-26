import copy
from pathlib import Path
import yaml
Base_dir=Path(__file__).resolve().parents[1]
Repo_dir=Base_dir.parent
Cfg_fil=Base_dir/"config"/"dashboard_config.yaml"
Edge_cfg=Repo_dir/"2_edge_device"/"configs"/"config.yaml"
Dep_cfg=Repo_dir/"2_edge_device"/"deployment_config.yaml"
Def_cfg={
	"system":{"device_name":"blind-nav-pi4","log_level":"INFO","mode":"simulation","sync_to_edge":False},
	"edge":{"edge_host":"192.168.1.50","edge_port":8080,"mqtt_enabled":False,"mqtt_host":"localhost","mqtt_port":1883},
	"model":{"name":"yolov26n","backend":"onnx","precision":"int8","imgsz":320,"confidence_threshold":0.25,"nms_iou_threshold":0.45},
	"camera":{"frame_width":640,"frame_height":480,"fps":30,"f_x":480.0,"f_y":480.0,"camera_height_m":1.45,"tilt_angle_deg":5.0},
	"ultrasonic":{"trig_pin":23,"echo_pin":24,"beam_angle_deg":30.0,"min_range_m":0.03,"max_range_m":4.0,"poll_rate_hz":40,"emergency_distance_m":1.0},
	"kalman_filter":{"q_accel_noise":0.5,"r_ultrasonic":0.0025,"r_camera":0.09,"gating_threshold":4.0},
	"risk_scoring":{"w_distance":0.40,"w_class":0.30,"w_heading":0.20,"w_ttc":0.10,"ttc_critical_s":2.0},
	"audio":{"language":"vi","rate":200,"speaker_volume":85,"emergency_cooldown_s":0.6,"tts_cooldown_s":2.0,"beep_enabled":True,"ptt_preempt":True},
	"gps":{"home_lat":16.0738,"home_lng":108.1499,"geofence_radius_m":250.0,"trail_limit":120}
}
Cur_cfg={}
Tmp_edg={}
Tmp_dep={}
def mrg_dct(dst:dict,src:dict)->None:
	for k,v in src.items():
		if(isinstance(v,dict) and isinstance(dst.get(k),dict)):
			mrg_dct(dst[k],v)
		else:
			dst[k]=copy.deepcopy(v)
def load_cfg()->dict:
	Cur_cfg.clear()
	Cur_cfg.update(copy.deepcopy(Def_cfg))
	if(Edge_cfg.is_file()):
		with Edge_cfg.open("r",encoding="utf-8") as f:
			raw=yaml.safe_load(f) or {}
		if(isinstance(raw,dict)):
			for sec in ("system","camera","ultrasonic","kalman_filter","risk_scoring","audio"):
				if(isinstance(raw.get(sec),dict) and isinstance(Cur_cfg.get(sec),dict)):
					for k,v in raw[sec].items():
						if(k in Cur_cfg[sec]): Cur_cfg[sec][k]=v
	if(Cfg_fil.is_file()):
		with Cfg_fil.open("r",encoding="utf-8") as f:
			raw=yaml.safe_load(f) or {}
		if(isinstance(raw,dict)): mrg_dct(Cur_cfg,raw)
	return copy.deepcopy(Cur_cfg)
def val_cfg(cfg:dict)->str:
	mdl=cfg.get("model",{})
	if(mdl.get("name") not in ("yolov8n","yolov9t","yolov12n","yolov26n")): return "Invalid model name"
	if(mdl.get("backend") not in ("onnx","ncnn")): return "Invalid backend"
	if(mdl.get("precision") not in ("fp32","int8")): return "Invalid precision"
	if(int(mdl.get("imgsz",320)) not in (320,640)): return "imgsz must be 320 or 640"
	if(not (0.01<=float(mdl.get("confidence_threshold",0.25))<=1.0)): return "confidence_threshold out of range"
	if(not (0.05<=float(mdl.get("nms_iou_threshold",0.45))<=1.0)): return "nms_iou_threshold out of range"
	us=cfg.get("ultrasonic",{})
	if(not (0.2<=float(us.get("emergency_distance_m",1.0))<=3.5)): return "emergency_distance_m must be between 0.2 and 3.5"
	if(not (5.0<=float(us.get("beam_angle_deg",30.0))<=90.0)): return "beam_angle_deg out of range"
	rsk=cfg.get("risk_scoring",{})
	w_sum=float(rsk.get("w_distance",0))+float(rsk.get("w_class",0))+float(rsk.get("w_heading",0))+float(rsk.get("w_ttc",0))
	if(w_sum<=0.0 or abs(w_sum-1.0)>0.05): return "Risk weights must sum to 1.0"
	aud=cfg.get("audio",{})
	if(not (0<=int(aud.get("speaker_volume",85))<=100)): return "speaker_volume must be 0..100"
	gps=cfg.get("gps",{})
	if(float(gps.get("geofence_radius_m",250.0))<20.0): return "geofence_radius_m must be >= 20"
	return ""
def sync_edg(cfg:dict)->bool:
	synced=False
	if(Edge_cfg.is_file()):
		with Edge_cfg.open("r",encoding="utf-8") as f:
			raw=yaml.safe_load(f) or {}
		Tmp_edg.clear()
		if(isinstance(raw,dict)): Tmp_edg.update(raw)
		for sec in ("camera","ultrasonic","kalman_filter","risk_scoring","audio"):
			if(isinstance(cfg.get(sec),dict)):
				if(not isinstance(Tmp_edg.get(sec),dict)): Tmp_edg[sec]={}
				for k,v in cfg[sec].items():
					if(k in Tmp_edg[sec] or sec in ("camera","ultrasonic","kalman_filter","risk_scoring","audio")): Tmp_edg[sec][k]=v
		if("system" in Tmp_edg and isinstance(Tmp_edg["system"],dict)):
			Tmp_edg["system"]["device_name"]=cfg["system"]["device_name"]
			Tmp_edg["system"]["log_level"]=cfg["system"]["log_level"]
		with Edge_cfg.open("w",encoding="utf-8") as f:
			yaml.safe_dump(Tmp_edg,f,allow_unicode=True,sort_keys=False)
		synced=True
	if(Dep_cfg.is_file()):
		with Dep_cfg.open("r",encoding="utf-8") as f:
			raw=yaml.safe_load(f) or {}
		Tmp_dep.clear()
		if(isinstance(raw,dict)): Tmp_dep.update(raw)
		Tmp_dep["default_model"]=cfg["model"]["name"]
		Tmp_dep["default_backend"]=cfg["model"]["backend"]
		Tmp_dep["default_imgsz"]=int(cfg["model"]["imgsz"])
		if(not isinstance(Tmp_dep.get("inference"),dict)): Tmp_dep["inference"]={}
		Tmp_dep["inference"]["confidence_threshold"]=float(cfg["model"]["confidence_threshold"])
		Tmp_dep["inference"]["nms_iou_threshold"]=float(cfg["model"]["nms_iou_threshold"])
		with Dep_cfg.open("w",encoding="utf-8") as f:
			yaml.safe_dump(Tmp_dep,f,allow_unicode=True,sort_keys=False)
		synced=True
	return synced
def save_cfg(patch:dict)->tuple[dict,bool,str]:
	if(not Cur_cfg): load_cfg()
	nxt=copy.deepcopy(Cur_cfg)
	mrg_dct(nxt,patch)
	err=val_cfg(nxt)
	if(err): return copy.deepcopy(Cur_cfg),False,err
	Cur_cfg.clear()
	Cur_cfg.update(nxt)
	Cfg_fil.parent.mkdir(parents=True,exist_ok=True)
	with Cfg_fil.open("w",encoding="utf-8") as f:
		yaml.safe_dump(Cur_cfg,f,allow_unicode=True,sort_keys=False)
	synced=False
	if(Cur_cfg.get("system",{}).get("sync_to_edge")): synced=sync_edg(Cur_cfg)
	return copy.deepcopy(Cur_cfg),synced,""
def rst_cfg()->dict:
	Cur_cfg.clear()
	Cur_cfg.update(copy.deepcopy(Def_cfg))
	Cfg_fil.parent.mkdir(parents=True,exist_ok=True)
	with Cfg_fil.open("w",encoding="utf-8") as f:
		yaml.safe_dump(Cur_cfg,f,allow_unicode=True,sort_keys=False)
	return copy.deepcopy(Cur_cfg)
