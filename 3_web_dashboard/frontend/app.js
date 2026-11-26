const St={tab:"live",loc:true,osm:true,ptt:false,dur:0,tmr:null,ws_t:null,ws_p:null,cfg:null,tel:null,scn:"normal",til_ok:false,til_err:0};
const Map_ref={osm:null,usr:null,con:null,trl:null,hom:null,obs:null};
const Fov_arr=[];
const Obs_arr=[];
const Log_arr=[];
const Pts_arr=[];
const Vu_arr=[];
const Cls_vi={"Person":"Người","Bike":"Xe đạp","Car":"Ô tô","Motorcycle":"Xe máy","Bus":"Xe buýt","Truck":"Xe tải","Stairs":"Cầu thang","Manhole":"Nắp cống","Electrical Pole":"Cột điện","Chair":"Ghế","Bench":"Ghế dài","Traffic Cone":"Nón giao thông","Traffic sign":"Biển báo","Dustbin":"Thùng rác","Dog":"Chó","Tree":"Cây","Guard rail":"Hàng rào","Pedestrian crosswalk":"Vạch qua đường","Fire hydrant":"Trụ cứu hỏa","Teraffic Barrel":"Thùng chắn","Plant Pot":"Chậu cây","Electrical Box":"Bốt điện","Bicycle Rack":"Giá xe đạp","Building":"Tòa nhà","Road":"Lòng đường","vật cản":"Vật cản"};
const Lan_vi={"left":"Trái","center":"Giữa","right":"Phải"};
function el(id){return document.getElementById(id);}
function fmt_n(v,d){return Number.isFinite(Number(v))?Number(v).toFixed(d):"--";}
function set_tab(t){
	St.tab=t;
	document.querySelectorAll(".tab").forEach((b)=>b.classList.toggle("act",b.dataset.tab===t));
	document.querySelectorAll(".pg").forEach((p)=>p.classList.toggle("act",p.id===("p_"+t)));
	if(t==="map"){
		if(Map_ref.osm) setTimeout(()=>Map_ref.osm.invalidateSize(),80);
		if(St.tel) upd_gps(St.tel);
	}
}
function init_vu(){
	const box=el("v_vu");
	box.innerHTML="";
	Vu_arr.length=0;
	for(let i=0;i<12;++i){
		const sp=document.createElement("span");
		box.appendChild(sp);
		Vu_arr.push(sp);
	}
}
function upd_vu(lvl){
	const act=Math.round((Math.min(100,Math.max(0,lvl))/100)*Vu_arr.length);
	for(let i=0;i<Vu_arr.length;++i){
		const sp=Vu_arr[i];
		sp.className="";
		if(i<act){
			sp.style.height=(35+i*5)+"%";
			sp.className=i>9?"pk":"on";
		}else{
			sp.style.height="20%";
		}
	}
}
function con_tel(){
	const proto=location.protocol==="https:"?"wss:":"ws:";
	const ws=new WebSocket(proto+"//"+location.host+"/ws/tel");
	St.ws_t=ws;
	ws.onopen=()=>{
		el("b_con").textContent="Đã kết nối";
		el("b_con").className="pil ok";
	};
	ws.onmessage=(ev)=>{
		try{
			const pkt=JSON.parse(ev.data);
			if(pkt.type==="telemetry" && pkt.data){
				St.tel=pkt.data;
				upd_ui(pkt.data,pkt.logs||[]);
			}
		}catch(_e){}
	};
	ws.onclose=()=>{
		el("b_con").textContent="Mất kết nối";
		el("b_con").className="pil wrn";
		setTimeout(con_tel,1800);
	};
}
function upd_ui(d,logs){
	const emg=Boolean(d.emergency);
	const mod=d.mode==="edge_live"?"THIẾT BỊ THẬT":"MÔ PHỎNG";
	el("v_mod").textContent=mod;
	el("c_mdl").textContent=((d.model&&d.model.name)||"yolov26n").toUpperCase()+" "+((d.model&&d.model.imgsz)||320);
	el("c_fps").textContent=fmt_n(d.system&&d.system.fps,1)+" FPS";
	el("h_bat").textContent=fmt_n(d.system&&d.system.battery_pct,0)+"%";
	el("h_tmp").textContent=fmt_n(d.system&&d.system.cpu_temp_c,1)+"°C";
	el("k_bat").textContent=fmt_n(d.system&&d.system.battery_pct,0)+"%";
	el("k_tmp").textContent=fmt_n(d.system&&d.system.cpu_temp_c,1)+"°C";
	el("k_spd").textContent=fmt_n(d.gps&&d.gps.speed_mps,2)+" m/s";
	el("k_sat").textContent=String((d.gps&&d.gps.satellites)||0)+" vệ tinh";
	el("v_alr").classList.toggle("hid",!emg);
	el("b_emg").classList.toggle("hid",!emg && !d.geofence_breach);
	if(emg){
		el("b_con").textContent="KHẨN CẤP";
		el("b_con").className="pil emg";
		el("t_emg").textContent="CẢNH BÁO KHẨN CẤP: Vật cản ở "+fmt_n(d.ultrasonic&&d.ultrasonic.distance_m,2)+" m";
	}else if(d.geofence_breach){
		el("b_con").textContent="NGOÀI VÙNG";
		el("b_con").className="pil wrn";
		el("t_emg").textContent="CẢNH BÁO GPS: Người dùng ra khỏi vùng an toàn ("+fmt_n(d.gps&&d.gps.home_distance_m,0)+" m)";
	}else{
		el("b_con").textContent=d.mode==="edge_live"?(d.connected?"Pi Trực tuyến":"Chờ Pi 4"):"Đang mô phỏng";
		el("b_con").className="pil "+(d.mode==="edge_live"&&!d.connected?"wrn":"ok");
	}
	upd_us(d.ultrasonic||{});
	upd_obs(d.obstacles||[]);
	upd_log(logs);
	upd_gps(d);
	if(!St.ptt && d.system) upd_vu(d.system.ptt_active?d.system.ptt_level:0);
}
function upd_us(us){
	const dst=Number(us.distance_m||0);
	const emg=Number(us.emergency_distance_m||1.0);
	const pct=Math.min(100,Math.max(0,(dst/4.0)*100));
	el("h_us").textContent=fmt_n(dst,2)+" m";
	el("u_dst").textContent=fmt_n(dst,2)+" m";
	el("u_ang").textContent=fmt_n(us.beam_angle_deg,0)+"°";
	el("u_thr").textContent=fmt_n(emg,1)+"m";
	el("u_bar").style.width=pct+"%";
	el("u_mrk").style.left=Math.min(100,(emg/4.0)*100)+"%";
	if(dst<emg){
		el("u_bar").style.background="var(--dng)";
		el("u_tag").textContent="Khẩn cấp";
		el("u_tag").className="tag dng";
	}else if(dst<2.2){
		el("u_bar").style.background="var(--wrn)";
		el("u_tag").textContent="Chú ý";
		el("u_tag").className="tag wrn";
	}else{
		el("u_bar").style.background="var(--ok)";
		el("u_tag").textContent="An toàn";
		el("u_tag").className="tag ok";
	}
}
function upd_obs(lst){
	Obs_arr.length=0;
	for(let i=0;i<lst.length;++i) Obs_arr.push(lst[i]);
	el("o_cnt").textContent=String(Obs_arr.length);
	el("o_cnt2").textContent=String(Obs_arr.length);
	const htm=Obs_arr.length?Obs_arr.map((o)=>{
		const nm=Cls_vi[o.class_name]||o.class_name;
		const ln=Lan_vi[o.lane]||o.lane;
		return `<div class="obs_i ${o.is_emergency?"emg":""}"><div><strong>#${o.track_id} ${nm}</strong> · ${ln}</div><div><strong>${fmt_n(o.distance_m,2)}m</strong> · R:${fmt_n(o.risk_score,2)}</div></div>`;
	}).join(""):'<div class="mut">Chưa phát hiện vật cản gần</div>';
	el("l_obs").innerHTML=htm;
	el("l_obs2").innerHTML=htm;
}
function upd_log(lst){
	Log_arr.length=0;
	for(let i=0;i<lst.length;++i) Log_arr.push(lst[i]);
	el("l_log").innerHTML=Log_arr.length?Log_arr.map((l)=>`<div class="log_i ${l.level||"info"}"><strong>[${l.ts}] ${l.source}:</strong> ${l.message}</div>`).join(""):'<div class="mut">Chưa có sự kiện nào</div>';
}
function init_map(){
	if(typeof L==="undefined"){
		St.osm=false;
		el("m_osm").classList.add("hid");
		el("m_cvs").classList.remove("hid");
		el("b_mmd").textContent="Chế độ: Radar Offline";
		return;
	}
	try{
		const m=L.map("m_osm",{zoomControl:true}).setView([16.0738,108.1499],18);
		const lyr=L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:20,attribution:"&copy; OpenStreetMap"});
		lyr.on("tileload",()=>{St.til_ok=true;});
		lyr.on("tileerror",()=>{
			++St.til_err;
			if(!St.til_ok && St.til_err>=4 && St.osm) tog_map();
		});
		lyr.addTo(m);
		m.on("dragstart",()=>{
			if(St.loc){
				St.loc=false;
				el("b_loc").textContent="Khóa tâm: TẮT";
				el("b_loc").classList.remove("pri");
			}
		});
		Map_ref.osm=m;
		Map_ref.hom=L.circle([16.0738,108.1499],{radius:250,color:"#059669",fillColor:"#10b981",fillOpacity:0.08,weight:2}).addTo(m);
		Map_ref.trl=L.polyline([],{color:"#2563eb",weight:3,opacity:0.75}).addTo(m);
		Map_ref.con=L.polygon([],{color:"#f59e0b",fillColor:"#fbbf24",fillOpacity:0.22,weight:1}).addTo(m);
		Map_ref.usr=L.circleMarker([16.0738,108.1499],{radius:9,color:"#ffffff",weight:3,fillColor:"#1d4ed8",fillOpacity:1}).addTo(m);
		Map_ref.obs=L.layerGroup().addTo(m);
	}catch(_e){
		St.osm=false;
		el("m_osm").classList.add("hid");
		el("m_cvs").classList.remove("hid");
	}
}
function fov_pts(lat,lng,hdg,dst){
	Fov_arr.length=0;
	Fov_arr.push([lat,lng]);
	for(let a=-15;a<=15;a+=5){
		const r=(hdg+a)*Math.PI/180;
		const dx=dst*Math.sin(r);
		const dy=dst*Math.cos(r);
		Fov_arr.push([lat+dy/111111,lng+dx/(111111*Math.cos(lat*Math.PI/180))]);
	}
	return Fov_arr;
}
function upd_gps(d){
	const g=d.gps||{};
	const lat=Number(g.lat||16.0738);
	const lng=Number(g.lng||108.1499);
	const hdg=Number(g.heading_deg||0);
	el("m_crd").textContent=`GPS: ${fmt_n(lat,6)}, ${fmt_n(lng,6)} | Hướng ${fmt_n(hdg,0)}°`;
	el("g_lat").textContent=fmt_n(lat,6);
	el("g_lng").textContent=fmt_n(lng,6);
	el("g_hdg").textContent=fmt_n(hdg,0)+"°";
	el("g_spd").textContent=fmt_n(Number(g.speed_mps||0)*3.6,1)+" km/h";
	el("g_hom").textContent=fmt_n(g.home_distance_m,0)+" m";
	el("g_sat").textContent=`${g.satellites||0} / ${fmt_n(g.hdop,1)}`;
	el("g_tag").textContent=d.geofence_breach?"Ngoài vùng":"GPS OK";
	el("g_tag").className="tag "+(d.geofence_breach?"dng":"ok");
	el("m_gfs").textContent=d.geofence_breach?"CẢNH BÁO: Ra ngoài vùng an toàn":"Trong vùng an toàn ("+fmt_n(g.home_distance_m,0)+" / "+fmt_n(g.geofence_radius_m,0)+" m)";
	drw_rad(el("c_min"),d,true);
	if(St.tab==="map"){
		if(St.osm && Map_ref.osm){
			upd_osm(d,lat,lng,hdg);
		}else{
			drw_rad(el("m_cvs"),d,false);
		}
	}
}
function upd_osm(d,lat,lng,hdg){
	const g=d.gps||{};
	Map_ref.usr.setLatLng([lat,lng]);
	Map_ref.con.setLatLngs(fov_pts(lat,lng,hdg,22));
	Map_ref.trl.setLatLngs(g.trail||[]);
	Map_ref.hom.setLatLng([Number(g.home_lat||lat),Number(g.home_lng||lng)]);
	Map_ref.hom.setRadius(Number(g.geofence_radius_m||250));
	Map_ref.hom.setStyle({color:d.geofence_breach?"#dc2626":"#059669",fillColor:d.geofence_breach?"#ef4444":"#10b981"});
	Map_ref.obs.clearLayers();
	for(let i=0;i<Obs_arr.length;++i){
		const o=Obs_arr[i];
		if(o.lat && o.lng){
			L.circleMarker([o.lat,o.lng],{radius:6,color:"#fff",weight:2,fillColor:o.is_emergency?"#dc2626":"#f59e0b",fillOpacity:0.95}).bindTooltip(`${Cls_vi[o.class_name]||o.class_name} (${fmt_n(o.distance_m,2)}m)`).addTo(Map_ref.obs);
		}
	}
	if(St.loc) Map_ref.osm.panTo([lat,lng],{animate:true,duration:0.25});
}
function drw_rad(cvs,d,min_m){
	if(!cvs) return;
	const box=cvs.parentElement;
	if(!min_m && box){
		cvs.width=box.clientWidth||720;
		cvs.height=box.clientHeight||520;
	}
	const ctx=cvs.getContext("2d");
	const w=cvs.width;
	const h=cvs.height;
	const cx=w/2;
	const cy=min_m?h*0.76:h/2;
	const max_r=min_m?Math.min(w,h)*0.68:Math.min(w,h)*0.42;
	ctx.fillStyle="#0f172a";
	ctx.fillRect(0,0,w,h);
	ctx.strokeStyle="#1e293b";
	ctx.lineWidth=1;
	for(let i=1;i<=4;++i){
		ctx.beginPath();
		ctx.arc(cx,cy,(max_r*i)/4,0,Math.PI*2);
		ctx.stroke();
		ctx.fillStyle="#64748b";
		ctx.font="11px sans-serif";
		ctx.fillText(i+"m",cx+4,cy-(max_r*i)/4+12);
	}
	const hdg=((d.gps&&d.gps.heading_deg)||0)*Math.PI/180;
	const ang_c=min_m?-Math.PI/2:(hdg-Math.PI/2);
	ctx.fillStyle="rgba(251,191,36,0.18)";
	ctx.beginPath();
	ctx.moveTo(cx,cy);
	ctx.arc(cx,cy,max_r,ang_c-0.26,ang_c+0.26);
	ctx.closePath();
	ctx.fill();
	if(!min_m && d.gps && Array.isArray(d.gps.trail) && d.gps.trail.length>1){
		const lat0=Number(d.gps.lat);
		const lng0=Number(d.gps.lng);
		const scl=max_r/35;
		ctx.strokeStyle="#38bdf8";
		ctx.lineWidth=2;
		ctx.beginPath();
		for(let i=0;i<d.gps.trail.length;++i){
			const pt=d.gps.trail[i];
			const dx=(pt[1]-lng0)*111111*Math.cos(lat0*Math.PI/180)*scl;
			const dy=-(pt[0]-lat0)*111111*scl;
			if(i===0) ctx.moveTo(cx+dx,cy+dy);
			else ctx.lineTo(cx+dx,cy+dy);
		}
		ctx.stroke();
	}
	for(let i=0;i<Obs_arr.length;++i){
		const o=Obs_arr[i];
		const r=Math.min(4.0,Math.max(0.2,Number(o.distance_m||1)))*(max_r/4.0);
		let off=0;
		if(o.lane==="left") off=-0.35;
		else if(o.lane==="right") off=0.35;
		const a=ang_c+off;
		const px=cx+r*Math.cos(a);
		const py=cy+r*Math.sin(a);
		ctx.fillStyle=o.is_emergency?"#ef4444":"#f59e0b";
		ctx.beginPath();
		ctx.arc(px,py,6,0,Math.PI*2);
		ctx.fill();
		ctx.fillStyle="#f8fafc";
		ctx.font="11px sans-serif";
		ctx.fillText((Cls_vi[o.class_name]||o.class_name)+" "+fmt_n(o.distance_m,1)+"m",px+8,py+4);
	}
	ctx.fillStyle="#3b82f6";
	ctx.beginPath();
	ctx.arc(cx,cy,7,0,Math.PI*2);
	ctx.fill();
	ctx.strokeStyle="#ffffff";
	ctx.lineWidth=2;
	ctx.stroke();
}
function tog_loc(){
	St.loc=!St.loc;
	el("b_loc").textContent=St.loc?"Khóa tâm: BẬT":"Khóa tâm: TẮT";
	el("b_loc").classList.toggle("pri",St.loc);
	if(St.loc && St.tel && St.tel.gps && Map_ref.osm) Map_ref.osm.panTo([St.tel.gps.lat,St.tel.gps.lng]);
}
function tog_map(){
	if(typeof L==="undefined") return;
	St.osm=!St.osm;
	el("m_osm").classList.toggle("hid",!St.osm);
	el("m_cvs").classList.toggle("hid",St.osm);
	el("b_mmd").textContent=St.osm?"Chế độ: Đường phố":"Chế độ: Radar Cục bộ";
	if(St.osm && Map_ref.osm) setTimeout(()=>Map_ref.osm.invalidateSize(),60);
	else if(St.tel) drw_rad(el("m_cvs"),St.tel,false);
}
async function set_hom(){
	if(!St.tel || !St.tel.gps) return;
	const res=await fetch("/api/config",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({gps:{home_lat:St.tel.gps.lat,home_lng:St.tel.gps.lng}})});
	if(res.ok){
		const dat=await res.json();
		St.cfg=dat.config;
		fill_cfg(St.cfg);
		shw_msg("Đã cập nhật tâm vùng an toàn theo vị trí hiện tại",false);
	}
}
function pcm_16(flt,inp_r){
	const ratio=inp_r/16000;
	const len=Math.floor(flt.length/ratio);
	const out=new Int16Array(len);
	for(let i=0;i<len;++i){
		const s=Math.max(-1,Math.min(1,flt[Math.floor(i*ratio)]));
		out[i]=s<0?s*0x8000:s*0x7fff;
	}
	return out.buffer;
}
function con_ptt(){
	return new Promise((ok,fail)=>{
		if(St.ws_p && St.ws_p.readyState===1) return ok(St.ws_p);
		const proto=location.protocol==="https:"?"wss:":"ws:";
		const ws=new WebSocket(proto+"//"+location.host+"/ws/ptt");
		ws.onopen=()=>{
			St.ws_p=ws;
			ok(ws);
		};
		ws.onmessage=(ev)=>{
			try{
				const p=JSON.parse(ev.data);
				if(p.type==="vu") upd_vu(p.level);
				else if(p.type==="ptt_saved" && p.bytes>0) el("b_ply").disabled=false;
			}catch(_e){}
		};
		ws.onerror=()=>fail(new Error("ws"));
		ws.onclose=()=>{St.ws_p=null;};
	});
}
async function beg_ptt(){
	if(St.ptt) return;
	St.ptt=true;
	el("btn_ptt").classList.add("on");
	el("btn_pt2").classList.add("on");
	el("p_lbl").textContent="ĐANG NÓI... THẢ ĐỂ DỪNG";
	el("p_st").textContent="Đang mở micro...";
	el("p_st").className="tag wrn";
	el("p_st2").textContent="Đang nói";
	el("p_st2").className="tag dng";
	try{
		const ws=await con_ptt();
		if(!St.ptt) return;
		ws.send(JSON.stringify({type:"ptt_start"}));
		const stm=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
		if(!St.ptt){
			stm.getTracks().forEach((t)=>t.stop());
			if(ws.readyState===1) ws.send(JSON.stringify({type:"ptt_stop"}));
			return;
		}
		const Ctx=window.AudioContext||window.webkitAudioContext;
		const ctx=new Ctx();
		const src=ctx.createMediaStreamSource(stm);
		const prc=ctx.createScriptProcessor(2048,1,1);
		src.connect(prc);
		prc.connect(ctx.destination);
		prc.onaudioprocess=(e)=>{
			if(!St.ptt || !St.ws_p || St.ws_p.readyState!==1) return;
			const ch=e.inputBuffer.getChannelData(0);
			St.ws_p.send(pcm_16(ch,ctx.sampleRate));
		};
		Mic_ref.stm=stm;
		Mic_ref.ctx=ctx;
		Mic_ref.src=src;
		Mic_ref.prc=prc;
		St.dur=Date.now();
		St.tmr=setInterval(()=>{
			const s=((Date.now()-St.dur)/1000).toFixed(1);
			el("p_dur").textContent="00:"+s.padStart(4,"0");
		},100);
		el("p_st").textContent="Đang truyền tới loa";
		el("p_st").className="tag dng";
	}catch(_e){
		el("p_st").textContent="Không truy cập được micro";
		el("p_st").className="tag wrn";
		end_ptt();
	}
}
function end_ptt(){
	if(!St.ptt) return;
	St.ptt=false;
	el("btn_ptt").classList.remove("on");
	el("btn_pt2").classList.remove("on");
	el("p_lbl").textContent="NHẤN & GIỮ ĐỂ NÓI";
	el("p_st").textContent="Sẵn sàng";
	el("p_st").className="tag ok";
	el("p_st2").textContent="Sẵn sàng";
	el("p_st2").className="tag ok";
	if(St.tmr){clearInterval(St.tmr);St.tmr=null;}
	if(Mic_ref.prc){try{Mic_ref.prc.disconnect();}catch(_e){}Mic_ref.prc=null;}
	if(Mic_ref.src){try{Mic_ref.src.disconnect();}catch(_e){}Mic_ref.src=null;}
	if(Mic_ref.stm){Mic_ref.stm.getTracks().forEach((t)=>t.stop());Mic_ref.stm=null;}
	if(Mic_ref.ctx){try{Mic_ref.ctx.close();}catch(_e){}Mic_ref.ctx=null;}
	if(St.ws_p && St.ws_p.readyState===1) St.ws_p.send(JSON.stringify({type:"ptt_stop"}));
	upd_vu(0);
}
function ply_wav(){
	const a=new Audio("/api/ptt/last.wav?t="+Date.now());
	a.play().catch(()=>{});
}
async function snd_tts(txt){
	const v=(txt||el("i_tts").value).trim();
	if(!v) return;
	const res=await fetch("/api/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:v})});
	if(res.ok && !txt) el("i_tts").value="";
}
async function set_scn(s){
	await fetch("/api/scenario",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({scenario:s})});
}
async function load_cfg(){
	const res=await fetch("/api/config");
	if(!res.ok) return;
	St.cfg=await res.json();
	fill_cfg(St.cfg);
}
function fill_cfg(c){
	el("c_mnm").value=c.model.name;
	el("c_mbk").value=c.model.backend;
	el("c_mpr").value=c.model.precision;
	el("c_msz").value=String(c.model.imgsz);
	el("c_cnf").value=c.model.confidence_threshold;
	el("c_iou").value=c.model.nms_iou_threshold;
	el("c_fw").value=c.camera.frame_width;
	el("c_fh").value=c.camera.frame_height;
	el("c_ch").value=c.camera.camera_height_m;
	el("c_tl").value=c.camera.tilt_angle_deg;
	el("u_emg").value=c.ultrasonic.emergency_distance_m;
	el("u_bm").value=c.ultrasonic.beam_angle_deg;
	el("u_hz").value=c.ultrasonic.poll_rate_hz;
	el("u_mn").value=c.ultrasonic.min_range_m;
	el("u_mx").value=c.ultrasonic.max_range_m;
	el("k_q").value=c.kalman_filter.q_accel_noise;
	el("k_ru").value=c.kalman_filter.r_ultrasonic;
	el("k_rc").value=c.kalman_filter.r_camera;
	el("k_gt").value=c.kalman_filter.gating_threshold;
	el("r_wd").value=c.risk_scoring.w_distance;
	el("r_wc").value=c.risk_scoring.w_class;
	el("r_wh").value=c.risk_scoring.w_heading;
	el("r_wt").value=c.risk_scoring.w_ttc;
	el("r_tc").value=c.risk_scoring.ttc_critical_s;
	el("a_lg").value=c.audio.language;
	el("a_rt").value=c.audio.rate;
	el("a_vl").value=c.audio.speaker_volume;
	el("a_ec").value=c.audio.emergency_cooldown_s;
	el("a_tc").value=c.audio.tts_cooldown_s;
	el("a_bp").checked=Boolean(c.audio.beep_enabled);
	el("a_pr").checked=Boolean(c.audio.ptt_preempt);
	el("g_hlat").value=c.gps.home_lat;
	el("g_hlng").value=c.gps.home_lng;
	el("g_rad").value=c.gps.geofence_radius_m;
	el("s_nm").value=c.system.device_name;
	el("s_md").value=c.system.mode;
	el("e_hst").value=c.edge.edge_host;
	el("e_prt").value=c.edge.edge_port;
	el("e_mhs").value=c.edge.mqtt_host;
	el("e_mpr").value=c.edge.mqtt_port;
	el("e_mq").checked=Boolean(c.edge.mqtt_enabled);
	el("s_sy").checked=Boolean(c.system.sync_to_edge);
	el("d_scn").classList.toggle("hid",c.system.mode==="edge_live");
	upd_wsm();
}
function upd_wsm(){
	const s=Number(el("r_wd").value||0)+Number(el("r_wc").value||0)+Number(el("r_wh").value||0)+Number(el("r_wt").value||0);
	el("w_sum").textContent="Tổng: "+s.toFixed(2);
	el("w_sum").className="tag "+(Math.abs(s-1.0)<=0.05?"ok":"dng");
}
function read_cfg(){
	return {
		system:{device_name:el("s_nm").value,mode:el("s_md").value,sync_to_edge:el("s_sy").checked},
		edge:{edge_host:el("e_hst").value,edge_port:Number(el("e_prt").value),mqtt_enabled:el("e_mq").checked,mqtt_host:el("e_mhs").value,mqtt_port:Number(el("e_mpr").value)},
		model:{name:el("c_mnm").value,backend:el("c_mbk").value,precision:el("c_mpr").value,imgsz:Number(el("c_msz").value),confidence_threshold:Number(el("c_cnf").value),nms_iou_threshold:Number(el("c_iou").value)},
		camera:{frame_width:Number(el("c_fw").value),frame_height:Number(el("c_fh").value),camera_height_m:Number(el("c_ch").value),tilt_angle_deg:Number(el("c_tl").value)},
		ultrasonic:{emergency_distance_m:Number(el("u_emg").value),beam_angle_deg:Number(el("u_bm").value),poll_rate_hz:Number(el("u_hz").value),min_range_m:Number(el("u_mn").value),max_range_m:Number(el("u_mx").value)},
		kalman_filter:{q_accel_noise:Number(el("k_q").value),r_ultrasonic:Number(el("k_ru").value),r_camera:Number(el("k_rc").value),gating_threshold:Number(el("k_gt").value)},
		risk_scoring:{w_distance:Number(el("r_wd").value),w_class:Number(el("r_wc").value),w_heading:Number(el("r_wh").value),w_ttc:Number(el("r_wt").value),ttc_critical_s:Number(el("r_tc").value)},
		audio:{language:el("a_lg").value,rate:Number(el("a_rt").value),speaker_volume:Number(el("a_vl").value),emergency_cooldown_s:Number(el("a_ec").value),tts_cooldown_s:Number(el("a_tc").value),beep_enabled:el("a_bp").checked,ptt_preempt:el("a_pr").checked},
		gps:{home_lat:Number(el("g_hlat").value),home_lng:Number(el("g_hlng").value),geofence_radius_m:Number(el("g_rad").value)}
	};
}
async function save_cfg(){
	const res=await fetch("/api/config",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(read_cfg())});
	const dat=await res.json();
	if(!res.ok) return shw_msg(dat.detail||"Lỗi lưu cấu hình",true);
	St.cfg=dat.config;
	fill_cfg(St.cfg);
	shw_msg(dat.synced?"Đã lưu cấu hình và đồng bộ xuống 2_edge_device":"Đã lưu và áp dụng cấu hình mới",false);
}
async function rst_cfg(){
	const res=await fetch("/api/config/reset",{method:"POST"});
	if(!res.ok) return;
	const dat=await res.json();
	St.cfg=dat.config;
	fill_cfg(St.cfg);
	shw_msg("Đã khôi phục thông số mặc định",false);
}
async function tst_edg(){
	const res=await fetch("/api/edge/test",{method:"POST"});
	const dat=await res.json();
	shw_msg(dat.ok?("Kết nối thành công tới "+dat.url):("Chưa kết nối được tới "+dat.url),!dat.ok);
}
function shw_msg(txt,err){
	const b=el("f_msg");
	b.textContent=txt;
	b.className="tst"+(err?" err":"");
}
function snap_cam(){
	const a=document.createElement("a");
	a.href="/api/snapshot.jpg?t="+Date.now();
	a.download="blindnav_"+Date.now()+".jpg";
	a.click();
}
function full_cam(){
	const b=el("v_wrp");
	if(!document.fullscreenElement) b.requestFullscreen().catch(()=>{});
	else document.exitFullscreen().catch(()=>{});
}
async function clr_log(){
	await fetch("/api/logs",{method:"DELETE"});
	el("l_log").innerHTML='<div class="mut">Chưa có sự kiện nào</div>';
}
const Mic_ref={stm:null,ctx:null,src:null,prc:null};
function bind_ptt(b){
	b.addEventListener("pointerdown",(e)=>{e.preventDefault();beg_ptt();});
}
function init_app(){
	init_vu();
	init_map();
	load_cfg();
	con_tel();
	window.addEventListener("pointerup",end_ptt);
	window.addEventListener("pointercancel",end_ptt);
	window.addEventListener("blur",end_ptt);
	document.querySelectorAll(".tab").forEach((b)=>b.addEventListener("click",()=>set_tab(b.dataset.tab)));
	el("b_gom").addEventListener("click",()=>set_tab("map"));
	el("b_qst").addEventListener("click",()=>set_tab("live"));
	el("b_snp").addEventListener("click",snap_cam);
	el("b_ful").addEventListener("click",full_cam);
	el("b_rec").addEventListener("click",()=>{el("v_cam").src="/api/video_feed?t="+Date.now();});
	el("b_clr").addEventListener("click",clr_log);
	el("b_ply").addEventListener("click",ply_wav);
	el("b_tts").addEventListener("click",()=>snd_tts(""));
	el("i_tts").addEventListener("keydown",(e)=>{if(e.key==="Enter") snd_tts("");});
	document.querySelectorAll(".chp").forEach((c)=>c.addEventListener("click",()=>snd_tts(c.dataset.tts)));
	document.querySelectorAll("[data-scn]").forEach((b)=>b.addEventListener("click",()=>set_scn(b.dataset.scn)));
	el("b_loc").addEventListener("click",tog_loc);
	el("b_mmd").addEventListener("click",tog_map);
	el("b_hom").addEventListener("click",set_hom);
	el("b_sav").addEventListener("click",save_cfg);
	el("b_rst").addEventListener("click",rst_cfg);
	el("b_tst").addEventListener("click",tst_edg);
	["r_wd","r_wc","r_wh","r_wt"].forEach((id)=>el(id).addEventListener("input",upd_wsm));
	bind_ptt(el("btn_ptt"));
	bind_ptt(el("btn_pt2"));
	window.addEventListener("keydown",(e)=>{
		if(e.code==="Space" && !e.repeat && document.activeElement.tagName!=="INPUT" && document.activeElement.tagName!=="SELECT"){
			e.preventDefault();
			beg_ptt();
		}
	});
	window.addEventListener("keyup",(e)=>{
		if(e.code==="Space" && St.ptt){
			e.preventDefault();
			end_ptt();
		}
	});
}
window.addEventListener("DOMContentLoaded",init_app);
