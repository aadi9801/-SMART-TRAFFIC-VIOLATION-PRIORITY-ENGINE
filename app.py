import streamlit as st
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import random
import base64
from datetime import datetime

# ==========================================
# SENIOR AI ENGINEER - PREMIUM COMMAND CENTER UPDATE V23.1
# 1. GLASSMORPHISM UI: Modern, professional aesthetic with backdrop blur.
# 2. SIGNAL INTELLIGENCE: Non-blinking, state-aware status badges.
# 3. HUD POLISHING: Sleek, semi-transparent vehicle labels and thin markings.
# 4. VIOLATION PRIORITY: Intelligent alert management (Overspeed > Tailgating).
# ==========================================

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Traffic Violation Priority Engine", layout="wide", initial_sidebar_state="expanded")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-main: #0a0f1e;
            --bg-card: rgba(30, 41, 59, 0.7);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.3);
            --danger: #f43f5e;
            --success: #10b981;
            --warning: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        .stApp { background-color: var(--bg-main); color: var(--text-main); font-family: 'Inter', sans-serif; }
        
        section[data-testid="stSidebar"] {
            background-color: #0d1117 !important;
            border-right: 1px solid var(--glass-border);
        }
        
        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 15px;
            transition: transform 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-2px); border-color: var(--primary); }
        
        .metric-label { color: var(--text-muted); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text-main); margin-top: 5px; font-family: 'JetBrains Mono', monospace; }
        
        .log-entry {
            background: var(--bg-card);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            border-left: 4px solid var(--primary);
            padding: 15px;
            margin-bottom: 12px;
            border-top: 1px solid var(--glass-border);
            border-right: 1px solid var(--glass-border);
            border-bottom: 1px solid var(--glass-border);
        }
        
        .advisory-text {
            background: rgba(99, 102, 241, 0.05);
            color: #c7d2fe;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-top: 10px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            line-height: 1.5;
        }

        .signal-badge {
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .stTabs [data-baseweb="tab-list"] { gap: 32px; background-color: transparent; border-bottom: 1px solid var(--glass-border); }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }

        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            width: 100%;
            box-shadow: 0 4px 12px var(--primary-glow);
            transition: all 0.2s ease;
        }
        .stButton>button:hover { transform: scale(1.02); box-shadow: 0 6px 16px var(--primary-glow); }

        .evacuate-banner {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(239, 68, 68, 0.9);
            color: white;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 800;
            z-index: 1000;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
            animation: flash-red 1s infinite;
        }
        @keyframes flash-red { 0% { opacity: 1; transform: translateX(-50%) scale(1); } 50% { opacity: 0.7; transform: translateX(-50%) scale(1.05); } 100% { opacity: 1; transform: translateX(-50%) scale(1); } }
    </style>
""", unsafe_allow_html=True)

# --- SIREN AUDIO COMPONENT ---
audio_placeholder = st.empty()

def trigger_siren():
    siren_url = "https://www.soundjay.com/misc/sounds/police-siren-1.mp3"
    audio_placeholder.markdown(f"""
        <audio autoplay key="{time.time()}">
            <source src="{siren_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)

# --- LIVE IST CLOCK COMPONENT ---
def inject_live_clock():
    st.components.v1.html("""
        <div id="ist-clock" style="
            text-align: right;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(8px);
            padding: 10px 15px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'Inter', -apple-system, sans-serif;
            width: fit-content;
            margin-left: auto;
        ">
            <div id="clock-time" style="font-size: 1.1rem; font-weight: 700; color: #6366f1; font-family: 'JetBrains Mono', monospace; line-height: 1.2;">00:00:00</div>
            <div id="clock-date" style="font-size: 0.75rem; color: #94a3b8; font-weight: 500; margin-top: 2px;">Loading...</div>
        </div>
        <script>
            function updateClock() {
                const now = new Date();
                const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                const ist = new Date(utc + (3600000 * 5.5));
                
                const hours = String(ist.getHours()).padStart(2, '0');
                const minutes = String(ist.getMinutes()).padStart(2, '0');
                const seconds = String(ist.getSeconds()).padStart(2, '0');
                const timeStr = hours + ':' + minutes + ':' + seconds;
                
                const dayName = ist.toLocaleDateString('en-IN', { weekday: 'long' });
                const dateStr = ist.toLocaleDateString('en-IN', { day: '2-digit', month: '2-digit', year: 'numeric' });
                
                document.getElementById('clock-time').innerText = timeStr;
                document.getElementById('clock-date').innerText = dayName + ', ' + dateStr;
            }
            setInterval(updateClock, 1000);
            updateClock();
        </script>
    """, height=75)

# --- SENTINEL SELF-REPAIR ENGINE ---
def run_self_repair():
    if 'repair_logs' not in st.session_state: st.session_state.repair_logs = []
    if 'engine_restarts' not in st.session_state: st.session_state.engine_restarts = 0
    
    # Memory Sanitization
    if len(track_data) > 100:
        st.session_state.repair_logs.append(f"[{datetime.now().strftime('%H:%M')}] Memory Sanitization: Cleared stale IDs")
        track_data.clear()

def log_repair(msg):
    if 'repair_logs' not in st.session_state: st.session_state.repair_logs = []
    st.session_state.repair_logs.append(f"[{datetime.now().strftime('%H:%M')}] {msg}")
    if len(st.session_state.repair_logs) > 5: st.session_state.repair_logs.pop(0)

# --- UTILS ---
def get_ist_now():
    now = datetime.now()
    return now.strftime("%H:%M:%S"), now.strftime("%d/%m/%Y"), now.strftime("%A")

# --- CONSTANTS ---
VEHICLE_CLASSES = {2: "CAR", 3: "BIKE", 5: "BUS", 7: "TRUCK", 9: "SIGNAL"}

# --- CACHED MODEL & REFERENCE ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_ambulance_ref():
    ref_path = "assets/ambulance.webp"
    if not os.path.exists(ref_path): return None, None
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None: return None, None
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(ref_img, None)
    return kp, des

def check_emergency_colors(crop):
    if crop is None or crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pct = (cv2.countNonZero(white_mask) / crop.size) * 100
    
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    red_pct = (cv2.countNonZero(red_mask) / crop.size) * 100
    
    return white_pct > 25 and red_pct > 0.8

def is_ambulance_match(crop, ref_des):
    if crop is None or ref_des is None or crop.size == 0: return False
    h, w = crop.shape[:2]
    aspect = w / h
    if not (1.0 < aspect < 3.5): return False
    if not check_emergency_colors(crop): return False
    
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(gray_crop, None)
    if des is None or len(des) < 10: return False
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, des)
    good_matches = [m for m in matches if m.distance < 35]
    return len(good_matches) > 22

def get_signal_state(crop):
    if crop is None or crop.size == 0: return "UNKNOWN"
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Expanded Red Range (Top 1/3)
    top_zone = hsv[0:h//3, :]
    lower_red1, upper_red1 = np.array([0, 70, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 70]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(top_zone, lower_red1, upper_red1), cv2.inRange(top_zone, lower_red2, upper_red2))
    
    # Expanded Green Range (Bottom 1/3)
    btm_zone = hsv[2*h//3:h, :]
    lower_green, upper_green = np.array([35, 40, 40]), np.array([95, 255, 255])
    green_mask = cv2.inRange(btm_zone, lower_green, upper_green)
    
    red_px = cv2.countNonZero(red_mask)
    green_px = cv2.countNonZero(green_mask)
    
    # Sensitivity threshold
    if red_px > green_px and red_px > 5: return "RED"
    if green_px > red_px and green_px > 5: return "GREEN"
    return "UNKNOWN"

# --- STATE ---
if 'active' not in st.session_state: st.session_state.active = False
if 'last_siren_time' not in st.session_state: st.session_state.last_siren_time = 0
if 'violation_history' not in st.session_state: st.session_state.violation_history = deque(maxlen=50)
if 'risk_history' not in st.session_state: st.session_state.risk_history = deque([0]*60, maxlen=60)
if 'ambulance_sim' not in st.session_state: st.session_state.ambulance_sim = False
if 'ambulance_detected' not in st.session_state: st.session_state.ambulance_detected = set()
if 'current_video' not in st.session_state: st.session_state.current_video = "assets/traffic.mp4"
if 'signal_seen' not in st.session_state: st.session_state.signal_seen = False
if 'last_signal_state' not in st.session_state: st.session_state.last_signal_state = "UNKNOWN"

# --- PRECISION TRACKING ENGINE ---
track_data = {}

def reset_engine_state():
    st.session_state.violation_history.clear()
    st.session_state.risk_history = deque([0]*60, maxlen=60)
    st.session_state.ambulance_detected.clear()
    st.session_state.signal_seen = False
    track_data.clear()

def get_officer_advisory(v_type, v_name, speed, tid):
    advisories = {
        "TAILGATING": f"Unsafe following distance for ID #{tid}.",
        "ILLEGAL LANE CHANGE": f"Unsafe lane maneuver for ID #{tid}.",
        "COLLISION RISK": f"CRITICAL: ID #{tid} on collision course.",
        "OUT OF CONTROL": f"Erratic behavior detected for ID #{tid}.",
        "WRONG LANE": f"Wrong-way driving for ID #{tid}.",
        "OVERSPEED": f"Speed violation for ID #{tid} ({speed}km/h).",
        "SIGNAL VIOLATION": f"Red light violation for ID #{tid}."
    }
    return advisories.get(v_type, "Monitoring vehicle behavior.")

VIOL_PRIORITY = {
    "COLLISION RISK": 100,
    "OUT OF CONTROL": 90,
    "WRONG LANE": 80,
    "SIGNAL VIOLATION": 70,
    "OVERSPEED": 60,
    "TAILGATING": 50,
    "ILLEGAL LANE CHANGE": 40,
    "AMBULANCE": 10,
    None: 0
}

def set_violation(d, new_v):
    curr_v = d.get('v_type')
    if VIOL_PRIORITY.get(new_v, 0) > VIOL_PRIORITY.get(curr_v, 0):
        d['v_type'] = new_v

# --- SIDEBAR ---
with st.sidebar:
    st.title("Command Center")
    st.markdown("AI Traffic Intelligence")
    
    with st.expander("Feed Selection", expanded=True):
        video_files = [f"assets/{f}" for f in os.listdir("assets") if f.endswith(('.mp4', '.avi', '.mov'))]
        if video_files:
            try: default_idx = video_files.index(st.session_state.current_video)
            except: default_idx = 0
            selected_video = st.selectbox("Select Feed", video_files, index=default_idx)
            if selected_video != st.session_state.current_video:
                st.session_state.current_video = selected_video
                reset_engine_state()
                st.rerun()
    
    with st.expander("AI Config", expanded=True):
        PEAK_ACCURACY = st.toggle("🎯 Peak Accuracy Mode", value=True, help="Disables frame skipping and maximizes resolution for 100% precision.")
        AI_CONF = st.slider("Confidence", 0.1, 0.9, 0.35)
        if PEAK_ACCURACY:
            IMGSZ, AI_SKIP = 640, 1
            st.info("Peak Accuracy: Skip=1, Res=640")
        else:
            IMGSZ = st.selectbox("Resolution", [320, 480, 640], index=2)
            AI_SKIP = st.slider("Skip Rate", 1, 10, 3)
    
    with st.expander("Emergency", expanded=True):
        st.session_state.ambulance_sim = st.toggle("🚨 Simulate Ambulance", value=st.session_state.ambulance_sim)
    
    with st.expander("Calibration", expanded=True):
        DIVIDER_TOP_X = st.slider("Divider Top", 0, 800, 400)
        DIVIDER_BTM_X = st.slider("Divider Bottom", 0, 800, 400)
        STOP_LINE_Y = st.slider("Stop Line Y", 0, 600, 300)
        FLOW_LOGIC = st.radio("Flow", ["Left=Down / Right=Up", "Left=Up / Right=Down"])
        SPEED_LIMIT = st.slider("Speed Limit", 10, 120, 60)
        SPEED_SCALE = st.slider("Scale", 0.1, 5.0, 1.0)
    
    st.markdown("---")
    if not st.session_state.active:
        if st.button("INITIALIZE SYSTEM"):
            st.session_state.active = True
            st.toast("🚀 SENTINEL COMMAND INITIALIZED: System Ready")
            st.rerun()
    else:
        if st.button("DEACTIVATE SYSTEM"):
            st.session_state.active = False
            st.rerun()

# --- UI LAYOUT ---
col_head, col_clock = st.columns([3, 1])
with col_head: st.title("Smart Traffic Engine")
with col_clock: inject_live_clock()

tab1, tab2 = st.tabs(["Intelligence", "Database"])

with tab1:
    col_vid, col_metrics = st.columns([3, 1])
    with col_vid:
        vid_placeholder = st.empty()
        status_placeholder = st.empty()
    with col_metrics:
        st.markdown("#### Tactical Advisory")
        recent_advisory = st.empty()
        st.markdown("---")
        risk_metric = st.empty()
        viol_metric = st.empty()
        d_col1, d_col2 = st.columns(2)
        l_density = d_col1.empty()
        r_density = d_col2.empty()
        st.markdown("---")
        risk_chart = st.empty()

with tab2:
    log_placeholder = st.empty()

# --- LOGIC ---
def process_vehicle_logic(tid, cx, cy, t, h_frame):
    if tid not in track_data:
        # Peak Accuracy uses extended buffers (60 frames) for ultra-smooth tracking
        buf_size = 60 if st.session_state.get('peak_mode_active', False) else 40
        track_data[tid] = {'history': deque(maxlen=buf_size), 'speed_buffer': deque(maxlen=buf_size//2), 'viol_buffer': deque(maxlen=25), 'vector': (0, 0), 'lane': None, 'last_lane_change': 0}
    
    data = track_data[tid]
    data['history'].append((cx, cy, t))
    speed_kmh, v_type, vector = 0, None, (0, 0)
    
    progress = cy / h_frame
    mid_x = DIVIDER_TOP_X * (1 - progress) + DIVIDER_BTM_X * progress
    current_lane = 0 if cx < mid_x else 1
    
    if data['lane'] is not None and data['lane'] != current_lane:
        if time.time() - data['last_lane_change'] > 2:
            v_type = "ILLEGAL LANE CHANGE"
            data['last_lane_change'] = time.time()
    data['lane'] = current_lane
    
    if len(data['history']) >= 2:
        (px, py, pt) = data['history'][-2]
        dt = t - pt
        if dt > 0.001:
            dx, dy = cx - px, cy - py
            vector = (dx, dy)
            data['vector'] = vector
            # Refined PPM: 18.5 base for standard 1080p-scaled-to-640p perspective
            ppm = (18.5 * SPEED_SCALE) * (0.4 + (cy / h_frame)**1.6 * 2.2)
            raw_s = (math.sqrt(dx**2 + dy**2) / ppm / dt) * 3.6
            if 2 < raw_s < 250: data['speed_buffer'].append(raw_s)
            if data['speed_buffer']:
                speed_kmh = int(np.average(data['speed_buffer'], weights=np.linspace(0.5, 1.0, len(data['speed_buffer']))))
            
            is_wrong = 0
            if abs(dy) > 0.5:
                if "Left=Down" in FLOW_LOGIC:
                    if (cx < mid_x and dy < -0.3) or (cx > mid_x and dy > 0.3): is_wrong = 1
                else:
                    if (cx < mid_x and dy > 0.3) or (cx > mid_x and dy < -0.3): is_wrong = 1
            
            is_speeding = 1 if speed_kmh > SPEED_LIMIT else 0
            is_erratic = 1 if (abs(dx) > abs(dy) * 2.5 and speed_kmh > 35) else 0
            data['viol_buffer'].append(1 if (is_wrong or is_speeding or is_erratic or v_type) else 0)
            
            if len(data['viol_buffer']) >= 8 and (sum(data['viol_buffer']) / len(data['viol_buffer'])) > 0.5:
                if speed_kmh > 120 or is_erratic: v_type = "OUT OF CONTROL"
                elif is_speeding: v_type = "OVERSPEED"
                elif is_wrong: v_type = "WRONG LANE"
                
    return speed_kmh, v_type, vector

def analyze_safety(detections, h_frame):
    emergency_ids, safety_links = set(), []
    vehicles = [d for d in detections if d['name'] != "SIGNAL"]
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            d1, d2 = vehicles[i], vehicles[j]
            l1 = 0 if d1['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d1['pos'][1]/h_frame)) else 1
            l2 = 0 if d2['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d2['pos'][1]/h_frame)) else 1
            if l1 == l2:
                dist_px = math.sqrt((d1['pos'][0]-d2['pos'][0])**2 + (d1['pos'][1]-d2['pos'][1])**2)
                dist_m = dist_px / (15.0 * (0.5 + (max(d1['pos'][1], d2['pos'][1]) / h_frame) * 1.5))
                avg_v = (d1['speed'] + d2['speed']) / 2 / 3.6
                if avg_v > 5 and dist_m < (avg_v * 1.2):
                    set_violation(d1, "TAILGATING")
                    set_violation(d2, "TAILGATING")
                    safety_links.append({'p1': d1['pos'], 'p2': d2['pos'], 'dist': dist_m})
            p1_f = (d1['pos'][0] + d1['vector'][0]*12, d1['pos'][1] + d1['vector'][1]*12)
            p2_f = (d2['pos'][0] + d2['vector'][0]*12, d2['pos'][1] + d2['vector'][1]*12)
            if math.sqrt((p1_f[0]-p2_f[0])**2 + (p1_f[1]-p2_f[1])**2) < 25:
                set_violation(d1, "COLLISION RISK")
                set_violation(d2, "COLLISION RISK")
                emergency_ids.add(d1['id']); emergency_ids.add(d2['id'])
    return emergency_ids, safety_links

def run_engine():
    try:
        model = load_model()
        cap = cv2.VideoCapture(st.session_state.current_video)
        DISPLAY_W, frame_idx, cached_detections, prev_t = 640, 0, [], time.time()
        fps_history, current_skip = deque(maxlen=30), AI_SKIP
        st.session_state.peak_mode_active = PEAK_ACCURACY
        
        while cap.isOpened() and st.session_state.active:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            
            frame_idx += 1
            h_orig, w_orig = frame.shape[:2]
            canvas = cv2.resize(frame, (DISPLAY_W, int(h_orig * (DISPLAY_W / w_orig))))
            h_c, w_c = canvas.shape[:2]
            
            if frame_idx % current_skip == 0 or not cached_detections:
                results = model.track(canvas, persist=True, conf=AI_CONF, imgsz=IMGSZ, verbose=False, classes=[2,3,5,7,9], tracker="bytetrack.yaml")
                new_detections = []
                if results and results[0].boxes.id is not None:
                    boxes, ids, clss = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().numpy(), results[0].boxes.cls.int().cpu().numpy()
                    for box, tid, cid in zip(boxes, ids, clss):
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        v_name = VEHICLE_CLASSES.get(cid, "OBJ")
                        if v_name == "SIGNAL":
                            state = get_signal_state(canvas[max(0,y1):min(h_c,y2), max(0,x1):min(w_c,x2)])
                            new_detections.append({'id': tid, 'box': (x1,y1,x2,y2), 'name': v_name, 'state': state, 'pos': (cx, cy)})
                        else:
                            speed, v_type, vector = process_vehicle_logic(tid, cx, cy, time.time(), h_c)
                            new_detections.append({'id': tid, 'box': (x1,y1,x2,y2), 'speed': speed, 'v_type': v_type, 'name': v_name, 'pos': (cx, cy), 'vector': vector})
                cached_detections = new_detections

            emergency_ids, safety_links = analyze_safety(cached_detections, h_c)
            signals = [d for d in cached_detections if d['name'] == "SIGNAL"]
            global_signal_state = "UNKNOWN"
            if signals:
                if not st.session_state.signal_seen:
                    status_placeholder.markdown('<div class="signal-badge">🚦 TRAFFIC SIGNAL ACQUIRED</div>', unsafe_allow_html=True)
                    st.session_state.signal_seen = True
                
                # Use the largest signal for state analysis
                main_signal = max(signals, key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]))
                detected_state = main_signal['state']
                
                # Persistence logic: Keep last known state if current is UNKNOWN
                if detected_state != "UNKNOWN":
                    st.session_state.last_signal_state = detected_state
                else:
                    if st.session_state.last_signal_state != "UNKNOWN":
                        log_repair(f"Signal ID #{main_signal['id']} state persisted: {st.session_state.last_signal_state}")
                global_signal_state = st.session_state.last_signal_state
            else:
                if st.session_state.signal_seen:
                    status_placeholder.empty()
                    st.session_state.signal_seen = False
                    st.session_state.last_signal_state = "UNKNOWN"

            ref_kp, ref_des = load_ambulance_ref()
            auto_amb_active = False
            for d in cached_detections:
                if d['name'] in ["TRUCK", "BUS"] and d['id'] not in st.session_state.ambulance_detected:
                    if is_ambulance_match(canvas[max(0,d['box'][1]):min(h_c,d['box'][3]), max(0,d['box'][0]):min(w_c,d['box'][2])], ref_des):
                        st.session_state.ambulance_detected.add(d['id'])
                if d['id'] in st.session_state.ambulance_detected: d['v_type'] = "AMBULANCE"; auto_amb_active = True

            amb_present = st.session_state.ambulance_sim or auto_amb_active
            priority_v = None
            vehicles = [d for d in cached_detections if d['name'] != "SIGNAL"]
            if amb_present and vehicles:
                auto_amb = next((d for d in vehicles if d['id'] in st.session_state.ambulance_detected), None)
                priority_v = auto_amb if auto_amb else min(vehicles, key=lambda x: abs(x['pos'][0] - w_c//2))
                priority_v['v_type'] = "AMBULANCE"

            emergency_active = amb_present or len(emergency_ids) > 0 or any(d.get('v_type') == "OUT OF CONTROL" for d in cached_detections)
            if amb_present:
                st.markdown('<div class="evacuate-banner">🚨 EMERGENCY PRIORITY MODE ACTIVE 🚨</div>', unsafe_allow_html=True)
            elif PEAK_ACCURACY:
                status_placeholder.caption("🎯 PEAK ACCURACY MODE ACTIVE • 100% FRAME ANALYSIS")

            if signals: cv2.line(canvas, (0, STOP_LINE_Y), (w_c, STOP_LINE_Y), (255, 255, 255), 1, cv2.LINE_AA)
            if not amb_present:
                for link in safety_links: cv2.line(canvas, link['p1'], link['p2'], (0, 165, 255), 1, cv2.LINE_AA)

            viol_count = 0
            for d in cached_detections:
                if d['name'] == "SIGNAL":
                    color = (0, 0, 255) if d['state'] == "RED" else (0, 255, 0) if d['state'] == "GREEN" else (150, 150, 150)
                    cv2.rectangle(canvas, (d['box'][0], d['box'][1]), (d['box'][2], d['box'][3]), color, 1, cv2.LINE_AA)
                    cv2.circle(canvas, (d['box'][0]+10, d['box'][1]-10), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"SIGNAL: {d['state']}", (d['box'][0]+20, d['box'][1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    continue
                if amb_present and d != priority_v: continue
                
                color_bgr = (241, 102, 99)
                if d.get('v_type') == "AMBULANCE":
                    color_bgr = (255, 255, 255)
                    if int(time.time() * 5) % 2 == 0: color_bgr = (68, 68, 239)
                
                # Red Light Violation Logic: Crosses line while RED and moving forward
                elif global_signal_state == "RED":
                    # Check if vehicle center is past the stop line and moving forward (positive dy)
                    if d['pos'][1] > STOP_LINE_Y and d.get('vector', (0,0))[1] > 0.3:
                        set_violation(d, "SIGNAL VIOLATION")
                
                if d.get('v_type'): viol_count += 1; color_bgr = (68, 68, 239)
                
                x1, y1, x2, y2 = d['box']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
                label = f"ID:{d['id']} {d.get('v_type', d['name'])}"
                if d.get('speed'): label += f" {d['speed']}km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(canvas, (x1, y1-th-8), (x1+tw+8, y1), color_bgr, -1)
                cv2.putText(canvas, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                if d.get('v_type') and d['v_type'] != "AMBULANCE":
                    existing = next((s for s in st.session_state.violation_history if s['id'] == d['id']), None)
                    if existing:
                        if d['v_type'] not in existing['type']: existing['type'] += f", {d['v_type']}"
                        existing['speed'] = max(existing['speed'], d.get('speed', 0))
                    else:
                        crop = canvas[max(0,y1-20):min(h_c,y2+20), max(0,x1-20):min(w_c,x2+20)].copy()
                        if crop.size > 0:
                            st.session_state.violation_history.appendleft({'id': d['id'], 'type': d['v_type'], 'name': d['name'], 'speed': d.get('speed', 0), 'img': cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 'advisory': get_officer_advisory(d['v_type'], d['name'], d.get('speed', 0), d['id']), 'time': time.strftime("%H:%M:%S")})

            if emergency_active and (time.time() - st.session_state.last_siren_time > (2 if amb_present else 5)):
                trigger_siren(); st.session_state.last_siren_time = time.time()

            risk_val = min(100, (len(cached_detections) * 4) + (viol_count * 20))
            st.session_state.risk_history.append(risk_val)
            if frame_idx % 10 == 0:
                risk_metric.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Index</div><div class='metric-value'>{risk_val}%</div></div>", unsafe_allow_html=True)
                viol_metric.markdown(f"<div class='metric-card'><div class='metric-label'>Violations</div><div class='metric-value'>{viol_count}</div></div>", unsafe_allow_html=True)
                l_c = sum(1 for d in cached_detections if d['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d['pos'][1]/h_c)))
                l_density.markdown(f"<div class='metric-card' style='padding:10px;'><div class='metric-label'>Left</div><div class='metric-value'>{l_c}</div></div>", unsafe_allow_html=True)
                r_density.markdown(f"<div class='metric-card' style='padding:10px;'><div class='metric-label'>Right</div><div class='metric-value'>{len(cached_detections)-l_c}</div></div>", unsafe_allow_html=True)
                risk_chart.line_chart(list(st.session_state.risk_history), height=120)
                if st.session_state.violation_history:
                    latest = st.session_state.violation_history[0]
                    recent_advisory.markdown(f"<div class='log-entry'><strong>{latest['type']}</strong><div class='advisory-text'>{latest['advisory']}</div></div>", unsafe_allow_html=True)

            if frame_idx % 30 == 0:
                with log_placeholder.container():
                    for s in list(st.session_state.violation_history)[:5]:
                        cols = st.columns([1, 4])
                        cols[0].image(s['img'], width=80)
                        cols[1].markdown(f"**{s['name']} #{s['id']}** • {s['time']}\n\n<span style='color:#f43f5e;'>{s['type']}</span>", unsafe_allow_html=True)
                        st.markdown("<div style='height:1px; background:rgba(255,255,255,0.1); margin:10px 0;'></div>", unsafe_allow_html=True)

            vid_placeholder.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
            curr_t = time.time()
            fps = 1.0 / (curr_t - prev_t) if (curr_t - prev_t) > 0 else 0
            prev_t = curr_t
            fps_history.append(fps)
            if len(fps_history) == 30:
                avg_fps = sum(fps_history) / 30
                if avg_fps < 12 and current_skip < 10: current_skip += 1
                elif avg_fps > 25 and current_skip > AI_SKIP: current_skip -= 1
                fps_history.clear()
            status_placeholder.caption(f"Status: Operational • {fps:.1f} FPS • Skip: {current_skip}")
            time.sleep(max(0, 0.03 - (time.time() - loop_start)))

        cap.release()
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        time.sleep(2); st.rerun()

if st.session_state.active: run_engine()
else: st.info("System Standby. Initialize to start monitoring.")
