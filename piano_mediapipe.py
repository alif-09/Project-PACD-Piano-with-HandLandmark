import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Inisialisasi pygame mixer
pygame.mixer.init()

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load suara dengan mapping yang direvisi
sounds = {
    "Do": pygame.mixer.Sound("Sounds/do.wav"),
    "Re": pygame.mixer.Sound("Sounds/re.wav"),
    "Mi": pygame.mixer.Sound("Sounds/mi.wav"),
    "Fa": pygame.mixer.Sound("Sounds/fa.wav"),
    "So": pygame.mixer.Sound("Sounds/sol.wav"),
    "La": pygame.mixer.Sound("Sounds/la.wav"),
    "Si": pygame.mixer.Sound("Sounds/si.wav"),
    "Do tinggi": pygame.mixer.Sound("Sounds/do_tinggi.wav"),
}

# Inisialisasi channel untuk setiap nada
channels = {}
for note in sounds.keys():
    channels[note] = pygame.mixer.Channel(len(channels))

# Fungsi untuk mendapatkan daftar kamera yang tersedia
@st.cache_data
def get_available_cameras():
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# Fungsi untuk mendeteksi jari aktif dengan multiple landmark points
def detect_finger(landmarks):
    fingers = []
    
    # Landmark IDs untuk setiap jari
    # Format: [tip, dip, pip, mcp] untuk setiap jari
    finger_landmarks = {
        0: [4, 3],    # Thumb: tip, ip
        1: [8, 7, 6, 5],    # Index: tip, dip, pip, mcp  
        2: [12, 11, 10, 9], # Middle: tip, dip, pip, mcp
        3: [16, 15, 14, 13], # Ring: tip, dip, pip, mcp
        4: [20, 19, 18, 17]  # Pinky: tip, dip, pip, mcp
    }
    
    # Deteksi ibu jari (orientasi horizontal)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_cmc = landmarks[1]
    
    # Cek apakah ibu jari terangkat (menggunakan multiple points)
    thumb_extended = (
        thumb_tip.x < thumb_ip.x and
        thumb_ip.x < thumb_mcp.x and
        abs(thumb_tip.y - thumb_mcp.y) < 0.1  # Threshold untuk stabilitas
    )
    
    if thumb_extended:
        fingers.append(0)
    
    # Deteksi jari lainnya (orientasi vertikal)
    for finger_id in range(1, 5):
        tip_id, dip_id, pip_id, mcp_id = finger_landmarks[finger_id]
        
        tip = landmarks[tip_id]
        dip = landmarks[dip_id] 
        pip = landmarks[pip_id]
        mcp = landmarks[mcp_id]
        
        # Cek apakah jari terangkat menggunakan multiple points
        finger_extended = (
            tip.y < dip.y and          # Tip di atas DIP
            dip.y < pip.y and          # DIP di atas PIP  
            pip.y < mcp.y and          # PIP di atas MCP
            (tip.y < mcp.y - 0.05)     # Threshold minimum untuk menghindari false positive
        )
        
        # Tambahan: cek kemiringan jari untuk akurasi lebih baik
        finger_straight = abs(tip.x - mcp.x) < 0.08  # Jari harus relatif lurus
        
        if finger_extended and finger_straight:
            fingers.append(finger_id)
    
    return fingers

# Fungsi untuk mendapatkan nada berdasarkan kombinasi jari (mapping)
def get_note_from_fingers(fingers, landmarks=None):
    fingers_set = set(fingers)
    
    # Mapping kombinasi jari ke nada dengan priority scoring (mapping)
    note_combinations = {
        frozenset([0]): ("Do", 1.0),           # Ibu Jari saja
        frozenset([1]): ("Re", 1.0),           # Telunjuk saja
        frozenset([4]): ("Mi", 1.0),           # Kelingking saja
        frozenset([0, 1]): ("Fa", 1.2),        # Ibu Jari + Telunjuk
        frozenset([1, 2]): ("So", 1.2),        # Telunjuk + Jari Tengah
        frozenset([0, 4]): ("La", 1.2),        # Ibu Jari + Kelingking
        frozenset([0, 1, 2]): ("Si", 1.5),     # Ibu Jari + Telunjuk + Jari Tengah
        frozenset([1, 2, 3, 4]): ("Do tinggi", 1.8),  # 4 Jari (Telunjuk-Kelingking)
    }
    
    fingers_frozen = frozenset(fingers_set)
    
    # Cari exact match terlebih dahulu
    if fingers_frozen in note_combinations:
        return note_combinations[fingers_frozen][0]
    
    # Jika tidak ada exact match, cari yang paling mendekati dengan confidence scoring
    best_match = None
    best_score = 0
    
    for combination, (note, base_score) in note_combinations.items():
        if combination.issubset(fingers_frozen):
            # Hitung score berdasarkan kecocokan
            match_ratio = len(combination) / max(len(fingers_frozen), 1)
            score = base_score * match_ratio
            
            if score > best_score:
                best_score = score
                best_match = note
    
    return best_match

# Streamlit app
st.title("🎹 Piano Virtual dengan MediaPipe")
st.write("Gunakan tangan di depan kamera untuk memainkan nada dengan mapping.")

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan")

# Pilihan kamera
available_cameras = get_available_cameras()
if available_cameras:
    selected_camera = st.sidebar.selectbox(
        "Pilih Kamera:", 
        available_cameras, 
        format_func=lambda x: f"Kamera {x}"
    )
else:
    st.sidebar.error("Tidak ada kamera yang tersedia!")
    st.stop()

# Pengaturan sensitivitas
sensitivity = st.sidebar.slider("Sensitivitas Deteksi:", 0.5, 1.0, 0.8, 0.05)
stability_threshold = st.sidebar.slider("Stabilitas Jari:", 0.01, 0.15, 0.05, 0.01)
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=sensitivity,
    min_tracking_confidence=0.7
)

# Informasi mapping jari ke nada (mapping)
st.sidebar.subheader("🎵 Mapping Nada:")
st.sidebar.write("""
- **Ibu Jari**: Do
- **Telunjuk**: Re  
- **Kelingking**: Mi
- **Ibu Jari + Telunjuk**: Fa
- **Telunjuk + Jari Tengah**: So
- **Ibu Jari + Kelingking**: La
- **Ibu Jari + Telunjuk + Jari Tengah**: Si
- **4 Jari (Telunjuk-Kelingking)**: Do Tinggi
""")

st.sidebar.subheader("🔧 Tips Akurasi:")
st.sidebar.write("""
- Jaga jari tetap lurus saat diangkat
- Pastikan pencahayaan cukup terang
- Jaga jarak optimal dari kamera (30-50cm)
- Hindari gerakan terlalu cepat
- Gunakan kombinasi jari yang natural
""")



# Kontrol utama
col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox("🎹 Mulai Piano")
with col2:
    show_landmarks = st.checkbox("Tampilkan Landmark", value=True)

# Display area
FRAME_WINDOW = st.image([])
current_note_display = st.empty()

# Informasi status
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    fps_display = st.empty()
with status_col2:
    finger_display = st.empty()
with status_col3:
    stability_display = st.empty()

# Inisialisasi variabel
cap = None
current_playing_note = None
finger_stability_buffer = []
buffer_size = 5  # Buffer untuk stabilitas deteksi
frame_count = 0
start_time = time.time()

if run:
    cap = cv2.VideoCapture(selected_camera)
    
    if not cap.isOpened():
        st.error(f"Tidak dapat membuka kamera {selected_camera}!")
        st.stop()
    
    # Set resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca frame dari kamera!")
            break

        # Hitung FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            fps_display.metric("FPS", f"{fps:.1f}")
            frame_count = 0
            start_time = current_time

        # Flip frame secara horizontal agar seperti cermin
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_note = "Tidak ada"
        note_detected = False
        detected_fingers = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Gambar landmark jika diaktifkan
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Deteksi jari yang aktif
                fingers = detect_finger(hand_landmarks.landmark)
                detected_fingers = fingers.copy()
                
                # Tambahkan ke buffer untuk stabilitas
                finger_stability_buffer.append(fingers)
                if len(finger_stability_buffer) > buffer_size:
                    finger_stability_buffer.pop(0)
                
                # Hitung stabilitas - ambil yang paling konsisten
                if len(finger_stability_buffer) >= 3:
                    # Cari pola yang paling sering muncul
                    finger_patterns = {}
                    for pattern in finger_stability_buffer:
                        pattern_key = tuple(sorted(pattern))
                        finger_patterns[pattern_key] = finger_patterns.get(pattern_key, 0) + 1
                    
                    if finger_patterns:
                        # Ambil pola yang paling stabil (paling sering muncul)
                        stable_pattern = max(finger_patterns.items(), key=lambda x: x[1])
                        stable_fingers = list(stable_pattern[0])
                        stability_score = stable_pattern[1] / len(finger_stability_buffer)
                        
                        if stable_pattern[1] >= 2:  # Minimal muncul 2 kali untuk dianggap stabil
                            fingers = stable_fingers
                            stability_display.metric("Stabilitas", f"{stability_score:.1%}")
                
                if fingers:
                    note = get_note_from_fingers(fingers, hand_landmarks.landmark)
                    
                    if note:
                        current_note = note
                        note_detected = True
                        
                        # Jika nada berubah, stop nada sebelumnya dan mulai yang baru
                        if note != current_playing_note:
                            # Stop semua channel
                            for channel in channels.values():
                                channel.stop()
                            
                            # Mulai nada baru dengan loop
                            try:
                                channels[note].play(sounds[note], loops=-1)  # loops=-1 untuk infinite loop
                                current_playing_note = note
                            except KeyError:
                                st.error(f"Sound file untuk nada '{note}' tidak ditemukan!")
        
        # Jika tidak ada jari yang terdeteksi, stop semua suara
        if not note_detected:
            if current_playing_note:
                for channel in channels.values():
                    channel.stop()
                current_playing_note = None
            finger_stability_buffer.clear()  # Clear buffer ketika tidak ada deteksi
            stability_display.metric("Stabilitas", "0%")

        # Update display informasi jari
        finger_names = {0: "Ibu Jari", 1: "Telunjuk", 2: "Tengah", 3: "Manis", 4: "Kelingking"}
        finger_text = ", ".join([finger_names[f] for f in detected_fingers]) if detected_fingers else "Tidak ada"
        finger_display.metric("Jari Terdeteksi", finger_text)

        # Tambahkan informasi debug
        debug_info = f"Fingers: {detected_fingers}"
        cv2.putText(frame, debug_info, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Tambahkan text overlay untuk nada saat ini
        cv2.putText(frame, f"Nada: {current_note}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tambahkan informasi mapping pada frame
        mapping_info = [
            "Mapping: Ibu Jari=Do, Telunjuk=Re, Kelingking=Mi",
            "Kombinasi: IJ+T=Fa, T+TG=So, IJ+K=La, IJ+T+TG=Si, 4Jari=Do'"
        ]
        for i, info in enumerate(mapping_info):
            cv2.putText(frame, info, (10, frame.shape[0] - 40 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Update display
        FRAME_WINDOW.image(frame)
        
        # Update status nada dengan styling yang lebih menarik
        if current_note != "Tidak ada":
            current_note_display.markdown(f"### 🎵 **{current_note}** 🎶")
        else:
            current_note_display.markdown("### 🎵 Nada Saat Ini: **Tidak ada**")
        
        # Break jika checkbox dimatikan
        if not st.session_state.get('run', True):
            break

    # Cleanup
    if cap:
        cap.release()
    
    # Stop semua suara ketika piano dihentikan
    for channel in channels.values():
        channel.stop()
    current_playing_note = None
        
else:
    # Reset display ketika tidak berjalan
    current_note_display.markdown("### 🎵 Nada Saat Ini: **Piano Berhenti**")
    fps_display.metric("FPS", "0")
    finger_display.metric("Jari Terdeteksi", "Tidak aktif")
    stability_display.metric("Stabilitas", "0%")
    
    # Stop semua suara ketika checkbox dimatikan
    for channel in channels.values():
        channel.stop()
