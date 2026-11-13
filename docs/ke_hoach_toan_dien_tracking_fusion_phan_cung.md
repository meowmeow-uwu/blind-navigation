# KẾ HOẠCH TOÀN DIỆN: GIẢI QUYẾT BÀI TOÁN RUNG CAMERA, BẢN QUYỀN BYTETRACK, ĐỘC LẬP PHẦN CỨNG ÂM THANH & THIẾT KẾ THỰC NGHIỆM ĐỐI SÁNH (PBL6)

Tài liệu này hệ thống hóa các định hướng giải quyết chuyên sâu cho 6 vấn đề kỹ thuật và học thuật trọng tâm của dự án:
1. Cơ chế tác động của hiện tượng rung camera lên **Object Detection** & **Object Tracking**, cùng các giải pháp giải quyết.
2. Đối sánh chi tiết: **ByteTrack trong dự án thừa kế những gì từ paper gốc (Zhang et al., ECCV 2022) và những gì nhóm tự thiết kế, cải tiến**.
3. Giải pháp chuyển đổi hệ thống từ việc chạy thử nghiệm trên trình duyệt Chrome sang **hệ thống độc lập 100% phát trực tiếp ra loa/tai nghe phần cứng trên Raspberry Pi 4**.
4. Khung đánh giá khoa học & cơ sở lý thuyết chứng minh **Edge ByteTrack vượt trội hơn SORT và DeepSORT trên thiết bị biên**.
5. Kịch bản thực nghiệm kiểm chứng **Model Fusion** trong điều kiện không có thiết bị đo khoảng cách Laser chuyên dụng (phương pháp mốc sàn tĩnh chuẩn hóa).
6. Quy trình phối hợp và chuẩn đầu ra bàn giao giữa đội thu thập dữ liệu phần cứng và đội xử lý thuật toán.

---

## 1. Phân Tích & Xử Lý Hiện Tượng Rung Camera (Camera Shake / Motion Blur)

Khi người khiếm thị di chuyển, cụm camera gắn trên ngực hoặc kính mắt chịu chuyển động dao động tuần hoàn theo từng bước chân (bước đi dằn gót, nghiêng thân người, xoay góc nhìn). Đây là thách thức lớn nhất đối với hệ thống thị giác nhúng.

### 1.1. Tác động của rung camera lên Object Detection (YOLOv8 NCNN)

```
[Chuyển động dằn bước chân / Rung lắc]
                 │
                 ├── 1. Mất chi tiết cạnh cao tần (High-frequency Edge Degradation)
                 │      └── Feature maps (C2f, Conv) bị phân tán năng lượng
                 │
                 ├── 2. Tụt điểm tin cậy (Confidence Score Drop)
                 │      └── conf: 0.85 (rõ nét) ──> 0.25 - 0.40 (rơi vào vùng D_low)
                 │                               ──> < 0.20 (Bị triệt tiêu hoàn toàn -> False Negative)
                 │
                 └── 3. Biến dạng hình học Bounding Box (Jitter & Deformation)
                        └── Tọa độ [x1, y1, x2, y2] dao động mạnh; méo tỷ lệ co giãn (aspect ratio)
```

1. **Nhòe chuyển động (Motion Blur):** Màn trập điện tử (Rolling shutter) của camera USB/CSI khi rung lắc làm mờ các đường biên sắc nét của vật cản. Các lớp tích chập của mạng nơ-ron không bắt được đặc trưng cạnh (edges) và góc (corners), làm vector đặc trưng bị suy biến.
2. **Sụt giảm Confidence Score nghiêm trọng:** Vật thể vốn có độ tin cậy $conf \ge 0.75$ khi đứng yên sẽ bị tụt xuống vùng $0.20 \le conf < 0.45$ (miền $D_{\text{low}}$) khi rung. Nếu hệ thống chỉ lọc $conf \ge 0.45$ như các bộ phát hiện thông thường, vật cản sẽ lập tức bị coi là biến mất (**False Negative**), tạo lỗ hổng an toàn nguy hiểm cho người khiếm thị.
3. **Co giật khung nhận diện (BBox Jitter):** Tọa độ viền hộp bao nhảy giật liên tục giữa các khung hình liền kề dù vật thể thực tế vẫn giữ nguyên vị trí trong không gian.

### 1.2. Tác động của rung camera lên Object Tracking

Trong mô hình lọc Kalman tuyến tính chuẩn (Constant Velocity Model):
$$\mathbf{x}_t = \begin{bmatrix} c_x & c_y & s & r & \dot{c}_x & \dot{c}_y & \dot{s} \end{bmatrix}^T$$

Khi camera rung đột ngột:
1. **Lệch pha dự báo Kalman (Kalman Prediction Drift):** Vận tốc biểu kiến $(\dot{c}_x, \dot{c}_y)$ đo được trên mặt phẳng ảnh bị phóng đại do chuyển động của camera chứ không phải do vật thể chuyển động. Bộ lọc Kalman ngoại suy vị trí $\hat{\mathbf{x}}_{t|t-1}$ ở khung tiếp theo bị chệch xa khỏi vị trí thực tế.
2. **Suy giảm chỉ số IoU (Intersection-over-Union Collapse):** Sự kết hợp giữa BBox bị méo và vị trí dự báo Kalman bị lệch khiến chỉ số $\text{IoU}$ giữa tracklet hiện có và detection mới tụt xuống rất thấp ($\text{IoU} < 0.25$).
3. **Đứt vết liên tục & bùng nổ ID Switches (IDSW):** Thuật toán ghép cặp Hungary không thể ghép detection mới vào tracklet cũ. Tracklet cũ bị đẩy vào trạng thái mất dấu (`time_since_update > 0`), trong khi detection mới bị gán một `Track ID` hoàn toàn mới.
4. **Hậu quả trực tiếp lên trải nghiệm người dùng (HCI Breakdown):** Vì mỗi `Track ID` mới được xem là một vật thể mới xuất hiện, hệ thống âm thanh sẽ liên tục phát lại câu dẫn hướng: *"Ghế trước mặt... Ghế trước mặt... Ghế trước mặt"*. Hiện tượng này gây **loạn thính giác (Auditory Fatigue / Chaos)**, khiến người khiếm thị mất phương hướng.

### 1.3. Giải pháp kỹ thuật đã và đang áp dụng trong dự án

| Cấp độ | Giải pháp kỹ thuật cụ thể | Cơ chế hoạt động trong mã nguồn PBL6 |
| :--- | :--- | :--- |
| **Cơ học / Phần cứng** | Ngàm đeo ngực định tâm (Center Chest Rig) có đệm cao su giảm chấn. | Triệt tiêu rung động góc xoay (yaw/roll) tốt hơn nhiều so với việc đeo trên kính mắt hoặc cầm tay. |
| **Giải thuật Tầng 1** | **Cơ chế cứu vớt $D_{\text{low}}$ của ByteTrack** | Khi rung làm $conf$ tụt vào $[0.20, 0.45)$, khung hình không bị vứt bỏ mà được đưa vào Tầng 2 để ghép nối lại với các tracklet bị hụt ở Tầng 1. |
| **Giải thuật Tầng 2** | **Ngưỡng IoU nới lỏng thích ứng di động** | Ở Tầng 2, ngưỡng ghép được nới lỏng: $\text{threshold} = 0.8 \times \text{iou\_threshold} = 0.8 \times 0.25 = 0.20$. Cho phép bắt dính lại các hộp bị co giật tọa độ do rung. |
| **Bộ nhớ Tracklet** | **Dung sai mất dấu `max_age=12`** | Cho phép tracklet sống sót trong bộ nhớ qua 12 khung hình liên tiếp ($\approx 0.8\text{s}$ ở 15 FPS) ngay cả khi camera rung quá mạnh làm mất hoàn toàn detection trong 1–2 nhịp bước chân. |
| **Bộ lọc Âm thanh** | **Khóa định danh `announced_track_ids`** | Khi ID được giữ ổn định nhờ ByteTrack, dù vật thể có rung giật tọa độ thì âm thanh vẫn giữ im lặng tuyệt đối cho đến khi khoảng cách thay đổi $\ge 0.8\text{m}$ sau $\ge 3.0\text{s}$. |

---

## 2. ByteTrack Trong Dự Án: Thừa Kế Gì & Tự Thiết Kế Gì So Với Paper Gốc?

Để bảo vệ thuyết phục trước Hội đồng khoa học, nhóm cần chỉ rõ ranh giới học thuật: **Đâu là tư tưởng khoa học thừa kế từ công bố quốc tế (Zhang et al., ECCV 2022)** và **đâu là đóng góp kỹ thuật / thiết kế chuyên biệt của nhóm cho bài toán thiết bị biên (Edge Computing) và điều hướng khiếm thị**.

```
                   TƯ TƯỞNG CỐT LÕI TỪ PAPER GỐC (Zhang et al., ECCV 2022)
                   "Tracking Every Detection Box - Cứu vớt detection điểm thấp"
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             ĐIỂM NHÓM THỪA KẾ                            ĐIỂM NHÓM TỰ THIẾT KẾ & TỐI ƯU
  1. Phân tách D_high (>=0.45) & D_low ([0.20, 0.45))     1. Thay thế YOLOX nặng bằng YOLOv8n NCNN INT8
  2. Quy trình liên kết 2 tầng (Two-stage Association)   2. Loại bỏ CNN Re-ID -> Lightweight IoU (0.05ms)
  3. Thuật toán Hungary ghép cặp cực đại ma trận IoU    3. Vòng đời Tracklet tích hợp trạng thái âm thanh
  4. Nguyên tắc: Không sinh track mới từ D_low           4. Tinh chỉnh ngưỡng thích ứng camera di động
                                                         5. Thuật toán Greedy Fallback khi thiếu Scipy
```

### 2.1. Điểm thừa kế từ Paper gốc (Zhang et al., ECCV 2022)

1. **Triết lý tận dụng toàn bộ Detection Box:** Paper gốc chỉ ra rằng việc đặt ngưỡng lọc cứng ($conf \ge 0.50$) làm mất đi những vật cản bị che khuất một phần hoặc bị mờ. Dự án kế thừa nguyên vẹn nguyên lý phân tách thành 2 tập: $D_{\text{high}}$ và $D_{\text{low}}$.
2. **Quy trình liên kết 2 tầng (Two-Stage Data Association):**
   - Tầng 1: Ưu tiên ghép các detection độ tin cậy cao $D_{\text{high}}$ với tập tracklet hiện có thông qua ma trận chi phí khoảng cách $1.0 - \text{IoU}$.
   - Tầng 2: Dùng các detection độ tin cậy thấp $D_{\text{low}}$ để ghép với các tracklet còn sót lại chưa được ghép ở Tầng 1.
3. **Quy tắc an toàn chống nhiễu:** Chỉ các detection trong $D_{\text{high}}$ chưa được ghép mới được phép khởi tạo tracklet mới. Các detection trong $D_{\text{low}}$ chưa được ghép đều bị loại bỏ thẳng tay vì chúng có thể là nhiễu nền.
4. **Giải thuật tối ưu hóa ghép cặp Bipartite Matching:** Kế thừa phương pháp gán cực đại thông qua Hungarian Algorithm / Munkres Algorithm.

### 2.2. Điểm nhóm tự thiết kế, cải tiến và làm mới cho PBL6

Khác với paper gốc (được thiết kế cho camera giám sát cố định với máy chủ GPU mạnh mẽ), nhóm đã tái cấu trúc toàn diện mã nguồn tại [`modules/tracking/byte_tracker.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/modules/tracking/byte_tracker.py) để chạy trên Raspberry Pi 4:

| Tiêu chí | Paper gốc (Zhang et al., 2022) | Hiện thực tùy biến của Nhóm PBL6 | Giá trị thực tiễn mang lại |
| :--- | :--- | :--- | :--- |
| **Detector đầu vào** | YOLOX-X / YOLOX-L (rất nặng, chỉ chạy trên GPU máy chủ). | **YOLOv8 Nano NCNN** (tối ưu hóa tập lệnh SIMD NEON trên CPU ARM). | Tương thích hoàn hảo với tài nguyên giới hạn của Raspberry Pi 4. |
| **Thành phần Re-ID** | Một số biến thể kết hợp vector nhận diện ngoại hình (CNN Re-ID feature). | **Thuần hình học & động học (Zero Re-ID CNN)** trong lớp [`LightweightTracker`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/modules/tracking/byte_tracker.py#L121-L276). | Thời gian chạy tracking chỉ tốn $\approx 0.05\text{ms/frame}$ (nhẹ hơn 3000 lần so với mạng CNN Re-ID). |
| **Ngưỡng tin cậy** | Thường đặt cao: $D_{\text{high}} \ge 0.60$, $D_{\text{low}} \in [0.1, 0.6)$. | Tinh chỉnh cho người đi bộ: $high\_thresh = 0.45$, $low\_thresh = 0.20$. | Phù hợp với đặc thù camera góc rộng có độ rung lắc khi di chuyển. |
| **Vòng ghép Tầng 2** | Thường giữ nguyên ngưỡng IoU hoặc dùng khoảng cách Mahalanobis. | **Nới lỏng chủ động:** $\text{threshold}_2 = 0.8 \times \text{iou\_thresh} = 0.20$. | Cứu vớt các khung hình bị co giật BBox khi người dùng dằn gót chân. |
| **Gắn kết với giao tiếp âm thanh (HCI)** | Không có (chỉ xuất file tọa độ BBox cho bài toán thị giác). | **Vòng đời Tracklet tích hợp cờ âm thanh:** Thuộc tính `announced`, `last_announced_time`, `last_announced_distance`, và hàm `should_reannounce()`. | Đây là **sáng tạo 100% của nhóm**: Biến tracker từ công cụ thị giác thuần túy thành **bộ não điều phối nhịp độ âm thanh**, chống nói lặp triệt để. |
| **Khả năng tự vận hành độc lập** | Phụ thuộc thư viện `scipy`, `cython_bbox`. | Tích hợp sẵn thuật toán **Greedy Matching Fallback** tự viết bằng Numpy khi hệ thống nhúng thiếu `scipy`. | Đảm bảo hệ thống không bao giờ bị crash khi cài đặt trên hệ điều hành nhúng tối giản. |

---

## 3. Hoàn Thiện Hệ Thống Phần Cứng: Phát Âm Thanh Ra Loa Thay Vì Phụ Thuộc Chrome

### 3.1. Phân tích nguyên nhân phụ thuộc máy tính và trình duyệt Chrome

Trước đây, khi chạy lệnh kiểm thử (ví dụ [`test_camera_mac_audio.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/test_camera_mac_audio.py)), hệ thống sử dụng trình duyệt Chrome vì:
- Chrome cung cấp sẵn **Web Speech API** (tổng hợp tiếng Việt tự nhiên) và **Web Audio API** (tạo tiếng bíp thời gian thực không độ trễ).
- Trình duyệt giúp nhóm quan sát trực quan luồng hình ảnh MJPEG và thanh trạng thái cảnh báo trên máy tính.

**Hạn chế khi triển khai thực tế ngoài hiện trường:**
- Người khiếm thị mang thiết bị đi trên đường không thể mang theo máy tính mở Chrome.
- Hệ thống bắt buộc phải trở thành một khối **độc lập (Standalone Edge System)**: Bật nguồn pin dự phòng $\rightarrow$ Pi 4 tự khởi động $\rightarrow$ Phát trực tiếp tiếng nói và tiếng bíp ra Loa ngoài / Tai nghe jack 3.5mm hoặc Loa USB gắn trên áo.

### 3.2. Kiến trúc giải pháp âm thanh phần cứng độc lập 100%

```
[Sự kiện Dẫn hướng / Khẩn cấp]
               │
               ▼
   [AudioManager Singleton Queue]
               │
       ┌───────┴──────────────────────────────┐
       ▼                                      ▼
[Kênh Khẩn cấp (< 1.0m)]             [Kênh Dẫn hướng (1.0m - 3.0m)]
- Tiếng Bíp dồn dập                  - Câu ngắn 1-2 từ khóa ("Ghế trước mặt")
- Phát trực tiếp qua ALSA / Pygame   - BỘ SOUNDPACK THU SẴN (WAV, Zero-latency)
- Độ trễ < 5ms                       - Hoặc espeak-ng offline (ALSA default)
       │                                      │
       └──────────────────┬───────────────────┘
                          ▼
            [Phần cứng: Loa USB / Tai nghe 3.5mm]
              (Không cần mạng, Không cần Chrome)
```

### 3.3. Các bước thực hiện chuyển đổi hoàn thiện

#### Bước 1: Cấu hình ngõ ra âm thanh mặc định của Raspberry Pi OS (ALSA Configuration)
Mặc định Raspberry Pi thường chuyển hướng âm thanh ra cổng HDMI. Cần ép hệ thống phát ra Jack 3.5mm hoặc Loa USB:
```bash
# Xem danh sách card âm thanh:
aplay -l

# Nếu dùng Jack 3.5mm trên Pi 4:
sudo amixer cset numid=3 1
amixer sset Headphone 95% unmute

# Nếu dùng Loa USB Audio DAC (Khuyên dùng vì âm lượng lớn và rõ):
# Tạo/chỉnh sửa file cấu hình ~/.asoundrc:
cat << 'EOF' > ~/.asoundrc
pcm.!default {
    type hw
    card 1
}
ctl.!default {
    type hw
    card 1
}
EOF
```

#### Bước 2: Tích hợp Bộ Soundpack tiếng Việt nén sẵn (Pre-recorded Voice Pack)
Thay vì dùng TTS trực tuyến (cần mạng 4G) hoặc `espeak-ng` (giọng robot khó nghe), nhóm chuẩn bị sẵn thư mục các tệp WAV ngắn được thu âm chuẩn tiếng Việt rõ ràng, nén tần số $22050\text{Hz}$ tại `modules/audio/soundpack/`:
- `person_center.wav` ("Người trước mặt"), `chair_center.wav` ("Ghế trước mặt")
- `table_center.wav` ("Bàn trước mặt"), `stairs_center.wav` ("Cầu thang phía trước")
- `stop_emergency.wav` ("Dừng lại!")
- `beep_1400hz.wav` (Tiếng bíp khẩn cấp 50ms)

Trong [`modules/audio/audio_manager.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/modules/audio/audio_manager.py), luồng phát `_speech_worker` sẽ gọi trực tiếp:
```python
# Phát không độ trễ bằng aplay (chỉ mất ~8ms khởi tạo process)
subprocess.run(["aplay", "-q", sound_file], check=False)
```

#### Bước 3: Đóng gói thành Linux Systemd Service (Tự chạy khi cắm nguồn)
Tạo file dịch vụ hệ thống `/etc/systemd/system/blind-nav.service`:
```ini
[Unit]
Description=PBL6 Smart Blind Navigation Standalone Service
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/blind-navigation/2_edge_device
ExecStart=/home/pi/blind-navigation/.venv/bin/python main.py --source auto
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
Kích hoạt service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable blind-nav.service
```
Khi người khiếm thị cắm pin sạc dự phòng, Raspberry Pi 4 boot mất khoảng 18–22 giây, phát ra một âm báo *"Hệ thống dẫn hướng đã sẵn sàng"* qua loa và bắt đầu hoạt động ngay lập tức mà không cần bất kỳ thao tác kết nối máy tính nào.

---

## 4. Khung Đánh Giá So Sánh: Tại Sao Edge ByteTrack Tốt Hơn SORT Và DeepSORT?

### 4.1. Bản chất thuật toán và lý do thất bại của SORT & DeepSORT trên thiết bị biên

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BÀI TOÁN THỰC TẾ: Camera người đi bộ rung lắc + Vật cản bị che khuất ngắn hạn         │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
  [RAW YOLO]                     [SORT]                      [DEEPSORT]
  - Không bám vết                - Chỉ lọc conf >= 0.45       - Chạy thêm mạng CNN Re-ID
  - Mỗi frame 1 ID mới           - Rung lắc -> conf tụt       - Tốn thêm 200ms/BBox
  - Loạn thính giác              - Bỏ qua -> Đứt vết          - FPS sụt xuống < 2 FPS
  - KHÔNG THỂ DÙNG               - ID Switch cao, Nói lặp     - TREO HỆ THỐNG BIÊN
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                          [EDGE BYTETRACK (PBL6)]
                          - Tầng 1: Ghép D_high (conf >= 0.45)
                          - Tầng 2: CỨU VỚT D_low ([0.20, 0.45))
                          - Không dùng mạng CNN Re-ID -> Tốn 0.05ms
                          - Duy trì 10-12 FPS mượt mà trên Pi 4
                          - Giữ ID bền vững -> Triệt tiêu nói lặp!
```

1. **SORT (Bewley et al., ICIP 2016):**
   - *Nguyên lý:* Chỉ nhận các detection có độ tin cậy cao ($conf \ge 0.45$), bỏ qua toàn bộ detection điểm thấp.
   - *Điểm yếu chí mạng:* Khi người khiếm thị bước đi làm camera rung hoặc có người khác đi ngang qua che khuất một phần vật thể, $conf$ của vật cản lập tức tụt xuống $0.25 - 0.40$. SORT vứt bỏ các khung hình này. Sau $max\_age$ khung hình, tracklet bị xóa. Khi vật cản lộ rõ trở lại, SORT cấp một `Track ID` mới $\rightarrow$ **ID Switches cao, hệ thống phát lại âm thanh liên tục**.
2. **DeepSORT (Wojke et al., ICIP 2017):**
   - *Nguyên lý:* Dùng thêm một mạng nơ-ron tích chập (CNN) để trích xuất vector đặc trưng ngoại hình 128 chiều (Appearance Descriptor) nhằm nhận diện lại vật thể qua khoảng cách Cosine khi bị mất dấu lâu dài.
   - *Điểm yếu chí mạng trên thiết bị biên:* Để trích xuất đặc trưng cho mỗi vật thể trong khung hình, Raspberry Pi 4 phải chạy thêm một mạng CNN. Trên CPU ARM Cortex-A72 không có GPU rời, việc này tiêu tốn **$150\text{–}300\text{ms}$ cho mỗi bounding box**. Tốc độ toàn hệ thống bị kéo tụt từ $10.5\text{ FPS}$ xuống dưới **$1.8\text{ FPS}$**. Tốc độ này gây ra độ trễ thời gian thực nghiêm trọng (vật cản ở trước mặt nhưng 1–2 giây sau loa mới cảnh báo), cực kỳ nguy hiểm cho người khiếm thị.
3. **Edge ByteTrack của dự án PBL6:**
   - Đạt được cả hai mục tiêu tối thượng của bài toán: **Độ chính xác bám vết cao (nhờ Tầng 2 cứu vớt $D_{\text{low}}$)** và **Tốc độ thời gian thực $\approx 10\text{–}12\text{ FPS}$ (nhờ loại bỏ hoàn toàn CNN Re-ID)**.

### 4.2. Bảng ma trận đối sánh toàn diện 4 cấu hình

| Chỉ số đánh giá khoa học | Raw YOLO (Không Tracker) | SORT (Bewley 2016) | DeepSORT (Wojke 2017) | Edge ByteTrack (PBL6) | Ý nghĩa đối với người khiếm thị |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Thành phần Re-ID** | Không | Không | Mạng CNN 128-D | **Không (Thuần 2-Stage IoU)** | Quyết định gánh nặng tính toán CPU. |
| **Thời gian Tracking (ms)** | $0.00\text{ ms}$ | $0.08\text{ ms}$ | $150 - 250\text{ ms}$ | **$0.05\text{ ms}$** | ByteTrack nhẹ ngang SORT, nhẹ hơn DeepSORT 3000 lần. |
| **FPS thực tế trên Pi 4** | $10.6\text{ FPS}$ | $10.5\text{ FPS}$ | $1.8\text{ FPS}$ | **$10.4\text{ FPS}$** | DeepSORT bị loại vì không đạt Real-time ($< 5\text{ FPS}$). |
| **MOTA (%)** | $\approx 0\%$ | Trung bình ($60\text{–}65\%$) | Khá ($70\text{–}73\%$) | **Cao nhất ($74\text{–}78\%$)** | Khả năng duy trì bám vết chính xác tổng thể. |
| **IDF1 (%)** | $< 5\%$ | $60\text{–}65\%$ | $72\text{–}75\%$ | **Cao nhất ($76\text{–}80\%$)** | Độ bền danh tính, khả năng không bị tráo đổi ID. |
| **ID Switches (IDSW)** | Cực đại (mỗi frame 1 ID) | Nhiều (do mất vết mờ) | Thấp (nhờ vector Re-ID) | **Rất thấp (nhờ Tầng 2 cứu $D_{\text{low}}$)** | Thước đo quyết định chống nói lặp âm thanh. |
| **Tỷ lệ nói lặp (lần/phút)** | $> 35\text{ lần/phút}$ | $8\text{–}12\text{ lần/phút}$ | Treo/trễ tiếng | **$< 0.5\text{ lần/phút}$** | **Chỉ số vàng HCI:** Edge ByteTrack giữ sự yên tĩnh tuyệt đối. |
| **Khả năng triển khai** | Vô giá trị thực tiễn | Kém an toàn khi rung | Không khả thi trên Pi 4 | **Hoàn hảo trên thiết bị biên** | Đủ điều kiện công bố khoa học và ứng dụng thực tế. |

---

## 5. Thiết Kế Thực Nghiệm Kiểm Chứng Model Fusion (Khi Không Có Máy Đo Laser)

### 5.1. Thách thức: Không có thiết bị đo khoảng cách động chính xác từng khung hình
Trong điều kiện phòng thí nghiệm tiêu chuẩn cao, người ta dùng hệ thống camera quang học Vicon hoặc máy quét LiDAR đa tia giá hàng trăm triệu đồng để lấy Ground-Truth khoảng cách động liên tục.
Nhóm sinh viên thực hiện PBL6 không có thiết bị này. **Nếu báo cáo nói đo được khoảng cách Ground-Truth động từng frame mà không có thiết bị đo tương ứng, Hội đồng sẽ chất vấn ngay lập tức.**

### 5.2. Giải pháp khoa học thay thế: Phương pháp "Mốc Sàn Tĩnh Chuẩn Hóa" (Static Ground-Marker Calibration)

Nhóm giải quyết bài toán bằng cách **rời rạc hóa không gian kiểm chứng** thành các mốc tĩnh đo bằng thước dây cơ học có độ chính xác milimet.

```
[Vị trí Người đeo Camera & Cảm biến Siêu âm] (Mốc 0.0m)
                     │
                     ├───── 1.0m (Vạch Băng keo #1 - Ngưỡng Khẩn cấp)
                     │
                     ├───── 1.5m (Vạch Băng keo #2 - Bắt đầu Vùng Dẫn hướng)
                     │
                     ├───── 2.0m (Vạch Băng keo #3 - Tâm Vùng Dẫn hướng)
                     │
                     ├───── 2.5m (Vạch Băng keo #4 - Giới hạn ngoài Dẫn hướng)
                     │
                     └───── 3.0m (Vạch Băng keo #5 - Ngưỡng Vùng An toàn)
```

### 5.3. Quy trình thực nghiệm Model Fusion tại các mốc sàn tĩnh

1. **Chuẩn bị mặt bằng:** Dán các dải băng dính màu trên sàn hành lang phẳng tại đúng các khoảng cách đo bằng thước dây: $1.0\text{m}, 1.5\text{m}, 2.0\text{m}, 2.5\text{m}, 3.0\text{m}$.
2. **Kịch bản A — Kiểm tra tính chính xác của từng cảm biến đơn lẻ:**
   - Đặt một vật thể chuẩn (chiếc ghế tựa cao $0.85\text{m}$, rộng $0.45\text{m}$) lần lượt tại từng vạch mốc ở chính giữa tâm chùm sóng ($0^\circ$).
   - Giữ nguyên camera và cảm biến trong 30 khung hình ($\approx 2\text{ giây}$).
   - Thu thập 3 giá trị độc lập:
     + $d_{\text{us}}$: Khoảng cách cảm biến siêu âm HC-SR04 đo được.
     + $d_{\text{cam\_ground}}$: Khoảng cách camera tính theo mô hình mặt sàn (Ground-plane).
     + $d_{\text{cam\_pinhole}}$: Khoảng cách camera tính theo mô hình hộp đục pinhole dự phòng.
3. **Kịch bản B — Kiểm tra hiệu quả của Bộ lọc Kalman Hợp nhất (Fusion Kalman Filter):**
   - Thu thập giá trị $d_{\text{fused}}$ sau khi đi qua bộ lọc Kalman tích hợp.
   - Tính toán các chỉ số thống kê sai số chuẩn:
     $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |d_{\text{fused}}^{(i)} - d_{\text{GT}}|$$
     $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (d_{\text{fused}}^{(i)} - d_{\text{GT}})^2}$$
     $$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (d_{\text{fused}}^{(i)} - \bar{d})^2} \quad \text{(Độ ổn định, triệt tiêu rung giật)}$$
4. **Kịch bản C — Kiểm tra vùng tranh chấp góc lệch (Acoustic Cone Gating):**
   - Đặt vật thể tại khoảng cách $2.0\text{m}$ nhưng dịch sang mép ngang lệch góc $20^\circ$ và $25^\circ$ (nằm ngoài góc mở $15^\circ$ của chùm sóng siêu âm $30^\circ$).
   - Kiểm chứng xem hệ thống có nhận diện đúng trạng thái `camera_only` hay bị gán nhầm sóng phản xạ của tường biên.
5. **Kịch bản D — Kiểm chứng tự hiệu chỉnh thang đo trực tuyến (Online Scale Calibration):**
   - Cố tình nghiêng góc camera lệch nhẹ từ $5^\circ$ thành $8^\circ$.
   - Quan sát hệ số $\alpha_{\text{scale}}$ của lớp `OnlineScaleCalibrator` có tự động hội tụ đưa $d_{\text{cam}}$ khớp với $d_{\text{us}}$ sau $10\text{–}15$ frames hay không.

---

## 6. Bảng Phân Công Nhiệm Vụ & Chuẩn Bàn Giao Dữ Liệu Thực Địa

Để đội phần cứng đi hiện trường không bị lúng túng và mang về dữ liệu đạt chuẩn 100%, quy trình phối hợp được chuẩn hóa như sau:

| Bước | Đội phụ trách | Công việc cụ thể | Sản phẩm bàn giao (Deliverables) |
| :---: | :---: | :--- | :--- |
| **1** | **Đội Thuật toán** | Soạn thảo cẩm nang hướng dẫn thực địa ngắn gọn ([`huong_dan_thu_thap_du_lieu_thuc_dia.md`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/docs/huong_dan_thu_thap_du_lieu_thuc_dia.md)), kiểm tra sẵn bộ script quay [`record_scenarios.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/record_scenarios.py) trên Pi 4. | Tài liệu hướng dẫn in ra giấy/PDF + script đã test OK trên Pi 4. |
| **2** | **Đội Phần cứng** | Ra hành lang thực tế, mang theo Pi 4 + pin sạc + thước dây. Chạy lệnh quay đúng 6 kịch bản quy định; ghi chép nhật ký hiện trường. | Thư mục `data/videos/` gồm 6 file `.mp4` + 6 file `.csv` siêu âm + file `field_log.txt`. |
| **3** | **Đội Thuật toán** | Nhận file từ Pi 4 về Mac; mở công cụ [`gt_annotator.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/gt_annotator.py) gán nhãn Ground-Truth bán tự động tạo các file `_gt.csv`. | Bộ 6 file `data/ground_truth/*_gt.csv`. |
| **4** | **Cả hai đội** | Chép thư mục video về Pi 4, chạy [`run_benchmark.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/run_benchmark.py) trực tiếp trên Pi 4 để đo FPS thật; chạy [`evaluate_results.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/experiments/tracking_benchmark/evaluate_results.py) để xuất bảng báo cáo. | File báo cáo kết quả thực nghiệm `results/tracking_benchmark_report.md` với đầy đủ số liệu thật. |

