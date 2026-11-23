# HƯỚNG DẪN LẮP ĐẶT VÀ CÂN CHỈNH PHẦN CỨNG (CAMERA & HC-SR04)

**Đề tài:** Hệ thống Hỗ trợ Người Khiếm thị Thông minh trên Thiết bị Biên Raspberry Pi 4 (PBL6)  
**Tài liệu:** Hướng dẫn đấu nối mạch điện tử, gá lắp cơ khí đồng trục và quy trình chẩn đoán

---

## MỤC LỤC
1. [DANH SÁCH LINH KIỆN VÀ VẬT TƯ CẦN THIẾT](#1-danh-sách-linh-kiện-và-vật-tư-cần-thiết)
2. [SƠ ĐỒ ĐẤU NỐI MẠCH ĐIỆN TỬ (WIRING DIAGRAM)](#2-sơ-đồ-đấu-nối-mạch-điện-tử-wiring-diagram)
   - 2.1. Cảnh báo an toàn điện áp Raspberry Pi 4 (5V vs 3.3V)
   - 2.2. Mạch cầu phân áp trở bảo vệ chân Echo (Voltage Divider)
   - 2.3. Bảng tra cứu chân GPIO chi tiết (Pinout Table)
3. [GIẢI PHÁP GÁ LẮP CƠ KHÍ ĐỒNG TRỤC (MECHANICAL ALIGNMENT)](#3-giải-pháp-gá-lắp-cơ-khí-đồng-trục-mechanical-alignment)
   - 3.1. Nguyên tắc vàng: Cùng hướng và Đồng trục đứng
   - 3.2. Căn chỉnh góc nghiêng quang học ($\alpha \approx 5^\circ$)
   - 3.3. Các phương án chế tạo giá đỡ (DIY Formex/Mica & In 3D)
   - 3.4. Vị trí đeo trên cơ thể người khiếm thị
4. [QUY TRÌNH KIỂM THỬ VÀ CHẨN ĐOÁN SAU KHI LẮP RÁP](#4-quy-trình-kiểm-thử-và-chẩn-đoán-sau-khi-lắp-ráp)
5. [CÁC SỰ CỐ THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC (TROUBLESHOOTING)](#5-các-sự-cố-thường-gặp-và-cách-khắc-phục-troubleshooting)

---

## 1. DANH SÁCH LINH KIỆN VÀ VẬT TƯ CẦN THIẾT

| STT | Thiết bị / Linh kiện | Thông số kỹ thuật khuyến nghị | Số lượng | Ghi chú |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Raspberry Pi 4 Model B** | RAM 4GB hoặc 8GB, nguồn $5\text{V}/3\text{A}$ Type-C | 1 bộ | Board điều khiển trung tâm |
| **2** | **Cảm biến Siêu âm HC-SR04** | Điện áp hoạt động: $5\text{V}$, Tần số: $40\text{kHz}$ | 1 cái | Đo khoảng cách tầm gần |
| **3** | **Webcam USB hoặc CSI Camera** | Độ phân giải 720p/1080p, góc nhìn (FOV) $\ge 70^\circ$ | 1 cái | Camera thu nhận hình ảnh |
| **4** | **Điện trở $1\text{k}\Omega$** ($1000\,\Omega$) | Công suất $1/4\text{W}$ (Sai số 1% hoặc 5%) | 1 con | Làm cầu phân áp chân Echo |
| **5** | **Điện trở $2\text{k}\Omega$** ($2000\,\Omega$) | Công suất $1/4\text{W}$ (có thể dùng $2.2\text{k}\Omega$) | 1 con | Kéo xuống mass (GND) |
| **6** | **Dây bẹ Breadboard (Dupont)** | Dây Đực - Cái (Male to Female) dài $20\text{cm}$ | 6 - 8 sợi | Đấu nối GPIO |
| **7** | **Breadboard mini hoặc Test board** | Loại 170 lỗ mini hoặc mạch đồng hàn cố định | 1 cái | Cố định điện trở phân áp |
| **8** | **Vật liệu làm giá gá đỡ** | Tấm Formex dày $3\text{mm}$, bìa Mica, hoặc In 3D | 1 bộ | Cố định Camera và HC-SR04 |
| **9** | **Dây đai đeo ngực (Chest Strap)** | Loại dây đai co giãn cho Action Cam / GoPro | 1 bộ | Đeo cố định trước ngực |

---

## 2. SƠ ĐỒ ĐẤU NỐI MẠCH ĐIỆN TỬ (WIRING DIAGRAM)

### 2.1. Cảnh báo an toàn điện áp Raspberry Pi 4 (5V vs 3.3V)

> [!CAUTION]
> **CẢNH BÁO NGUY HIỂM CHÁY CHÂN GPIO:**  
> Các chân GPIO của Raspberry Pi 4 chỉ hoạt động ở mức logic **$3.3\text{V}$**. Chúng **KHÔNG chịu được mức điện áp $5\text{V}$**.  
> Cảm biến HC-SR04 chạy nguồn $5\text{V}$, do đó chân **ECHO** sẽ xuất tín hiệu xung ở mức **$5\text{V}$**.  
> **NẾU CẮM TRỰC TIẾP CHÂN ECHO VÀO GPIO CỦA PI, CHIP GPIO SẼ BỊ CHÁY HỎNG NGAY LẬP TỨC!**  
> Bắt buộc phải đấu qua mạch cầu phân áp điện trở dưới đây.

---

### 2.2. Mạch cầu phân áp trở bảo vệ chân Echo (Voltage Divider)

Nguyên lý cầu phân áp sử dụng 2 điện trở $R_1 = 1\text{k}\Omega$ và $R_2 = 2\text{k}\Omega$:
$$V_{GPIO} = V_{ECHO} \times \frac{R_2}{R_1 + R_2} = 5\text{V} \times \frac{2\text{k}\Omega}{1\text{k}\Omega + 2\text{k}\Omega} = 5\text{V} \times \frac{2}{3} \approx 3.33\text{V}$$
Mức điện áp $3.33\text{V}$ hoàn toàn an toàn và nằm đúng chuẩn ngưỡng logic HIGH của vi xử lý Broadcom BCM2711 trên Raspberry Pi 4.

```
                    +--------------------+
                    |  HC-SR04 ULTRASONIC|
                    +--------------------+
                      VCC   TRIG  ECHO  GND
                       |     |     |     |
      5V (Pin 2/4) ----+     |     |     |
                             |     |     |
      GPIO 23 (Pin 16) ------+     |     |
                                   |     |
                                [1k Ohm] |
                                   |     |
      GPIO 24 (Pin 18) <-----------+     |
                                   |     |
                                [2k Ohm] |
                                   |     |
      GND (Pin 6/9/14/20) ---------+-----+
```

*Nếu không có trở $2\text{k}\Omega$, bạn có thể dùng trở $2.2\text{k}\Omega$ ($V_{out} = 3.4\text{V}$) hoặc mắc nối tiếp hai điện trở $1\text{k}\Omega$ ($1\text{k}\Omega + 1\text{k}\Omega = 2\text{k}\Omega$).*

---

### 2.3. Bảng tra cứu chân GPIO chi tiết (Pinout Table)

Đối chiếu với cấu hình mặc định trong [`configs/config.yaml`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/configs/config.yaml#L23-L30):

| Chân trên HC-SR04 | Kết nối trung gian | Chân trên Header 40-Pin Raspberry Pi 4 | Tên chức năng (Broadcom BCM) |
| :--- | :--- | :---: | :--- |
| **VCC** | Nối trực tiếp dây cái - cái | **Pin 2** (hoặc Pin 4) | $5\text{V}$ Power |
| **GND** | Nối chung với chân dưới của $R_2$ | **Pin 6** (hoặc Pin 9, 14, 20) | Ground ($0\text{V}$) |
| **TRIG** | Nối trực tiếp dây cái - cái | **Pin 16** | **GPIO 23** (Output từ Pi sang cảm biến) |
| **ECHO** | Đi vào đầu trở $R_1$ ($1\text{k}\Omega$) | Điểm giữa $R_1$ và $R_2$ nối vào **Pin 18** | **GPIO 24** (Input từ cảm biến về Pi) |

*(Chân TRIG là tín hiệu xuất từ Pi ở mức $3.3\text{V}$, cảm biến HC-SR04 nhận diện mức $> 2.5\text{V}$ là mức HIGH nên chân TRIG cắm trực tiếp được, không cần trở).*

---

## 3. GIẢI PHÁP GÁ LẮP CƠ KHÍ ĐỒNG TRỤC (MECHANICAL ALIGNMENT)

### 3.1. Nguyên tắc vàng: Cùng hướng và Đồng trục đứng

Trong phần mềm, ma trận hợp nhất cảm biến [`fusion_engine.py`](file:///Users/nguyenthithuylinh/Downloads/PBL6/blind-navigation/2_edge_device/core/fusion_engine.py) hoạt động dựa trên giả thiết: **Tâm chùm sóng siêu âm $30^\circ$ trùng với tâm trục quang học camera ($c_x = 320\text{px}$)**.

```
       [ ĐÚNG: ĐỒNG TRỤC ĐỨNG ]                     [ SAI: LỆCH TRỤC NGANG ]
       +-----------------------+                   +---------+     +---------+
       |     [ CAMERA LENS ]   |                   | HC-SR04 |     | CAMERA  |
       |           (o)         |                   |  (o)(o) | <-> |   (o)   |
       +-----------------------+                   +---------+     +---------+
       |   HC-SR04 ULTRASONIC  |                   Khoảng cách lệch d_offset > 10cm
       |     ( O )   ( O )     |                   sẽ làm lệch điểm ngắm tâm!
       +-----------------------+
```

* **Yêu cầu 1: Đặt HC-SR04 ngay dưới (hoặc ngay trên) Camera:** Giữ tâm ngang trùng nhau ($X_{offset} \approx 0$).
* **Yêu cầu 2: Hai thiết bị phải cùng nhìn về một hướng:** Không được để camera nhìn chếch sang trái trong khi cảm biến siêu âm nhìn chếch sang phải.

---

### 3.2. Căn chỉnh góc nghiêng quang học ($\alpha \approx 5^\circ$)

Theo công thức hình học mặt phẳng sàn (Ground-plane Geometry):
$$d_{ground} = \frac{h_{cam}}{\tan(\alpha + \theta_y)}$$
Mô hình toán học trong file cấu hình đã được hiệu chỉnh tối ưu ở thông số:
* Độ cao gắn thiết bị: $h_{cam} = 1.40\text{m} - 1.45\text{m}$ (ngang xương ức ngực người trưởng thành).
* Góc nghiêng chúi xuống: $\alpha \approx 5^\circ - 7^\circ$ so với phương ngang thẳng đứng.

```
                     Trục phương ngang (Horizontal)
                      ------------------------
                     \
                      \  Góc nghiêng alpha = 5 độ
                       \
                        \  Trục ngắm Camera + Siêu âm
                         \
                          v
                          Vùng quan sát tối ưu: 1.0m đến 4.5m trên mặt sàn
```

* **Mẹo cân góc $5^\circ$ thực tế:** Bạn đặt thiết bị đứng thẳng vuông góc, sau đó chêm nhẹ một miếng đệm mỏng khoảng $2\text{mm} - 3\text{mm}$ ở phía trên đỉnh của camera để cụm cảm biến hơi chúc nhẹ xuống đất. Góc nghiêng nhẹ này giúp camera vừa bao quát được thân người đi đối diện, vừa nhìn thấy chân bàn ghế trên mặt sàn.

---

### 3.3. Các phương án chế tạo giá đỡ (Mounting Options)

#### Phương án 1: Tự chế nhanh bằng tấm Formex $3\text{mm}$ hoặc Bìa Mica (Khuyên dùng thử nghiệm)
1. Cắt một miếng Formex chữ nhật kích thước: **Rộng $60\text{mm} \times$ Cao $90\text{mm}$**.
2. Trên nửa dưới: Dùng compa khoét 2 lỗ tròn đường kính **$16\text{mm}$**, tâm 2 lỗ cách nhau **$26\text{mm}$** (vừa khít 2 mắt nhôm của HC-SR04).
3. Đẩy 2 mắt của HC-SR04 xuyên qua 2 lỗ, dùng súng bắn keo nến cố định mặt sau.
4. Trên nửa trên: Đặt Webcam/Camera sao cho ống kính nằm thẳng hàng trên đường trung trực của 2 mắt siêu âm. Dán bằng băng keo 2 mặt 3M siêu dính hoặc dây rút (zip tie).

```
          +-----------------------+  ^
          |                       |  |
          |     [ CAMERA LENS ]   |  | 45mm
          |           (o)         |  |
          +-----------------------+  v
          |   ( O )       ( O )   |  | 45mm
          |  (16mm)  26mm  (16mm) |  |
          +-----------------------+  v
          <-------- 60mm --------->
```

#### Phương án 2: In 3D (3D Printing Bracket) cho sản phẩm hoàn thiện
* Nếu trường có máy in 3D, bạn có thể thiết kế một cụm gá (Bracket) tích hợp ngàm gài chuẩn GoPro/Action Cam ở phía sau:
  * Khoảng cách 2 lỗ cảm biến: Đường kính $16.2\text{mm}$, khoảng cách tâm $26.0\text{mm}$.
  * Khoang camera: Kẹp giữ vừa thân webcam.
  * Mặt sau: Có khe luồn dây bẹ và rãnh ngàm gắn vào dây đai ngực.

---

### 3.4. Vị trí đeo trên cơ thể người khiếm thị

Có 3 vị trí đeo thực nghiệm:

```
[ VỊ TRÍ 1: ĐAI NGỰC (CHEST STRAP) ] -> KHUYÊN DÙNG TỐT NHẤT (PBL6)
- Cố định vào ngực bằng đai co giãn.
- Ưu điểm: Độ cao ổn định (1.40m - 1.45m), camera ít bị lắc lư đảo góc khi quay đầu.
- Rất phù hợp với công thức Ground-plane Flat World.

[ VỊ TRÍ 2: QUAI ĐEO BALO ] -> TỐT THỨ HAI
- Kẹp gá vào một bên quai đeo balo trước ngực.
- Chú ý: Cần siết chặt quai để cụm cảm biến không bị lệch xoay sang một bên.

[ VỊ TRÍ 3: MŨ / TRÁN ] -> KHÔNG NÊN DÙNG CHO CẢM BIẾN SIÊU ÂM
- Đầu người quay liên tục qua lại làm chùm nón siêu âm quét loạn xạ, gây sai lệch cự ly bám vết.
```

---

## 4. QUY TRÌNH KIỂM THỬ VÀ CHẨN ĐOÁN SAU KHI LẮP RÁP

Sau khi hàn và cắm dây xong, tuyệt đối **không chạy `main.py` ngay**, hãy thực hiện quy trình kiểm tra từng bước theo kịch bản:

### Bước 1: Kiểm tra mạch điện và dây cắm (Cold Check)
* Đảm bảo dây GND của Pi đã nối thông với GND của HC-SR04 và chân dưới điện trở $R_2$.
* Đảm bảo chân ECHO đi qua điện trở $1\text{k}\Omega$ trước khi chạm vào GPIO 24.
* Cắm USB Webcam vào **cổng USB 3.0 màu xanh** của Raspberry Pi 4 để đảm bảo đủ băng thông truyền hình ảnh.

### Bước 2: Chạy công cụ kiểm tra độc lập phần cứng
Mở Terminal trên Pi (hoặc qua SSH) và chạy script chẩn đoán chuyên dụng:
```bash
cd ~/Documents/pbl6/blind-navigation/2_edge_device
python experiments/test_hardware_pi.py
```

*Kết quả mong đợi trên màn hình:*
```
[1/3] Kiểm tra Âm thanh... ✅ espeak-ng hoạt động tốt.
[2/3] Kiểm tra Camera...   ✅ Mở camera thành công! Độ phân giải: 640x480
[3/3] Kiểm tra Cảm biến HC-SR04...
    Lần 1: 125.4 cm (1.25 m)
    Lần 2: 124.8 cm (1.25 m)
    Lần 3: 125.1 cm (1.25 m)
✅ Cảm biến HC-SR04 đọc xung thành công!
```

### Bước 3: Chạy thước đo khoảng cách trực quan thời gian thực
Chạy công cụ đo khoảng cách siêu âm có thanh đo đồ họa:
```bash
python experiments/test_ultrasonic.py
```
* Đưa bàn tay lại gần từ $1\text{m} \to 0.5\text{m} \to 0.2\text{m}$.
* Quan sát thanh tiến trình trên màn hình co ngắn lại chính xác theo khoảng cách tay bạn.

### Bước 4: Kiểm tra đồng bộ hình ảnh Camera và Âm thanh Mac
```bash
python experiments/test_camera_mac_audio.py
```
* Mở trình duyệt trên máy Mac: `http://<IP_RASPBERRY_PI>:8080`.
* Bấm **[ 🔊 BẬT ÂM THANH TRÌNH DUYỆT ]**.
* Đứng trước camera ở khoảng cách $2\text{m}$: Quan sát hình ảnh mượt mà trên Chrome và lắng nghe loa Mac đọc: *"Người trước mặt"*.

---

## 5. CÁC SỰ CỐ THƯỜNG GẶP VÀ CÁCH KHẮC PHỤC (TROUBLESHOOTING)

| Hiện tượng lỗi | Nguyên nhân gốc rễ | Cách xử lý triệt để |
| :--- | :--- | :--- |
| **Khoảng cách siêu âm luôn báo $0.0\text{ cm}$ hoặc `None`** | 1. Dây ECHO hoặc TRIG cắm nhầm chân GPIO.<br/>2. Cầu phân áp bị hở mass (GND chưa tiếp xúc).<br/>3. Cảm biến bị cháy do trước đó cắm trực tiếp $5\text{V}$ vào GPIO. | 1. Kiểm tra lại chân: Pin 16 (GPIO 23), Pin 18 (GPIO 24).<br/>2. Dùng đồng hồ VOM đo thông mạch chân GND.<br/>3. Thay thử một module HC-SR04 mới. |
| **Camera báo lỗi `can't open camera by index`** | 1. Cổng USB camera tiếp xúc lỏng.<br/>2. Cổng index không phải là 0 mà là 1, 2. | 1. Rút camera cắm sang cổng USB 3.0 khác.<br/>2. Chạy lệnh: `v4l2-ctl --list-devices` để xem danh sách cổng video khả dụng. |
| **Số đo siêu âm bị nhảy loạn xạ (lúc $50\text{cm}$, lúc $200\text{cm}$)** | 1. Mặt phát siêu âm hướng vào vật liệu tiêu âm (rèm vải, áo len bông).<br/>2. Cảm biến bị rung lắc khi bước đi. | 1. Thử nghiệm trước một bức tường phẳng hoặc cánh cửa gỗ cứng.<br/>2. Siết chặt dây đai ngực để cảm biến cố định vào thân người. |
| **Báo lỗi `RuntimeWarning: This channel is already in use`** | Lần chạy trước tắt bằng Ctrl+C đột ngột khiến chân GPIO chưa kịp giải phóng. | Mã nguồn đã được thêm lệnh `GPIO.setwarnings(False)`. Bạn có thể bỏ qua cảnh báo này hoặc chạy lệnh: `python -c "import RPi.GPIO as G; G.setmode(G.BCM); G.cleanup()"`. |

