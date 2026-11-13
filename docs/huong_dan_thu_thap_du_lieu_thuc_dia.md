# CẨM NANG HIỆN TRƯỜNG: HƯỚNG DẪN THU THẬP TẬP DỮ LIỆU VIDEO CHO THỰC NGHIỆM ĐỐI SÁNH OBJECT TRACKING (PBL6)

> [!IMPORTANT]
> **Dành riêng cho Đội Thu thập Dữ liệu Phần cứng (Field Data Team):** 
> - Tài liệu này là **Quy trình Thao tác Chuẩn (SOP - Standard Operating Procedure)** để đi quay dữ liệu thực tế tại hành lang.
> - **Chỉ tập trung 100% vào bài toán Bám vết Đối tượng (Object Tracking Benchmark)** thông qua Camera. **Hoàn toàn KHÔNG sử dụng và KHÔNG cần mang theo cảm biến siêu âm**.
> - Đội hiện trường chỉ cần thực hiện đúng các thao tác bấm máy, tuân thủ kịch bản và bàn giao đúng định dạng video chuẩn để Đội Thuật toán chạy Replay đối sánh 4 cấu hình: **Raw YOLO (Không tracker)**, **SORT**, **DeepSORT**, và **Edge ByteTrack**.

---

## 1. Giải Đáp Quy Chuẩn Khoa Học: Thời Lượng & Số Lần Chạy Lặp (Runs)

Trước khi ra hiện trường, đội ngũ cần nắm rõ 2 nguyên lý khoa học cốt lõi chi phối quy cách quay video:

### 1.1. Thời lượng mỗi video: 15 giây có đủ không? Tại sao quy chuẩn phải là 25 – 30 giây?
- **Hạn chế của thời lượng 15 giây ($15\text{ FPS} \times 15\text{s} = 225\text{ frames}$):**
  - Tốc độ di chuyển thực tế của người khiếm thị khi đi trong hành lang là khoảng $0.6 - 0.8\text{ m/s}$. Chiều dài hành lang thực tế từ lúc bắt đầu nhìn thấy vật cản đến lúc đi ngang qua thường từ $10\text{m} - 15\text{m}$.
  - Nếu chỉ quay 15 giây, người mang thiết bị chỉ bước được khoảng 8 – 10 bước ($5 - 6\text{m}$). Người thử nghiệm sẽ phải bước rất vội vàng, hoặc video sẽ bị ngắt giữa chừng khi vừa mới tiến đến gần vật cản, **làm đứt đoạn toàn bộ vòng đời của một Track** (khởi tạo $\rightarrow$ bám vết ổn định $\rightarrow$ che khuất/rung lắc $\rightarrow$ phục hồi vết $\rightarrow$ vượt qua vật cản).
  - Trong các tập dữ liệu chuẩn quốc tế về bám vết đa đối tượng (MOT Challenge như MOT16, MOT17, MOT20), các chuỗi video đánh giá đều kéo dài từ $20\text{s} - 60\text{s}$ để đảm bảo độ tin cậy của chỉ số bám vết dài hạn $IDF_1$.
- **Quy chuẩn chính thức được chọn: 25 đến 30 giây ($\approx 375 - 450\text{ frames}$ ở 15 FPS):**
  - Đây là **"điểm ngọt" (Sweet Spot)**: Đủ dài để người mang thiết bị đi thong thả từ xa tiếp cận vật cản, lách qua vật cản hoặc để người đi ngang qua che khuất hoàn toàn; đồng thời không quá dài để tránh làm nặng bộ nhớ Raspberry Pi 4 và tránh làm Đội Gán nhãn Ground-Truth bị quá tải (gán nhãn 400 frames bằng công cụ bán tự động mất khoảng 15 phút/video).

### 1.2. Mỗi kịch bản cần quay lặp lại bao nhiêu lần (Runs)?
- **Nguyên lý phương sai thống kê:**
  - Nếu chỉ quay 1 lần duy nhất, kết quả đo đạc sẽ mang nặng yếu tố ngẫu nhiên (chẳng may bước hụt chân rung lắc bất thường, hoặc góc quay vô tình quá lý tưởng). Hội đồng khoa học hoặc giảng viên phản biện sẽ chất vấn về độ lặp lại (reproducibility) của thực nghiệm.
- **Quy chuẩn chạy lặp chính thức:**
  - **Tối thiểu: 3 lần lặp lại độc lập (`Run 1`, `Run 2`, `Run 3`)** cho mỗi kịch bản.
  - **Lý tưởng: 5 lần lặp** đối với các kịch bản quan trọng cốt lõi (như che khuất `scen4` hoặc rung lắc `scen8`).
  - **Mục tiêu đầu ra:** Cho phép đội xử lý tính toán giá trị **Trung bình (Mean)** và **Độ lệch chuẩn (Std Dev)** của các chỉ số: $\text{MOTA} \pm \sigma$, $\text{IDF1} \pm \sigma$, $\text{FPS} \pm \sigma$, $\text{IDSW} \pm \sigma$.
  - **Quy ước đặt tên tệp đồng nhất:**  
    `<mã_kịch_bản>_run1.mp4`, `<mã_kịch_bản>_run2.mp4`, `<mã_kịch_bản>_run3.mp4`

---

## 2. Danh Sách Dụng Cụ Cần Chuẩn Bị (Checklist Hiện Trường)

Đội phần cứng chỉ cần mang theo đúng các vật dụng sau:

- [ ] **01 Cụm Raspberry Pi 4 (kèm Camera):**
  - Bo mạch Raspberry Pi 4 Model B (4GB hoặc 8GB RAM).
  - Camera USB (Logitech C270/C920/Webcam 720p) hoặc Camera CSI gắn cáp ruy-băng.
  - Ngàm đeo ngực (Chest Harness) hoặc giá kẹp áo chắc chắn.
  - *(Lưu ý: Không cần cắm hay nối dây cảm biến siêu âm HC-SR04)*.
- [ ] **01 Sạc dự phòng chất lượng tốt:** Đầu ra chuẩn Type-C, dòng tối thiểu $5\text{V} / 3\text{A}$ (tránh sụt áp làm Pi 4 giảm xung nhịp khi ghi video).
- [ ] **01 Thước dây (3m hoặc 5m):** Dùng để xác định cự ly ban đầu giữa người xuất phát và vật cản.
- [ ] **01 Cuộn băng dính màu:** Đánh dấu vạch xuất phát trên sàn hành lang.
- [ ] **01 Điện thoại phát Hotspot 4G + Laptop:**
  - Bật Hotspot để Pi 4 tự nhận IP nội bộ.
  - Dùng Laptop hoặc app Termius trên điện thoại để SSH vào Pi 4 phát lệnh quay.
- [ ] **Vật cản mẫu kiểm thử:**
  - 01 Ghế tựa văn phòng hoặc ghế nhựa (cao $\approx 0.8\text{m}$, màu sắc tương phản với tường).
  - 01 Thùng carton hoặc ba lô học sinh to.
  - 01 đến 02 Thành viên đi cùng đóng vai chướng ngại vật di động (người đi bộ).

---

## 3. Căn Chỉnh Góc Nhìn Camera Trước Khi Quay (3 Phút)

Để tập video thu về tương thích hoàn toàn với thuật toán YOLO và bộ lọc mặt sàn:

```
                          GÓC ĐẶT CAMERA TRÊN NGƯỜI ĐEO
                          
                [Người Đeo Thiết Bị]
                    (Mắt nhìn)
                        │
                  1.40m │       [Camera]
                  đến   │        \ 
                  1.45m │         \  Trục quang học chúc nhẹ xuống sàn 5 độ
                        │          \ (Góc chúc alpha = 5°)
                        │           \
    ────────────────────┴────────────\───────────────────── Mặt sàn hành lang
                                      \___> Điểm nhìn trung tâm sàn cách 5m - 8m
```

1. **Độ cao ngàm:** Đeo ngàm camera ngang tầm xương ức của người mang ($1.40\text{m} - 1.45\text{m}$ tính từ mặt sàn).
2. **Góc chúc $5^\circ$:** Chêm một miếng đệm mỏng ở mép trên của camera để ống kính hơi chúc nhẹ xuống đất khoảng $5^\circ$ (không chúc quá sâu khiến mất tầm nhìn xa, không ngửa lên trần nhà).
3. **Cố định ngàm:** Siết chặt ốc giữ camera để camera không bị xộc xệch, xoay lệch trái/phải trong khi bước đi.

---

## 4. Hệ Thống 12 Kịch Bản Thực Nghiệm Đánh Số Tuần Tự (Từ 1 Đến 12)

12 kịch bản được chia làm 4 nhóm thử thách và **được đánh số thứ tự liên tục từ 1 đến 12** để đội hiện trường dễ theo dõi, không bị nhầm lẫn:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               HỆ THỐNG 12 KỊCH BẢN BENCHMARK TRACKING (ĐÁNH SỐ LIÊN TỤC 1 -> 12)       │
├────────────────────────┬─────────────────────────┬─────────────────────────────────────┤
│ Nhóm I: Cơ bản &       │ Nhóm II: Che khuất &    │ Nhóm III: Chuyển động người đeo     │
│ Đường đi đơn giản      │ Động lực học đối tượng  │ (Ego-motion Challenges)             │
│ ────────────────────── │ ─────────────────────── │ ─────────────────────────────────── │
│ 1. scen1_static        │ 4. scen4_occlusion      │ 7. scen7_edge_to_center             │
│ 2. scen2_multi         │ 5. scen5_cross_paths    │ 8. scen8_shake (rung lắc)           │
│ 3. scen3_empty (trống) │ 6. scen6_reenter        │ 9. scen9_fast_approach (tiếp cận)   │
│                        │                         │ 10. scen10_turn_corner (rẽ cua)     │
├────────────────────────┴─────────────────────────┴─────────────────────────────────────┤
│ Nhóm IV: Điều kiện môi trường khắc nghiệt & Mật độ dày đặc                             │
│ ────────────────────────────────────────────────────────────────────────────────────── │
│ 11. scen11_lighting_change (ngược sáng / tối sang sáng)                                │
│ 12. scen12_dense_clutter (hành lang đông đúc, mật độ vật thể dày đặc)                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Mỗi kịch bản dưới đây đều được quay **tối thiểu 3 lần (`--run 1`, `--run 2`, `--run 3`)**, mỗi lần **25 giây** ở **15 FPS**.

---

### NHÓM I: CƠ BẢN & ĐƯỜNG ĐI ĐƠN GIẢN (KỊCH BẢN 1, 2, 3)

#### Kịch bản 1 (`scen1_static`): Đi thẳng qua hành lang có 1 vật cản đứng yên
- **Mục tiêu kỹ thuật:** Đo lường độ ổn định của ID track đơn lẻ. Thuật toán tốt phải giữ đúng 1 ID duy nhất từ khi phát hiện ở xa đến khi tới gần ($IDSW = 0$).
- **Bố trí:** Đặt 1 chiếc ghế ở giữa hành lang, cách vạch xuất phát đúng $3.5\text{m}$.
- **Thao tác người mang thiết bị:**
  - Đứng tại vạch xuất phát, hướng thẳng vào ghế.
  - Khi bắt đầu quay: Bước đi thẳng thong thả ($0.6 - 0.8\text{ m/s}$) về phía ghế. Khi cách ghế khoảng $0.8\text{m}$ thì dừng lại đứng yên cho đến khi hết 25 giây.
- **Lệnh thực hiện (chạy 3 lần lặp):**
  ```bash
  python3 record_scenarios.py --scenario scen1_static --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen1_static --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen1_static --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 2 (`scen2_multi`): Đi qua nhiều vật cản cách nhau so le
- **Mục tiêu kỹ thuật:** Kiểm tra thuật toán gán cặp Hungarian khi có nhiều đối tượng đồng thời trong tầm nhìn; đảm bảo không bị đổi chéo ID giữa các vật cản khác nhau.
- **Bố trí:**
  - Vật 1 (Ghế): Cách vạch xuất phát $2.0\text{m}$, đặt lệch về mép hành lang bên trái $0.5\text{m}$.
  - Vật 2 (Thùng carton/ba lô): Cách vạch xuất phát $3.5\text{m}$, đặt lệch về mép hành lang bên phải $0.5\text{m}$.
- **Thao tác:** Người đeo bước đi chậm theo đường ziczac nhẹ ở giữa hành lang, lần lượt vượt qua vật 1 rồi tiến đến vật 2.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen2_multi --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen2_multi --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen2_multi --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 3 (`scen3_empty`): Hành lang hoàn toàn trống không có vật cản
- **Mục tiêu kỹ thuật:** Đo lường **Tỷ lệ Cảnh báo Nhầm (False Alert Rate)** do ánh sáng phản chiếu trên sàn gạch, gờ tường hoặc hoa văn gây ra. Với kịch bản này, Ground-Truth là rỗng, mọi phát hiện tạo ra âm thanh đều bị tính là lỗi.
- **Bố trí:** Chọn đoạn hành lang dài $15\text{m} - 20\text{m}$ hoàn toàn trống trải, không có người qua lại, không có đồ đạc.
- **Thao tác:** Người đeo đi bộ thẳng từ đầu này đến đầu kia của hành lang trong suốt 25 giây.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen3_empty --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen3_empty --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen3_empty --run 3 --duration 25 --source auto --headless
  ```

---

### NHÓM II: CHE KHUẤT & ĐỘNG LỰC HỌC ĐỐI TƯỢNG (KỊCH BẢN 4, 5, 6)

#### Kịch bản 4 (`scen4_occlusion`): Người đi ngang che khuất vật thể tĩnh trong thời gian ngắn
> [!WARNING]
> **Điểm mấu chốt kỹ thuật:** Pha che khuất phải kéo dài **$1.0\text{–}1.5\text{ giây}$** (khoảng 15 đến 22 khung hình). Đây là ranh giới sống còn chứng minh ByteTrack vượt trội hơn SORT (SORT bị mất dấu sau khi đứt detection, trong khi ByteTrack cứu lại track ở tầng 2 nhờ $D_{\text{low}}$).
- **Bố trí:** Chiếc ghế đặt cố định cách $2.5\text{m}$. Một người trợ lý đứng nép sát tường ở cự ly $1.5\text{m}$.
- **Thao tác:**
  - Người đeo thiết bị đứng yên tại chỗ nhìn vào chiếc ghế.
  - Ở giây thứ 8: Người trợ lý bước dứt khoát cắt ngang qua giữa camera và chiếc ghế, che khuất một phần thân ghế trong khoảng $1.0 - 1.5\text{s}$, rồi bước hẳn sang bờ tường đối diện.
  - Người đeo tiếp tục giữ yên hướng nhìn vào ghế cho đến hết 25 giây.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen4_occlusion --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen4_occlusion --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen4_occlusion --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 5 (`scen5_cross_paths`): Hai người đi ngược chiều nhau đan chéo cắt mặt trước camera
- **Mục tiêu kỹ thuật:** Kiểm tra hiện tượng hoán đổi ID (ID Switch) giữa 2 đối tượng cùng loại (`person`) khi quỹ đạo của họ giao nhau. Đây là điểm yếu kinh điển của SORT dựa trên vị trí thuần túy.
- **Bố trí:** Hai trợ lý A và B đứng cách camera khoảng $4.0\text{m}$, cách nhau $2\text{m}$ bề ngang.
- **Thao tác:**
  - Người đeo thiết bị đứng yên.
  - Khi bắt đầu quay: Trợ lý A đi chéo từ trái sang phải, trợ lý B đi chéo từ phải sang trái. Hai người đi ngang qua nhau ở vị trí cách camera khoảng $2.0\text{m}$ tạo ra sự che khuất chéo nhau, rồi tiếp tục đi về hai phía đối diện.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen5_cross_paths --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen5_cross_paths --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen5_cross_paths --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 6 (`scen6_reenter`): Đối tượng rời khỏi khung hình rồi xuất hiện trở lại sau 1-2 giây
- **Mục tiêu kỹ thuật:** Kiểm tra vòng đời của Track khi bị xóa (`max_age = 12` frames $\approx 0.8\text{s}$). Nếu đối tượng rời khung hình quá lâu và quay lại, thuật toán cần tạo ID mới một cách dứt khoát, không sinh ra lỗi gán sai vào các track lân cận.
- **Bố trí:** Một người trợ lý đứng cách camera $3.0\text{m}$.
- **Thao tác:**
  - Giây 0–6: Trợ lý đứng trong tầm nhìn camera.
  - Giây 7–10: Trợ lý bước hẳn ra khỏi rìa khung hình (đi vào cửa phòng hoặc nấp sau cột tường) trong khoảng 2–3 giây.
  - Giây 11: Trợ lý bước trở lại vào giữa hành lang và đứng yên đến hết 25 giây.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen6_reenter --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen6_reenter --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen6_reenter --run 3 --duration 25 --source auto --headless
  ```

---

### NHÓM III: CHUYỂN ĐỘNG NGƯỜI ĐEO & RUNG LẮC CAMERA (KỊCH BẢN 7, 8, 9, 10)

#### Kịch bản 7 (`scen7_edge_to_center`): Vật thể từ rìa ảnh đi vào vùng chính giữa
- **Mục tiêu kỹ thuật:** Kiểm tra khả năng phát hiện ở vùng biên (nơi bounding box chỉ nhìn thấy một phần) và khởi tạo track mới chuẩn xác (`min_hits = 2`, `conf >= 0.45`).
- **Bố trí:** Đặt 1 chiếc ghế ở cự ly $2.5\text{m}$.
- **Thao tác:**
  - Người đeo đứng quay người chếch một góc $30^\circ$, sao cho chiếc ghế nằm sát mép ngoài cùng của khung hình (chỉ thấy 1 nửa chiếc ghế).
  - Giữ nguyên 5 giây đầu.
  - Từ giây thứ 6: Từ từ xoay thân người đưa chiếc ghế vào đúng trung tâm khung hình, sau đó bước thẳng tiến lại gần ghế.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen7_edge_to_center --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen7_edge_to_center --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen7_edge_to_center --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 8 (`scen8_shake`): Camera rung lắc khi người đeo bước đi nhanh, dằn gót
- **Mục tiêu kỹ thuật:** Kiểm tra độ bền vững của tracker trước hiện tượng nhòe chuyển động (motion blur) và bounding box bị giật tọa độ liên tục theo chu kỳ bước chân.
- **Bố trí:** Đặt chiếc ghế ở cự ly $4.0\text{m}$.
- **Thao tác:**
  - Người đeo bước đi nhanh hơn bình thường ($1.0 - 1.2\text{ m/s}$), cố ý dằn gót chân xuống sàn và hơi lắc nhẹ vai theo từng nhịp bước chân để tạo độ rung thực tế khi di chuyển vội. Tiến thẳng về phía vật cản.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen8_shake --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen8_shake --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen8_shake --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 9 (`scen9_fast_approach`): Vật cản tiếp cận nhanh về phía camera
- **Mục tiêu kỹ thuật:** Kiểm tra giới hạn bám vết khi kích thước bounding box nở rộng rất nhanh qua từng khung hình (tốc độ giãn nở diện tích lớn, IoU giữa 2 frame liên tiếp bị kéo dãn).
- **Bố trí:** Người đeo đứng yên tại chỗ. Trợ lý đứng cách $6.0\text{m}$.
- **Thao tác:**
  - Trợ lý bước đi nhanh hoặc đẩy một xe đẩy/thùng carton tiến thẳng về phía người đeo thiết bị với tốc độ cao ($\approx 1.5\text{ m/s}$), dừng lại khi cách camera $1.0\text{m}$.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen9_fast_approach --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen9_fast_approach --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen9_fast_approach --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 10 (`scen10_turn_corner`): Rẽ góc ngã rẽ hành lang 90 độ, vật cản bất ngờ xuất hiện
- **Mục tiêu kỹ thuật:** Đo lường độ trễ phản hồi của hệ thống khi vật cản bất ngờ xuất hiện toàn bộ trong tầm nhìn (độ trễ tích lũy $\text{Latency} = \text{min\_hits} \times \Delta t_{\text{frame}}$).
- **Bố trí:** Chọn góc ngã rẽ hành lang hình chữ L. Đặt một chiếc ghế ngay sau góc cua, cách mép tường cua $1.5\text{m}$.
- **Thao tác:**
  - Người đeo đi thẳng dọc bức tường che khuất (chưa thấy ghế).
  - Đến góc cua, xoay người rẽ dứt khoát 90 độ vào hành lang mới; chiếc ghế lập tức lọt vào giữa tầm nhìn camera. Người đeo tiếp tục bước về phía ghế.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen10_turn_corner --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen10_turn_corner --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen10_turn_corner --run 3 --duration 25 --source auto --headless
  ```

---

### NHÓM IV: ĐIỀU KIỆN MÔI TRƯỜNG & MẬT ĐỘ PHỨC TẠP (KỊCH BẢN 11, 12)

#### Kịch bản 11 (`scen11_lighting_change`): Thay đổi ánh sáng đột ngột (tối sang sáng / ngược sáng)
- **Mục tiêu kỹ thuật:** Kiểm tra khả năng thích ứng khi độ tương phản thay đổi mạnh. Khi ngược sáng hoặc thiếu sáng, YOLO thường bị tụt điểm tin cậy xuống vùng $D_{\text{low}}$ ($0.20 \le conf < 0.45$); ByteTrack sẽ chứng minh khả năng duy trì vết không bị đứt đoạn.
- **Bố trí:** Chọn đoạn hành lang đi từ khu vực trong nhà ánh sáng đèn huỳnh quang hướng thẳng ra sảnh có cửa sổ kính lớn ánh sáng mặt trời rọi vào, có một chiếc ghế đặt ở ranh giới sáng-tối.
- **Thao tác:** Người đeo đi bộ từ vùng tối tiến dần ra vùng ngược sáng cửa sổ.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen11_lighting_change --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen11_lighting_change --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen11_lighting_change --run 3 --duration 25 --source auto --headless
  ```

#### Kịch bản 12 (`scen12_dense_clutter`): Môi trường đông đúc hỗn hợp nhiều đồ vật
- **Mục tiêu kỹ thuật:** Kiểm tra hiệu năng tính toán và độ chính xác của bộ gán cặp khi mật độ bounding box dày đặc (nhiều class: `person`, `chair`, `backpack`). Đánh giá xem FPS của SORT/ByteTrack có bị tụt trên CPU Raspberry Pi 4 hay không.
- **Bố trí:** Một đoạn hành lang bố trí 2 chiếc ghế hai bên, 1 thùng carton ở giữa, 1 ba lô trên sàn, và 1 người đứng nói chuyện cách đó $3\text{m}$.
- **Thao tác:** Người đeo đi chậm rãi len lỏi qua khu vực đồ vật phức tạp trong suốt 25 giây.
- **Lệnh thực hiện:**
  ```bash
  python3 record_scenarios.py --scenario scen12_dense_clutter --run 1 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen12_dense_clutter --run 2 --duration 25 --source auto --headless
  python3 record_scenarios.py --scenario scen12_dense_clutter --run 3 --duration 25 --source auto --headless
  ```

---

## 5. Quy Trình Vận Hành Tự Động Hàng Loạt (Dành Cho Đội Hiện Trường Muốn Thu Nhanh)

Nếu đội hiện trường muốn quay liên tục theo thứ tự từ Kịch bản 1 đến 12 mà không cần gõ từng dòng lệnh dài, công cụ `record_scenarios.py` đã tích hợp sẵn chế độ `--all`:

```bash
cd ~/blind-navigation/2_edge_device/experiments/tracking_benchmark
python3 record_scenarios.py --all --runs 3 --duration 25 --source auto --countdown 5 --headless
```

**Cách thức hoạt động của lệnh trên:**
1. Màn hình Terminal sẽ in tên kịch bản và lần chạy hiện tại theo đúng thứ tự từ 1 đến 12 (ví dụ: `[scen1_static] Lần 1/3`).
2. Đội ngũ bố trí xong hiện trường, người đeo đứng vào vị trí rồi nhấn **[Enter]**.
3. Hệ thống đếm ngược $5\text{s}$ bằng chữ trên màn hình (`5... 4... 3... 2... 1...`).
4. Hệ thống ghi hình đúng $25\text{s}$ ($\approx 375\text{ frames}$), tự đóng file và thông báo thành công.
5. Hệ thống chuyển sang lần chạy kế tiếp hoặc kịch bản kế tiếp (`[scen1_static] Lần 2/3... Nhấn [Enter] để tiếp tục`).

---

## 6. Tiêu Chuẩn Nghiệm Thu Dữ Liệu Tại Chỗ Trước Khi Rời Hiện Trường

Trước khi thu dọn thiết bị ra về, Đội Hiện trường phải kiểm tra nhanh tính toàn vẹn của dữ liệu:

1. Chạy lệnh kiểm tra số lượng file và dung lượng:
   ```bash
   ls -lh ~/blind-navigation/2_edge_device/experiments/tracking_benchmark/data/videos/
   ```
2. **Tiêu chuẩn nghiệm thu đạt 100%:**
   - [x] Đầy đủ **36 tệp video `.mp4`** (12 kịch bản $\times$ 3 lần chạy lặp, định dạng `<mã>_run1.mp4`, `<mã>_run2.mp4`, `<mã>_run3.mp4`).
   - [x] Mỗi tệp `.mp4` có dung lượng từ **$2.5\text{MB}$ đến $6.0\text{MB}$** (thể hiện ghi đủ 25 giây ở 15 FPS). Nếu tệp nào chỉ có vài chục KB là bị lỗi nguồn camera, phải quay lại ngay tệp đó.
   - [x] Hoàn thành điền đầy đủ thông tin vào **Bảng Nhật Ký Hiện Trường** dưới đây.

---

## 7. Mẫu Bảng Nhật Ký Hiện Trường & Bàn Giao Dữ Liệu

Tạo một tệp văn bản `field_log.txt` đặt cùng thư mục video với nội dung sau:

```text
================================================================================
NHẬT KÝ THU THẬP DỮ LIỆU HIỆN TRƯỜNG - BENCHMARK OBJECT TRACKING (PBL6)
================================================================================
Ngày thực hiện: 15/11/2026
Địa điểm: Hành lang Tầng 3, Nhà C, Đại học Bách Khoa
Thành viên thực hiện: [Tên thành viên 1, Tên thành viên 2]
Thiết bị sử dụng: Raspberry Pi 4 Model B (4GB RAM)
Camera: Logitech C270 USB Webcam (Độ phân giải ghi: 640x480 @ 15 FPS)
Độ cao gắn ngàm camera: 1.42m so với mặt sàn
Góc chúc camera: 5 độ (đã chêm đệm mép trên)

TỔNG HỢP CÁC LẦN CHẠY (12 KỊCH BẢN THEO THỨ TỰ):
- scen1_static (Kịch bản 1: Ghế đứng yên 3.5m)        : Đủ 3 runs (run1, run2, run3) - Tốt
- scen2_multi (Kịch bản 2: Ghế trái 2m, thùng phải)   : Đủ 3 runs (run1, run2, run3) - Tốt
- scen3_empty (Kịch bản 3: Hành lang vắng hoàn toàn)  : Đủ 3 runs (run1, run2, run3) - Không vật cản
- scen4_occlusion (Kịch bản 4: Trợ lý che ghế 1.2s)   : Đủ 3 runs (run1, run2, run3) - Pha che rõ nét
- scen5_cross_paths (Kịch bản 5: 2 người đan chéo)    : Đủ 3 runs (run1, run2, run3) - Cắt mặt ở 2m
- scen6_reenter (Kịch bản 6: Rời khung hình rồi vào)  : Đủ 3 runs (run1, run2, run3) - Khuất 2s rồi vào lại
- scen7_edge_to_center (Kịch bản 7: Rìa vào giữa)     : Đủ 3 runs (run1, run2, run3) - Tốt
- scen8_shake (Kịch bản 8: Dằn gót tạo rung lắc)      : Đủ 3 runs (run1, run2, run3) - Rung lắc mạnh
- scen9_fast_approach (Kịch bản 9: Tiếp cận nhanh)    : Đủ 3 runs (run1, run2, run3) - Tốc độ ~1.5m/s
- scen10_turn_corner (Kịch bản 10: Cua góc 90 độ)     : Đủ 3 runs (run1, run2, run3) - Ghế xuất hiện đột ngột
- scen11_lighting_change (Kịch bản 11: Tối sang sáng) : Đủ 3 runs (run1, run2, run3) - Ngược sáng cửa kính
- scen12_dense_clutter (Kịch bản 12: Đồ vật hỗn hợp)  : Đủ 3 runs (run1, run2, run3) - Ghế, thùng, người
================================================================================
```

### Cách gửi dữ liệu cho Đội Xử lý Thuật toán:
- **Cách nhanh nhất qua mạng nội bộ:**
  Trên Raspberry Pi 4 mở web server chia sẻ file:
  ```bash
  cd ~/blind-navigation/2_edge_device/experiments/tracking_benchmark/data/videos
  python3 -m http.server 8000
  ```
  Máy tính của Đội Xử lý mở trình duyệt gõ `http://<IP_Raspberry_Pi>:8000` để tải toàn bộ các tệp `.mp4` và `field_log.txt` về máy.
- **Cách sao lưu:** Copy toàn bộ thư mục `data/videos` vào USB hoặc nén `.zip` tải lên Google Drive của nhóm.
