# Fisika Sensor, 3 Tahap Pengolahan Sinyal, dan Arsitektur Hybrid ML + FSM RebahGuard

## 1. Dua Sensor Fisik pada Setiap Edge

- **Accelerometer** mengukur resultan percepatan linier dan gravitasi bumi pada tiga sumbu $(a_x, a_y, a_z)$ dalam satuan $\text{m/s}^2$.
  - Saat HP dipegang duduk tegak menatap pengguna: $a_z > 0$ ($\text{overhead\_deg} < 90^\circ$).
  - Saat HP dipegang di atas wajah sambil rebahan telentang (atau diletakkan telungkup di meja): $a_z < 0$ ($\text{overhead\_deg} > 110^\circ$, hingga $180^\circ$ saat telungkup rata).
- **Gyroscope** mengukur kecepatan sudut rotasi $(g_x, g_y, g_z)$ dalam satuan $\text{rad/s}$.
  - Saat HP ditaruh di meja, giroskop mendekati $0.0000\text{ rad/s}$.
  - Saat HP dipegang tangan manusia (meskipun berusaha diam), terdapat *physiological hand tremor* dan aktivitas mikro jari ($\text{gyro\_rms} > 0.015\text{ rad/s}$).

---

## 2. Tiga Tahap Data Processing (`shared/processing.py`)

### Processing A: Gravity Orientation Smoothing (`overhead_deg`, `roll_deg`, & `lean_deg`)
Untuk meredam getaran tangan sesaat sebelum menghitung sudut orientasi terhadap gravitasi, dihitung rata-rata vektor percepatan pada satu window:
$$\bar{a}_x = \frac{1}{N}\sum_{i=1}^N a_{x,i}, \quad \bar{a}_y = \frac{1}{N}\sum_{i=1}^N a_{y,i}, \quad \bar{a}_z = \frac{1}{N}\sum_{i=1}^N a_{z,i}$$
$$g_{\text{norm}} = \max\left(10^{-6}, \sqrt{\bar{a}_x^2 + \bar{a}_y^2 + \bar{a}_z^2}\right)$$
$$\text{overhead\_deg} = \text{degrees}\left(\arccos\left(\text{clamp}\left(\frac{\bar{a}_z}{g_{\text{norm}}}, -1.0, 1.0\right)\right)\right)$$
$$\text{roll\_deg} = \text{degrees}\left(\text{atan2}\left(|\bar{a}_x|, \sqrt{\bar{a}_y^2 + \bar{a}_z^2}\right)\right)$$

Selain itu, fungsi tunggal [`safe_lean_deg(overhead_deg, roll_deg)`](../shared/processing.py) merekonstruksi sudut kemiringan bidang layar (*in-plane lean angle* $0^\circ\text{--}90^\circ$) untuk membedakan orientasi **PORTRAIT** vs **LANDSCAPE**, dan mengembalikan `0.0` secara aman saat HP tergeletak datar.

### Processing B: Mean-Centered Dynamic Motion RMS (Bebas Bias Kalibrasi $9.81\text{ m/s}^2$)
Mengapa **tidak boleh** menghitung $\sqrt{\frac{1}{N}\sum (m_i - 9.81)^2}$?
Karena sensor MEMS akselerometer pada smartphone komersial memiliki *zero-g offset* / bias kalibrasi statis sekitar $0.1\text{--}0.3\text{ m/s}^2$ (misalnya saat diam di meja membaca $a_z = -9.55\text{ m/s}^2$ konstan). Jika dikurangi $9.81$, HP yang diam di meja akan memiliki $\text{RMS} = 0.26\text{ m/s}^2 > 0.05\text{ m/s}^2$ dan gagal dikenali sebagai ditaruh di meja!

Oleh karena itu, RebahGuard menghitung simpangan terhadap **rata-rata magnitudo window itu sendiri ($\bar{m}$)**:
$$m_i = \sqrt{a_{x,i}^2 + a_{y,i}^2 + a_{z,i}^2}, \quad \bar{m} = \frac{1}{N}\sum_{i=1}^N m_i$$
$$\text{motion\_rms} = \sqrt{\frac{1}{N}\sum_{i=1}^N (m_i - \bar{m})^2}$$
Dengan rumus ini, bias kalibrasi statis saling menghilangkan sehingga HP yang diam di meja menghasilkan `motion_rms = 0.0000` secara presisi.

### Processing C: Gyroscope Micro-Activity RMS (`gyro_rms`)
$$\text{gyro\_rms} = \sqrt{\frac{1}{N}\sum_{i=1}^N (g_{x,i}^2 + g_{y,i}^2 + g_{z,i}^2)}$$

---

## 3. Deteksi Orientasi Portrait vs Landscape & Histeresis (`shared/config.py`)

Saat `use_roll = True` (default aktif), sistem mendeteksi orientasi layar menggunakan `lean_deg` dengan pelindung zona datar (`flat_gate_deg = 20.0°`) dan konfirmasi durasi (`orientation_confirm_sec = 0.5s`):
- **Mode `PORTRAIT`**: `overhead_enter_deg = 110.0°`, `overhead_exit_deg = 95.0°`, `lying_limit_sec = 10.0s`.
- **Mode `LANDSCAPE`** (main game / menonton video): `landscape_overhead_enter_deg = 80.0°`, `landscape_overhead_exit_deg = 70.0°`, `landscape_lying_limit_sec = 15.0s`.
- Ketika pengguna duduk tegak bermain game *landscape* ($\text{overhead\_deg} \approx 78.5^\circ < 80.0^\circ$), sistem mengenali orientasi `LANDSCAPE` namun **tidak** memicu status rebahan palsu.

---

## 4. Arsitektur Hybrid Machine Learning + FSM (`nodes/posture_service/`)

### A. Pembingkaian Ilmiah: Data-Driven Threshold & Posture Learning
Alih-alih menggantikan hukum fisika dengan *black-box* murni, RebahGuard memosisikan model **Random Forest Classifier** sebagai *data-driven threshold & posture learner* yang mempelajari batas keputusan non-linier dari 5 fitur (`overhead_deg`, `roll_deg`, `lean_deg`, `motion_rms`, `gyro_rms`) sesuai kebiasaan memegang HP pengguna, dan dievaluasi secara objektif berdampingan dengan **Rule-Based Baseline** di [`scripts/train_model.py`](../scripts/train_model.py).

### B. Protokol Perekaman Multi-Sesi Bebas Data Leakage (Minimal 3 Sesi per Kelas = 9 Sesi)
Data deret waktu (*time-series*) sensor yang berdekatan dalam satu sesi rekaman memiliki kemiripan tinggi (*temporal autocorrelation*). Jika dibagi secara acak per baris (*random row split*), metrik akurasi akan mengalami *data leakage*. Oleh sebab itu:
1. Perekaman melalui [`scripts/collect_dataset.py`](../scripts/collect_dataset.py) memberi tag `session_id` unik (`sess_<YYYYMMDD_HHMMSS>_<4hex>`) dan menyaring paket dengan proteksi **anti-stale** serta **anti-replay**.
2. Pengguna dianjurkan merekam **minimal 3 sesi per kelas (total 9 sesi)** yang mencakup variasi:
   - `POSISI_DUDUK_SEHAT`: duduk *portrait*, duduk main game *landscape*, dan HP tergeletak menghadap atas di meja.
   - `DITARUH_TELUNGKUP`: HP telungkup diam di permukaan meja pada beberapa posisi.
   - `REBAHAN_AKTIF`: rebahan telentang *portrait*, rebahan *landscape*, serta posisi miring/menunduk ekstrem.
3. Pelatihan di [`scripts/train_model.py`](../scripts/train_model.py) melakukan **Group Split per `session_id`** dengan jaminan mutlak `Session Overlap (Train ∩ Test): 0 sesi`.

### C. Urutan Keputusan Hybrid di `RebahDetector._accept_edge1`
1. **Hard-Veto Fisik Meja (`is_face_down_table`)**: Jika `overhead_deg >= face_down_deg` ($175^\circ$) serta `motion_rms` dan `gyro_rms` berada di bawah ambang diam meja, sistem **selalu mengunci state ke `DITARUH_TELUNGKUP`**, mengosongkan buffer ML, dan me-reset timer rebahan—sekalipun model ML memprediksi kelas lain dengan confidence tinggi.
2. **Inferensi ML + Temporal Majority Smoothing (`ml_smoothing_window = 3`)**: Jika `use_ml = True` dan probabilitas prediksi Random Forest $\ge \text{ml\_confidence\_min}$ ($0.55$), prediksi dimasukkan ke *sliding window buffer* sepanjang $3$ window. Kelas postur efektif ditentukan melalui suara terbanyak (*majority vote*, dengan aturan mempertahankan postur efektif sebelumnya saat seri) agar 1 window *glitch* sensor tidak me-reset akumulasi waktu `lying_elapsed_sec`.
3. **Fallback Rule-Based Otomatis**: Jika model belum dilatih, `use_ml` dinonaktifkan dari Dashboard, atau confidence di bawah `ml_confidence_min`, sistem otomatis beralih mulus ke histeresis sudut matematis bawaan (`detection_mode = "rule"`).

---

## 5. Keterbatasan Ilmiah Sensor IMU Smartphone

Perlu dicatat secara jujur bahwa sensor IMU pada smartphone hanya mengukur orientasi perangkat terhadap vektor gravitasi bumi dan getaran mekanis pada bodi HP, **bukan mengukur anatomi leher atau punggung pengguna secara langsung**:
- Label `REBAHAN_AKTIF` merepresentasikan keadaan fisik ketika layar HP menghadap ke bawah di atas pengguna disertai mikro-aktivitas genggaman tangan.
- Apabila pengguna tidur telentang tetapi memegang HP tegak lurus di samping pinggang ($\text{overhead\_deg} \approx 90^\circ$), vektor gravitasi pada HP identik dengan posisi duduk tegak. Kombinasi **Hybrid ML + FSM**, kalibrasi per pengguna, dan fitur intervensi **Shake-to-Scold** dari HP Orang Tua dirancang untuk mengatasi keterbatasan sensor tunggal tersebut secara praktis.
