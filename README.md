<div align="center">

# SC 677 805 Deep Learning for Data Analytics

### การเรียนรู้เชิงลึกเพื่อการวิเคราะห์ข้อมูล

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Semester](https://img.shields.io/badge/Semester-2%2F2569-0A7B83)
![KKU](https://img.shields.io/badge/KKU-Statistics%20%26%20Data%20Science-8B1A1A)

คลังเก็บงาน โค้ด และเอกสารประกอบการเรียนรายวิชา SC 677 805<br>
หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาสถิติและวิทยาการข้อมูล คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น

</div>

---

## สารบัญ

- [ข้อมูลรายวิชา](#ข้อมูลรายวิชา)
- [เกี่ยวกับรายวิชา](#เกี่ยวกับรายวิชา)
- [ผลลัพธ์การเรียนรู้](#ผลลัพธ์การเรียนรู้)
- [แผนการเรียนรายสัปดาห์](#แผนการเรียนรายสัปดาห์)
- [การประเมินผล](#การประเมินผล)
- [โครงงาน](#โครงงาน)
- [การนำเสนอบทความวิจัย](#การนำเสนอบทความวิจัย)
- [งานปฏิบัติการ](#งานปฏิบัติการ)
- [การติดตั้งสภาพแวดล้อม](#การติดตั้งสภาพแวดล้อม)
- [โครงสร้าง Repository](#โครงสร้าง-repository)
- [ตำราและแหล่งอ้างอิง](#ตำราและแหล่งอ้างอิง)
- [จริยธรรมทางวิชาการ](#จริยธรรมทางวิชาการ)

---

## ข้อมูลรายวิชา

| หัวข้อ | รายละเอียด |
| --- | --- |
| **รหัสและชื่อวิชา** | SC 677 805 การเรียนรู้เชิงลึกเพื่อการวิเคราะห์ข้อมูล (Deep Learning for Data Analytics) |
| **หน่วยกิต** | 3(2-2-5) บรรยาย 2 ชม. · ปฏิบัติการ 2 ชม. · ศึกษาด้วยตนเอง 5 ชม. ต่อสัปดาห์ |
| **ประเภทวิชา** | วิชาเลือก |
| **อาจารย์ผู้สอน** | อ.ดร.พิชญา วิรัชโชติเสถียร |
| **ภาคการศึกษา** | ภาคปลาย ปีการศึกษา 2569 (บัณฑิตศึกษา ชั้นปีที่ 1) |
| **ช่วงเวลาเรียน** | 16 พ.ย. 2569 ถึง 12 มี.ค. 2570 |
| **วันและเวลา** | อังคาร 10:00–12:00 น. (บรรยาย) · ศุกร์ 13:30–15:30 น. (ปฏิบัติการ) |
| **สถานที่** | ห้อง mirrorRoom ภาควิชาสถิติ คณะวิทยาศาสตร์ |
| **วิชาบังคับก่อน** | ไม่มี |

---

## เกี่ยวกับรายวิชา

รายวิชานี้มุ่งให้นักศึกษา **พัฒนาแบบจำลองการเรียนรู้เชิงลึกได้ด้วยตนเอง** และ **ตัดสินใจได้ว่าเมื่อใด Deep Learning ให้ประโยชน์เหนือกว่าแบบจำลองทางสถิติหรือ Machine Learning แบบดั้งเดิม** โดยเน้นข้อมูล 4 ลักษณะที่พบบ่อยในงานธุรกิจและวิทยาศาสตร์สุขภาพ ได้แก่ ข้อมูลแบบตาราง อนุกรมเวลา ข้อความ และภาพ

> **คำถามประจำทุกสัปดาห์:** การเรียนรู้เชิงลึกคุ้มค่ากว่าแบบจำลองทางสถิติหรือไม่ และวัดจากอะไร

**แนวทางการเรียน**

1. **สัปดาห์ที่ 1–5 เขียนเองด้วย NumPy ล้วน** ไม่ใช้เฟรมเวิร์ก เพื่อให้เห็นว่าเกรเดียนต์และการปรับน้ำหนักเกิดขึ้นอย่างไร
2. **สัปดาห์ที่ 7 ย้ายเข้า PyTorch** สร้างแบบจำลองเดิมขึ้นใหม่แล้วเทียบผลกับรุ่น NumPy
3. **สัปดาห์ที่ 11–16 ศึกษาสถาปัตยกรรม** (CNN, RNN, Transformer, Autoencoder) ด้วยกรอบเดียวกันทุกสัปดาห์
   - โครงสร้างในข้อมูลที่เรียกร้องสถาปัตยกรรมนั้น
   - ข้อจำกัดเชิงโครงสร้างที่สถาปัตยกรรมกำหนดไว้
   - คุณลักษณะที่ได้ออกมา พร้อมเครื่องมือที่ใช้ตรวจสอบ
   - ตัวแบบฐาน (baseline) ที่พิสูจน์ด้วยตัวเลขว่าข้อจำกัดนั้นให้ประโยชน์จริง

<details>
<summary><b>Course description (English)</b></summary>
<br>

Introduction to deep learning techniques, understanding neural networks and architectures, applications of deep learning in data analytics, supervised and unsupervised learning models, convolutional neural networks, recurrent neural networks, optimization algorithms for deep learning, data preprocessing for deep learning models, evaluation metrics for deep learning models, case studies.

</details>

---

## ผลลัพธ์การเรียนรู้

| CLO | ผลลัพธ์การเรียนรู้ของรายวิชา | PLO |
| :---: | --- | :---: |
| **CLO1** | อธิบายหลักการของโครงข่ายประสาทเทียมและสถาปัตยกรรมเชิงลึก (MLP, CNN, RNN, Transformer, Autoencoder) เลือกสถาปัตยกรรมให้เหมาะกับข้อมูลและโจทย์ และอธิบายความแตกต่างระหว่าง DL กับ ML แบบดั้งเดิม | 2 |
| **CLO2** | เตรียมข้อมูลและออกแบบกระบวนการฝึก รวมถึงเลือก loss function, optimizer และเทคนิค regularization อย่างมีเหตุผลรองรับ | 2, 3 |
| **CLO3** | พัฒนาแบบจำลอง DL ทั้งแบบมีผู้สอนและไม่มีผู้สอนด้วย Python และ PyTorch บนข้อมูลขนาดใหญ่และ GPU | 3 |
| **CLO4** | ประเมินผลแบบจำลองด้วยเกณฑ์ที่เหมาะสม เปรียบเทียบกับตัวแบบฐานที่ไม่ใช่ DL และสื่อสารข้อจำกัดได้อย่างถูกต้อง | 2, 3 |
| **CLO5** | นำเสนอความรู้ทางเทคนิคแก่ผู้ฟังต่างสาขาด้วยภาษาที่เข้าใจง่าย วิจารณ์ผลงานผู้อื่นอย่างสร้างสรรค์ และตอบข้อวิจารณ์อย่างมีเหตุผล | 4 |

---

## แผนการเรียนรายสัปดาห์

> 📄 นำเสนอบทความวิจัยต้นคาบ · 📝 สัปดาห์ที่มี Quiz · ⛔ งดเรียน · **ตัวหนา** กำหนดส่งงานโครงงาน

### ช่วงที่ 1 · พื้นฐานจากศูนย์ด้วย NumPy สู่ PyTorch

| สัปดาห์ | อังคาร · บรรยาย | ศุกร์ · ปฏิบัติการ |
| :---: | --- | --- |
| 1 | **17 พ.ย.** ภาพรวมรายวิชา ความสัมพันธ์ AI, ML, DL ประวัติของ DL (perceptron, AI winter, backpropagation, AlexNet, Transformer) และเมื่อใดควรเลือก DL | **20 พ.ย.** ทบทวน supervised / unsupervised / reinforcement learning, kNN, decision tree, SVM, bias-variance เขียนอัลกอริทึมพื้นฐานด้วย NumPy |
| 2 | **24 พ.ย.** ⛔ งดเรียน (สีฐานเฟสติวัลและลอยกระทง) | **27 พ.ย.** บรรยายและปฏิบัติการรวม: linear regression จากพื้นฐาน, closed-form เทียบ gradient descent, squared loss ในฐานะ Gaussian NLL, ridge regression |
| 3 | **1 ธ.ค.** Ridge กับ weight decay, logistic regression, sigmoid, cross-entropy, softmax หลายกลุ่ม | **4 ธ.ค.** เขียน logistic regression และ softmax ด้วย NumPy ตรวจผลเทียบ scikit-learn |
| 4 | **8 ธ.ค.** จาก regression สู่ neural network: ชั้นซ่อน, activation function, universal approximation, forward pass ของ MLP | **11 ธ.ค.** เขียน forward pass ของ MLP สองชั้นด้วย NumPy ทดลองจำนวนนิวรอนและ activation |
| 5 | **15 ธ.ค.** Backpropagation, chain rule ทีละชั้น, numerical gradient check · มอบหมายโครงงาน | **18 ธ.ค.** เขียน MLP ครบวงจร (forward, backward, update) ด้วย NumPy ล้วน ฝึกบนข้อมูลจริง |
| 6 | **22 ธ.ค.** ⛔ งดเรียน (พิธีพระราชทานปริญญาบัตร) | **25 ธ.ค.** ⛔ งดเรียน |
| 7 📝 | **29 ธ.ค.** NumPy สู่ PyTorch: tensor, autograd, `nn.Module`, Colab และ GPU สร้างแบบจำลองเดิมใหม่แล้วเทียบผล | **1 ม.ค.** ⛔ งดเรียน (วันขึ้นปีใหม่) |
| 8 | **5 ม.ค.** Optimizer: SGD, momentum, RMSProp, Adam, learning-rate schedule, weight initialization | **8 ม.ค.** เปรียบเทียบ optimizer ภายใต้ seed เดียวกัน ใช้ `Dataset` และ `DataLoader` |
| 9 | **12 ม.ค.** การเตรียมข้อมูล, regularization (weight decay, dropout, normalization, early stopping), ตัวชี้วัด (ROC, PR, calibration) | **15 ม.ค.** Data pipeline ที่ทำซ้ำได้, ตรวจหา data leakage, วินิจฉัย overfitting · **ส่งข้อเสนอโครงงาน** |
| 10 | **19 ม.ค.** ⛔ งดเรียน (สัปดาห์สอบกลางภาค รายวิชานี้ไม่มีสอบ) | **22 ม.ค.** ⛔ งดเรียน |

### ช่วงที่ 2 · สถาปัตยกรรมเชิงลึก

| สัปดาห์ | อังคาร · บรรยาย | ศุกร์ · ปฏิบัติการ | ตัวแบบฐาน |
| :---: | --- | --- | --- |
| 11 | **26 ม.ค.** 📄 AlexNet · CNN: locality และ weight sharing, convolution, pooling, LeNet ถึง ResNet เขียน `conv2d` ด้วย NumPy | **29 ม.ค.** ฝึก CNN บนภาพ สำรวจ filter และ feature map ทดลองสับตำแหน่งพิกเซล | MLP |
| 12 | **2 ก.พ.** 📄 ResNet · Transfer learning, freeze เทียบ fine-tune, Grad-CAM | **5 ก.พ.** Fine-tune แบบจำลองที่ฝึกแล้วบนข้อมูลขนาดเล็ก อ่าน Grad-CAM · **ส่งผลตรวจคุณภาพข้อมูลและตัวแบบฐาน** | ฝึกใหม่จากศูนย์ |
| 13 📝 | **9 ก.พ.** 📄 LSTM · RNN, hidden state, gating ใน LSTM และ GRU เขียน RNN cell ด้วย NumPy | **12 ก.พ.** พยากรณ์อนุกรมเวลา ติดตาม hidden state ทดลองสับลำดับเวลา | ARIMA, sliding-window MLP |
| 14 | **16 ก.พ.** 📄 Attention Is All You Need · Attention, Transformer, positional encoding เขียน scaled dot-product attention ด้วย NumPy | **19 ก.พ.** Hugging Face กับข้อความภาษาไทย อ่าน attention map สำรวจ embedding ด้วย UMAP | TF-IDF + logistic regression |
| 15 | **23 ก.พ.** 📄 VAE & GAN · Autoencoder, VAE, latent space, anomaly detection | **26 ก.พ.** ตรวจจับความผิดปกติด้วย reconstruction error สำรวจ latent space · **ส่งความก้าวหน้าและเริ่มรอบวิจารณ์** | PCA, isolation forest |
| 16 | **2 มี.ค.** 📄 DINO & ConvNeXt · Responsible DL: MC dropout, conformal prediction, SHAP, fairness, reproducibility | **5 มี.ค.** ประเมินความไม่แน่นอนและตีความแบบจำลอง | – |
| 17 | **9 มี.ค.** ให้คำปรึกษาโครงงานรายบุคคลและซ้อมนำเสนอ | **12 มี.ค.** 🎤 **นำเสนอโครงงาน** และสรุปรายวิชา | – |

> หากมีวันหยุดเพิ่มเติมตรงกับวันอังคารหรือศุกร์ จะจัดชดเชยนอกเวลาหรือมอบหมายกิจกรรมออนไลน์แทน

---

## การประเมินผล

| กิจกรรม | CLO | สัปดาห์ | สัดส่วน |
| --- | :---: | :---: | :---: |
| งานที่ได้รับมอบหมายและรายงานปฏิบัติการ | 2, 3 | ตลอดภาค | **25%** |
| Quiz อย่างน้อย 2 ครั้ง (ประมาณ 20 นาที ต้นคาบ) | 1, 2, 4 | 7, 13 | **25%** |
| นำเสนอบทความวิจัย คนละ 2 เรื่อง | 1, 5 | 11–16 | **15%** |
| โครงงานวิเคราะห์ข้อมูลด้วย Deep Learning | 3, 4, 5 | 9, 12, 15 · นำเสนอ 17 | **35%** |
| **รวม** | | | **100%** |

> รายวิชานี้ **ไม่มีสอบกลางภาคและปลายภาค** ใช้ Quiz ระหว่างภาค และการนำเสนอโครงงานพร้อมรายงานฉบับสมบูรณ์เป็นการประเมินรวบยอด

<details>
<summary><b>เกณฑ์การตัดเกรด</b> (อิงเกณฑ์ อาจปรับอิงกลุ่มหากการกระจายคะแนนผิดปกติ)</summary>
<br>

| เกรด | A | B+ | B | C+ | C | D+ | D | F |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **คะแนน** | 80–100 | 75–79 | 70–74 | 65–69 | 60–64 | 55–59 | 50–54 | < 50 |

</details>

---

## โครงงาน

โครงงาน **รายบุคคล** เลือกปัญหาจริงหนึ่งปัญหาพร้อมข้อมูลที่เข้าถึงได้จริง ทำครบวงจรตั้งแต่ข้อมูลดิบจนถึงข้อสรุปที่ผู้ใช้นำไปประกอบการตัดสินใจได้

**โดเมนที่แนะนำ:** ข้อมูลสุขภาพ · อนุกรมเวลาและเซนเซอร์ · ข้อความภาษาไทย · ข้อมูลธุรกิจหรือข้อมูลเปิดภาครัฐ

| สัปดาห์ | สิ่งที่ต้องส่ง |
| :---: | --- |
| **9** | ข้อเสนอโครงงาน 2 หน้า: โจทย์ ผู้ใช้ผลลัพธ์และการตัดสินใจที่เกี่ยวข้อง ข้อมูลที่มีอยู่จริง ผลลัพธ์ที่จะทำนาย และตัวชี้วัดพร้อมเหตุผล |
| **12** | ผลการตรวจคุณภาพข้อมูล และผลของตัวแบบฐานที่ไม่ใช่ DL พร้อมตัวเลขประกอบ |
| **15** | ผลของแบบจำลอง DL และการวิเคราะห์ข้อผิดพลาด เข้าสู่รอบวิจารณ์ระหว่างนักศึกษา |
| **17** | รายงานฉบับสมบูรณ์ 8–10 หน้า รูปแบบบทความวิชาการ และนำเสนอ 25 นาที |

<details>
<summary><b>รายงานฉบับสมบูรณ์ต้องมี</b></summary>
<br>

- [ ] ผลตรวจคุณภาพข้อมูลและการรั่วไหลของข้อมูล (data leakage)
- [ ] การเปรียบเทียบกับตัวแบบฐานที่ไม่ใช่ DL
- [ ] ความไม่แน่นอนของทุกตัวเลขที่นำเสนอ
- [ ] การวิเคราะห์ว่าแบบจำลองผิดพลาดอย่างเป็นระบบกับกรณีใด
- [ ] ข้อจำกัดและประเด็นจริยธรรมของการนำไปใช้จริง
- [ ] ย่อหน้าสรุปผลสำหรับผู้อ่านที่ไม่ใช่นักสถิติ
- [ ] ภาคผนวกตอบข้อวิจารณ์จากรอบวิจารณ์

</details>

> 💡 หากแบบจำลอง DL ให้ผล **ด้อยกว่า** ตัวแบบฐาน ถือเป็นผลการศึกษาที่ถูกต้องและได้คะแนนเต็มได้ หากการเปรียบเทียบเป็นธรรมและมีคำอธิบายที่มีเหตุผล

**รอบวิจารณ์ (สัปดาห์ที่ 15):** อ่านร่างรายงานของเพื่อน 2 คน เขียนคำวิจารณ์แบบผู้ประเมินบทความ 1 หน้า

**การพบอาจารย์:** อย่างน้อย 3 ครั้งตลอดภาค ได้แก่ เสนอหัวข้อ รายงานผลตัวแบบฐาน และรายงานความก้าวหน้าก่อนนำเสนอ

---

## การนำเสนอบทความวิจัย

นำเสนอคนละ 2 เรื่อง คือ **สถาปัตยกรรมคลาสสิก 1 เรื่อง** (สัปดาห์ 11–14) และ **สถาปัตยกรรมร่วมสมัย 1 เรื่อง** (สัปดาห์ 15–16) นำเสนอ 10 นาที ตอบคำถาม 5 นาที

**ข้อกำหนด:** ผู้ฟังสมมติคือเพื่อนร่วมงานที่มีพื้นฐานสถิติแต่ไม่เคยเรียน DL · ใช้สมการได้ไม่เกิน 1 สมการ · มีอุปมาอย่างน้อย 1 อุปมาพร้อมบอกข้อจำกัดของอุปมานั้น · ต้องตอบ 3 คำถาม:

1. ก่อนหน้านี้วงการแก้ปัญหาอย่างไร และติดขัดที่ใด
2. แนวคิดใหม่คืออะไร (ในหนึ่งประโยค)
3. แนวคิดนั้นเปลี่ยนแปลงอะไรบ้าง

| # | บทความ | ประเด็นหลักที่ต้องสื่อสาร |
| :---: | --- | --- |
| 1 | Krizhevsky et al. (2012) · *ImageNet Classification with Deep CNNs* (AlexNet) | เครื่องที่เรียนรู้วิธีมองเองได้ดีกว่าวิธีที่มนุษย์ออกแบบให้ |
| 2 | He et al. (2016) · *Deep Residual Learning for Image Recognition* (ResNet) | ทำไมเครือข่ายที่ลึกขึ้นเคยให้ผลแย่ลง และทางลัดแก้ปัญหาอย่างไร |
| 3 | Hochreiter & Schmidhuber (1997) · *Long Short-Term Memory* | เครื่องที่เลือกได้เองว่าจะจำอะไรและลืมอะไร |
| 4 | Vaswani et al. (2017) · *Attention Is All You Need* | เลิกอ่านทีละคำ ให้ทุกคำพิจารณากันเองทั้งประโยค |
| 5 | Kingma & Welling (2014) · *Auto-Encoding Variational Bayes* (VAE) | บีบอัดข้อมูลเหลือแกนสำคัญไม่กี่แกน แล้วสร้างกลับขึ้นใหม่ได้ |
| 6 | Goodfellow et al. (2014) · *Generative Adversarial Nets* | สองเครือข่ายแข่งกันจนของที่สร้างแยกไม่ออกจากของจริง |
| 7 | Caron et al. (2021) · *Emerging Properties in Self-Supervised Vision Transformers* (DINO) | เรียนรู้โดยไม่มีเฉลย แล้วรู้จักขอบเขตวัตถุขึ้นมาเอง |
| 8 | Liu et al. (2022) · *A ConvNet for the 2020s* (ConvNeXt) | ความเหนือกว่าของ Transformer ส่วนใหญ่มาจากวิธีฝึก ไม่ใช่ตัวสถาปัตยกรรม |
| 9 | Ronneberger et al. (2015) · *U-Net* | แบ่งส่วนภาพระดับพิกเซล เชื่อมชั้นต้นกับชั้นลึกด้วย skip connection |
| 10 | Selvaraju et al. (2017) · *Grad-CAM* | เปิดกล่องดำ ชี้ตำแหน่งในภาพที่มีผลต่อการตัดสินใจมากที่สุด |

---

## งานปฏิบัติการ

**ช่วงแรก (สัปดาห์ 1–9)** ใช้คลังโค้ดเดียวต่อยอดกันทุกสัปดาห์ จบช่วงจะได้ไลบรารี DL ขนาดย่อมที่เขียนเองทั้งหมด

| Lab | สัปดาห์ | เนื้อหา | เกณฑ์ผ่าน |
| :---: | :---: | --- | --- |
| 1 | 1 | อัลกอริทึม ML พื้นฐานด้วย NumPy | ผลทำนายตรงกับ scikit-learn |
| 2 | 2 | Linear regression ด้วย gradient descent | ลู่เข้าค่าเดียวกับ closed-form |
| 3 | 3 | Logistic regression และ softmax | สัมประสิทธิ์สอดคล้องกับ scikit-learn |
| 4 | 4 | Forward pass ของ MLP | ประมาณฟังก์ชันไม่เชิงเส้นที่ตัวแบบเชิงเส้นทำไม่ได้ |
| 5 | 5 | Backpropagation ฉบับสมบูรณ์ | ผ่าน gradient check ทุกชั้น |
| 6 | 7 | ย้ายแบบจำลองเข้า PyTorch | loss curve ตรงกับรุ่น NumPy |
| 7 | 8 | เปรียบเทียบ optimizer | ใช้ seed เดียวกัน สรุปว่าวิธีใดดีกว่าและเพราะอะไร |
| 8 | 9 | ตรวจสอบ data pipeline | หาจุดรั่วไหลที่ซ่อนไว้ในโค้ดตัวอย่างได้ |
| 9 | 11–13 | CNN และ RNN | มีตัวแบบฐาน ภาพคุณลักษณะ และผลทดลองรื้อโครงสร้างข้อมูล |
| 10 | 14–16 | Transformer และ Autoencoder | มีตัวแบบฐาน ภาพคุณลักษณะ และการประเมินความไม่แน่นอน |

**ช่วงที่สอง (สัปดาห์ 11–16)** ส่ง 3 ส่วนทุกครั้ง:
1. ตารางเปรียบเทียบผลกับตัวแบบฐาน พร้อมช่วงความเชื่อมั่นจากหลาย seed
2. ภาพแสดงคุณลักษณะที่แบบจำลองเรียนรู้ได้
3. ย่อหน้าตีความประมาณ 150 คำ

**กติกาการส่ง**
- เริ่มในคาบศุกร์ ส่งก่อนคาบศุกร์ถัดไป
- Notebook ต้อง **รันซ้ำได้ตั้งแต่ต้นจนจบ** และ **กำหนด random seed คงที่**
- แนบไฟล์ระบุสภาพแวดล้อม (`environment.yml` หรือ `requirements.txt`)
- ตัดคะแนนปฏิบัติการที่ต่ำที่สุดออก 1 ชิ้น

---

## การติดตั้งสภาพแวดล้อม

ซอฟต์แวร์ที่ใช้: Python 3.11+, PyTorch, NumPy, pandas, scikit-learn, Matplotlib, Hugging Face Transformers, Optuna และ Google Colab หรือ GPU ของภาควิชา

สร้าง environment ด้วย Miniconda (เปิด **Anaconda Prompt** แล้วรันทีละบรรทัด):

```bash
conda create -n dl677 python=3.11 -y
conda activate dl677
pip install torch numpy pandas scikit-learn matplotlib transformers optuna jupyter
```

> ถ้าเครื่องมี GPU ของ NVIDIA ให้ใช้คำสั่งติดตั้ง PyTorch จาก [pytorch.org](https://pytorch.org/get-started/locally/) แทน `pip install torch`

บันทึกสภาพแวดล้อมสำหรับแนบไปกับงาน:

```bash
conda env export --no-builds > environment.yml
```

**แหล่งข้อมูล:** Kaggle · UCI Machine Learning Repository · PhysioNet · Hugging Face Datasets
(โครงงานที่ใช้ข้อมูลผู้ป่วยต้องผ่านการพิจารณาจริยธรรมการวิจัย หรือใช้ข้อมูลที่ไม่ระบุตัวตนเท่านั้น)

---

## โครงสร้าง Repository

```text
SC-677-805-Deep-Learning-for-Data-Analytics/
├── README.md
├── .gitignore
├── มคอ.3 รายละเอียดของรายวิชา (Course Specification).pdf
├── EDM-SQL1.pdf               # SQL พื้นฐาน (Enterprise Data Management)
├── EDM-SQL2.pdf               # SQL JOIN
├── Lab/                       # งานปฏิบัติการ
│   └── DLDA_Lab0_NumPy.ipynb
├── Slide/                     # สไลด์บรรยายและเอกสารประกอบ
│   ├── 0_logistics.pdf
│   ├── 1_linear_regression.pdf
│   ├── 2_svm.pdf
│   ├── 3_decision_tree.pdf
│   ├── week1_handout.pdf
│   └── Screenshot *.png       # คำตอบแบบฝึกเติมคำใน handout
└── Essential-Data-Science/    # repo แยก ไม่ถูก track ใน repo นี้
```

### Lab

| Lab | หัวข้อ | สถานะ |
| :---: | --- | :---: |
| 0 | [NumPy for Machine Learning](Lab/DLDA_Lab0_NumPy.ipynb): สร้าง array และรูปร่าง, broadcasting, axis, boolean mask, unique / argmax / argsort, พีชคณิตเชิงเส้น, การสุ่ม | ✅ 136 / 136 |

### Slide

| ไฟล์ | หัวข้อ |
| --- | --- |
| [0_logistics.pdf](Slide/0_logistics.pdf) | เป้าหมายรายวิชาและภาพรวม Machine Learning |
| [1_linear_regression.pdf](Slide/1_linear_regression.pdf) | Linear regression |
| [2_svm.pdf](Slide/2_svm.pdf) | Support vector machine |
| [3_decision_tree.pdf](Slide/3_decision_tree.pdf) | Decision tree |
| [week1_handout.pdf](Slide/week1_handout.pdf) | ใบงานคาบที่ 1: ภาพรวมรายวิชาและทบทวน Machine Learning |
| [Screenshot 2026-11-17](<Slide/Screenshot 2026-11-17 112029.png>) · [Screenshot 2026-11-20](<Slide/Screenshot 2026-11-20 141000.png>) | คำตอบแบบฝึกเติมคำใน handout สัปดาห์ที่ 1 |

### เอกสารอื่น

| ไฟล์ / โฟลเดอร์ | คำอธิบาย |
| --- | --- |
| [มคอ.3 รายละเอียดของรายวิชา](<มคอ.3 รายละเอียดของรายวิชา (Course Specification).pdf>) | เอกสาร Course Specification ฉบับเต็ม |
| [EDM-SQL1.pdf](EDM-SQL1.pdf) | สไลด์ SQL พื้นฐาน: ประวัติ SQL, RDBMS, คำสั่งพื้นฐาน |
| [EDM-SQL2.pdf](EDM-SQL2.pdf) | สไลด์ SQL JOIN: INNER, LEFT, RIGHT, FULL JOIN |
| `Essential-Data-Science/` | สื่อการสอนพื้นฐานจากวิชา SC 612 104 (fork จาก [PitchayaW/SC-612-104-Essential-Data-Science](https://github.com/PitchayaW/SC-612-104-Essential-Data-Science)) เก็บแยกเป็นอีก repo ที่ [tanpharamut/SC-612-104-Essential-Data-Science](https://github.com/tanpharamut/SC-612-104-Essential-Data-Science) |

---

## ตำราและแหล่งอ้างอิง

**ตำราหลัก**

1. Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. *Dive into Deep Learning*. Cambridge University Press. ฉบับออนไลน์ฟรีที่ [d2l.ai](https://d2l.ai) พร้อมโค้ด PyTorch ทุกบท
2. Bishop, C. M., & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. มุมมองเชิงความน่าจะเป็น เหมาะกับนักศึกษาสถิติ
3. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer. ฉบับออนไลน์ฟรีที่ [statlearning.com](https://www.statlearning.com) ใช้ทบทวน ML แบบดั้งเดิม

**อ่านเพิ่มเติม**

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Murphy, K. P. (2022, 2023). *Probabilistic Machine Learning: An Introduction* และ *Advanced Topics*. MIT Press.
- van der Ploeg, T., Austin, P. C., & Steyerberg, E. W. (2014). Modern modelling techniques are data hungry. *BMC Medical Research Methodology*, 14, 137.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS*.

---

## จริยธรรมทางวิชาการ

- ทุกโครงงานต้อง **ระบุแหล่งที่มาของข้อมูลและโค้ด**
- ต้อง **ระบุการใช้เครื่องมือ AI ช่วยงาน**
- รายงานผลตามจริง **รวมถึงผลที่ด้อยกว่าตัวแบบฐาน**
- การลอกผลงานหรือรายงานผลเกินจริง ถือเป็นการทุจริตทางวิชาการ
