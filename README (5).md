# 🏜️ DesertVision AI

> **Off-Road Autonomy Through Synthetic Data Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb)](https://mongodb.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The world's first web platform to Train, Evaluate & Deploy Semantic Segmentation Models for Desert Unmanned Ground Vehicles (UGVs).**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Solution](#-solution)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Semantic Segmentation](#-semantic-segmentation)
- [Duality AI Falcon](#-duality-ai-falcon--synthetic-data)
- [Metrics & Evaluation](#-metrics--evaluation)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

DesertVision AI converts reactive post-failure analysis into **proactive prevention** using Synthetic Data + AI Semantic Segmentation. Targeting a 90% pixel accuracy and a full risk scoring system (0→100), the platform empowers engineers to catch navigation failure zones **before** a UGV ever moves.

| Metric | Target |
|---|---|
| Pixel Accuracy | **90%** |
| Evaluation Method | **IoU (Intersection over Union)** |
| Risk Scoring | **0 → 100** |

---

## 🚨 The Problem

> UGV deployment fails **9× more often** in desert vs paved roads — yet **73% of global conflict zones** are arid terrain.

| Challenge | Description |
|---|---|
| 🏜️ **Unstructured Terrain** | No lane markings, road edges, or signals — every pixel must be AI-classified from scratch |
| 📂 **No Real-World Dataset** | Collecting desert footage is slow, dangerous & expensive. Thousands of labeled frames = months of work |
| 🔀 **Domain Shift Failure** | Models trained on one desert fail on another due to lighting changes, sand texture variance & context shifts |
| ⚠️ **Reactive Analysis Only** | Engineers review failures post-crash. No proactive platform exists to predict & prevent failure zones |

---

## 💡 Solution

An **end-to-end AI web platform** for synthetic data ingestion, model training, evaluation & deployment.

```
Ingest Synth Data  →  Label & Annotate  →  Train Model  →  Evaluate IoU  →  Deploy & Monitor
  (Falcon API)         (Auto + Manual)    (SegNet/UNet)    (Per-class)       (Real-time)
```

- **Predictive Intelligence** — Score model confidence per terrain class before deployment
- **Evidence-Led Training** — Structured synthetic data from Duality AI Falcon, zero labeling cost
- **Live IoU Dashboard** — Real-time metrics per class: Rock, Sand, Vegetation, Sky & more
- **Actionable Mitigations** — Automated augmentation strategies, domain randomization, re-training triggers

---

## 🧩 Key Features

### 01 · Dataset Manager
Upload Falcon-generated `.zip` datasets. Auto-extracts frames + ground truth masks. Browse by environment, time-of-day, and weather condition.

### 02 · Live Training Console
Configure model architecture (SegNet / UNet / DeepLab), batch size, epochs, and learning rate. Watch real-time loss & accuracy curves. Stop/resume training anytime.

### 03 · IoU Evaluation Dashboard
Per-class IoU table + mean IoU. Prediction vs. ground truth side-by-side overlay. Failure pixels highlighted in red. Export PDF report.

### 04 · Domain Shift Analyzer
Compare performance across Desert A (training) vs Desert B (test). Visualize where generalization breaks down. Suggests re-training strategies.

### 05 · Risk Score Engine
0–100 risk score per terrain frame. Identifies highest-risk navigation zones before vehicle deployment with ranked failure drivers.

### 06 · Mitigation Playbook
AI-generated fix suggestions including augmentation strategies, class weighting, and synthetic data diversity boosts — tailored to your model & IoU gap.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Framework** | PyTorch (GPU-ready) |
| **Computer Vision** | OpenCV |
| **Models** | SegNet · UNet · DeepLabV3+ |
| **Backend API** | FastAPI |
| **Frontend** | React.js |
| **Database** | MongoDB |
| **Synthetic Data** | Duality AI Falcon |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DesertVision AI                          │
├──────────────┬──────────────────┬──────────────────────────────┤
│  1. Data     │  2. Preprocessing│  3. NLP + Annotation         │
│  Ingestion   │  Pipeline        │  Engine                      │
│              │                  │                              │
│  Falcon API  │  Resize,         │  Auto-tag terrain classes,   │
│  → Raw synth │  Normalize,      │  map JSON masks to           │
│  frames +    │  Augment         │  pixel arrays                │
│  JSON labels │  (flip/rotate/   │                              │
│  → S3/local  │  brightness)     │                              │
├──────────────┼──────────────────┼──────────────────────────────┤
│  4. Model    │  5. IoU          │  6. Dashboard & Reports      │
│  Training    │  Evaluation      │                              │
│              │  Engine          │  React Frontend              │
│  SegNet /    │                  │  Node.js API                 │
│  DeepLabV3+  │  Per-class &     │  Live predictions            │
│  UNet        │  mean IoU        │  PDF export                  │
│  PyTorch,    │  Confusion       │                              │
│  GPU-ready   │  matrix +        │                              │
│              │  precision/recall│                              │
└──────────────┴──────────────────┴──────────────────────────────┘
```

> Monitoring closes the loop: every new Falcon environment auto-updates pattern models and recalibrates IoU thresholds — **continuous learning in production.**

---

## 🎯 Semantic Segmentation

Semantic segmentation assigns a class label to **every single pixel** in an image — unlike object detection (no bounding boxes). The UGV uses the output mask to plan real-time navigation paths.

### Desert Terrain Classes

| Color | Class | Navigation Meaning |
|---|---|---|
| 🟨 Sandy Yellow | **Sand / Flat Ground** | Safe zone — traversable |
| 🟫 Earthy Brown | **Rock / Boulder** | Obstacle — avoid |
| 🟩 Olive Green | **Dry Grass / Scrub** | Caution — unstable |
| 🟦 Sky Blue | **Sky** | Navigation reference |
| ⬛ Dark Grey | **Shadow Region** | Uncertain — slow down |

### IoU Formula

```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|

Score 1.0 = Perfect match
Score 0.0 = Total failure
```

---

## 🤖 Duality AI Falcon — Synthetic Data

Training on **photorealistic digital twin environments** instead of dangerous, expensive real-world data collection.

| Real Data ❌ | Synthetic Data / Falcon ✅ |
|---|---|
| Takes months to collect | Generated in hours |
| Dangerous fieldwork in deserts | Safe — runs on a server |
| Manual labeling = costly & error-prone | Auto-labeled with perfect accuracy |
| Limited edge cases captured | Unlimited synthetic edge cases |
| Fixed lighting & environment | Infinite domain randomization |
| Cannot simulate failures | Simulate any failure scenario |

> **The Twist:** Train on Desert A → Test on Desert B. Your model must generalize across unseen environments. Falcon's domain randomization is your secret weapon.

---

## 📊 Metrics & Evaluation

### Metrics Tracked

- ✅ Mean IoU (mIoU) across all terrain classes
- ✅ Per-class IoU — Rock, Sand, Grass, Sky, Shadow
- ✅ Precision & Recall by desert environment
- ✅ Domain shift delta: Train IoU vs. Test IoU
- ✅ Model inference time per frame (ms)
- ✅ Mitigation adoption rate & survival lift

### Scoring Breakdown

| Component | Weight |
|---|---|
| IoU Score | **80 pts** |
| Report Clarity | **20 pts** |

### Ecosystem Impact

- Reduce UGV deployment risk
- Increase model survival rate
- Enable smarter engineering decisions
- Shift the market from reactive post-crash analysis to **proactive desert intelligence**

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
Node.js 18+
MongoDB (local or Atlas)
CUDA-compatible GPU (recommended)
```

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/desertvision-ai.git
cd desertvision-ai
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your MongoDB URI, Falcon API key, S3 credentials
```

### 4. Run the Backend

```bash
uvicorn main:app --reload --port 8000
```

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 6. Access the Platform

```
Frontend:  http://localhost:5173
API Docs:  http://localhost:8000/docs
```

---

## 📁 Project Structure

```
desertvision-ai/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── requirements.txt
│   ├── api/
│   │   ├── datasets.py          # Dataset upload & management
│   │   ├── training.py          # Model training endpoints
│   │   ├── evaluation.py        # IoU evaluation
│   │   └── risk.py              # Risk score engine
│   ├── models/
│   │   ├── segnet.py
│   │   ├── unet.py
│   │   └── deeplabv3.py
│   ├── utils/
│   │   ├── iou.py               # IoU computation
│   │   ├── augmentation.py      # Data augmentation helpers
│   │   └── falcon_ingest.py     # Duality Falcon API integration
│   └── db/
│       └── mongo.py             # MongoDB connection
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Training.jsx
│   │   │   ├── Evaluation.jsx
│   │   │   ├── DomainShift.jsx
│   │   │   └── RiskScore.jsx
│   │   ├── components/
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_iou_evaluation.ipynb
│
├── data/
│   ├── sample_desert_a/         # Sample Falcon dataset
│   └── sample_desert_b/
│
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## 🌍 Real-World Applications

| Domain | Use Case |
|---|---|
| 🪖 **Military & Defense** | UGVs for reconnaissance, bomb disposal & logistics in sandy war zones |
| 🚀 **Mars & Planetary Rovers** | Distinguishing rock from regolith in zero-communication latency environments |
| 🆘 **Search & Rescue** | Navigating post-earthquake debris fields and desert stranding scenarios |
| ⛏️ **Mining & Oil Fields** | Autonomous logistics in remote arid-zone industrial sites |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**We don't analyse failure after it happens. We predict it before the vehicle moves.**

*DesertVision AI — Turning desert navigation risk into measurable, mitigable intelligence.*

</div>
