# ⚽ xG Model — Expected Goals Football Analytics

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-189AB4?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-7.0+-F37626?style=flat-square&logo=jupyter&logoColor=white)
![mplsoccer](https://img.shields.io/badge/mplsoccer-1.x-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

An end-to-end **Expected Goals (xG)** pipeline for football analytics — from raw StatsBomb event data through feature engineering, model training, and an interactive **Streamlit web app** where you can click anywhere on a pitch and get a live xG prediction.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What is xG?](#-what-is-expected-goals-xg)
- [Live App](#-live-app)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Model Training](#-model-training)
- [Results & Insights](#-results--insights)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project builds a modular xG system that trains **separate models for each shot category** — because a headed cross from six yards and a driven shot from outside the box are fundamentally different events that deserve different models.

The pipeline covers:

1. **Data extraction** — pulling all shot events from every StatsBomb open-data competition
2. **Feature engineering** — distance, angle, goalkeeper position, defenders in the shooting lane
3. **Model training** — dedicated XGBoost / scikit-learn models per shot type
4. **Interactive app** — a Streamlit dashboard where you click a pitch to get instant xG predictions

---

## ⚡ What is Expected Goals (xG)?

xG is a per-shot probability of scoring, calibrated from thousands of historical shots taken in comparable situations. Key factors:

| Factor | Effect |
|---|---|
| Distance from goal | Closer = higher xG |
| Angle to goal | Central = higher xG |
| Goalkeeper position | Exposed keeper = higher xG |
| Defenders in lane | More defenders = lower xG |
| 1v1 situation | Higher xG (~0.45–0.65) |
| Under pressure | Lower xG (~15–25% reduction) |
| Shot type | Foot / head / penalty treated separately |

**Reference values (from this model):**

| Situation | xG |
|---|---|
| Penalty | **0.74** *(empirical, from data)* |
| 1v1 inside box (foot) | ~0.45–0.65 |
| Central shot, 12 yards (foot) | ~0.25–0.35 |
| Header from 6-yard box | ~0.20–0.35 |
| Shot from outside box | ~0.03–0.10 |

---

## 🖥️ Live App

The Streamlit app (`xG.py`) lets you:

- **Click anywhere on the pitch** to place the shooter
- **Click to place the goalkeeper** (foot shots)
- Choose shot type: **Foot / Head / Penalty**
- Set number of nearby opponents and 1v1 flag
- Get an instant **xG value + progress bar + shot quality label**

The app routes each shot to the correct trained model automatically:

```
Penalty            → fixed value (0.74)
Open play / Head   → xG_head.joblib
Open play / Foot   →
  ├── Inside box + 1v1       → xG_foot_openplay_inside_1on1.json
  ├── Inside box + not 1v1   → xG_foot_openplay_inside_not1on1.json
  └── Outside box            → xG_foot_openplay_outside.json
```

---

## 📁 Project Structure

```
xG_model/
│
├── models/                                      # Trained model files
│   ├── xG_foot_openPlay_basic.ipynb             # Baseline foot shot model
│   ├── xG_foot_openPlay_inside_1on1.ipynb       # Model: inside box, 1v1
│   ├── xG_foot_openPlay_inside_not1on1.ipynb    # Model: inside box, not 1v1
│   ├── xG_foot_openPlay_outsidebox.ipynb        # Model: outside box
│   ├── xG_head_openPlay_basic.ipynb             # Header model
│   ├── xG_penalty.ipynb                         # Penalty conversion rate
│   ├── xG_foot_openPlay_inside_1on1.json        # XGBoost — foot, inside box, 1v1
│   ├── xG_foot_openplay_inside_not_1on1.json    # XGBoost — foot, inside box, not 1v1
│   ├── xG_foot_openplay_outside.json            # XGBoost — foot, outside box
│   └── xG_head.joblib                           # scikit-learn — open play headers
│
├── shots_data/                                  # All shot data (raw + processed)
│   ├── shots.ipynb                              # Data extraction from StatsBomb API
│   ├── shots_openplay_foot.ipynb                # Feature engineering — foot shots
│   ├── shots_openplay_head.ipynb                # Feature engineering — header shots
│   ├── shots_penalty.ipynb                      # Penalty data filtering
│   ├── matches_shots.pkl.gzip                   # Serialised raw StatsBomb shot events
│   ├── shots.csv                                # Processed shots (all types, all features)
│   ├── shots_data.csv                           # Full extracted shot dataset
│   ├── shots_openplay_foot.csv                  # Open play foot shots
│   ├── shots_openplay_head.csv                  # Open play headers
│   └── shots_penalty.csv                        # Penalty shots
│
│
├── xG.py                                        # Streamlit interactive app
├── LICENSE
└── README.md
```

---

## 🤖 Models

| Model file | Shot type | Key features |
|---|---|---|
| `xG_foot_openplay_inside_1on1.json` | Foot, inside box, 1v1 | `angle`, `distance`, `DistanceShooterGk`, `DistanceGk`, `num_opposing_players` |
| `xG_foot_openplay_inside_not_1on1.json` | Foot, inside box, not 1v1 | `angle`, `distance`, `y_gk`, `x_gk`, `DistanceGk`, `num_opposing_players` |
| `xG_foot_openplay_outside.json` | Foot, outside box | `angle`, `y`, `DistanceShooterGk`, `distance`, `minus` |
| `xG_head.joblib` | Open play header | `angle`, `distance` |
| Penalty | Fixed | Empirical conversion rate: **0.74** |

All XGBoost models are saved as `.json` (portable, version-independent). The header model uses scikit-learn and is saved with `joblib`.

---

## 🚀 Installation

### Prerequisites

- Python 3.12+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/adamsaidane/xG_model.git
cd xG_model
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib mplsoccer
pip install streamlit streamlit-image-coordinates joblib
pip install statsbombpy scipy
```

Or with a requirements file:

```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Run the interactive Streamlit app

```bash
streamlit run xG.py
```

Then open **http://localhost:8501** in your browser.

**How to use the app:**

1. Select the **shot type** (Foot / Head / Penalty) in the sidebar
2. For foot shots, choose whether to place the **Shooter** or **Goalkeeper** on click
3. Set the number of **nearby opponents** and toggle the **1v1** flag if applicable
4. **Click on the pitch** to position the shooter (or goalkeeper)
5. The **xG value** updates instantly on the right panel

### Run the data pipeline notebooks

Open Jupyter and run notebooks in this order:

```bash
jupyter notebook
```

1. `shots.ipynb` — extract shot events from StatsBomb open data → saves `shots_data.csv`
2. `shots_openplay_foot.ipynb` — engineer features for foot shots → saves `shots_openplay_foot.csv`
3. `shots_openplay_head.ipynb` — engineer features for headers → saves `shots_openplay_head.csv`
4. `shots_penalty.ipynb` — filter penalty shots → saves `shots_penalty.csv`
5. `xG_penalty.ipynb` — compute empirical penalty conversion rate
6. Model training notebooks (`xG_foot_openPlay_*.ipynb`, `xG_head_openPlay_basic.ipynb`)

---

## 📊 Data Pipeline

### Step 1 — Extract from StatsBomb API (`shots.ipynb`)

Iterates over every available competition and season, pulls all shot events, and extracts goalkeeper position + defender counts from freeze-frame data:

```python
from statsbombpy import sb

df_competitions = sb.competitions()

# Pull shot events for every match
for competition_id, season_id in zip(...):
    df_matches[match_id] = sb.events(match_id=match_id, split=True)["shots"]
```

Freeze-frame processing counts how many opposing players are inside the **shot triangle** (shot location → left post → right post):

```python
def is_point_inside_triangle(point, v1, v2, v3):
    # Barycentric coordinate method
    denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    c = 1 - a - b
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
```

Serialised match data is stored as `matches_shots.pkl.gzip` to avoid re-fetching.

### Step 2 — Split by shot type

```python
# Foot shots (open play)
df_foot = df_shot[df_shot['shot_type'] == 'Open Play']
df_foot = df_foot[df_foot['shot_body_part'] != 'Head']

# Headers (open play)
df_head = df_shot[df_shot['shot_type'] == 'Open Play']
df_head = df_head[df_head['shot_body_part'] == 'Head']

# Penalties
df_penalty = df_shot[df_shot['shot_type'] == 'Penalty']
```

---

## 🧮 Feature Engineering

All spatial features use a **120 × 80 pitch coordinate system** (StatsBomb standard).

```python
def calculateDistance(x, y):
    """Minimum distance from shot location to the goal opening."""
    x_distance = 120 - x
    y_distance = max(0, 36 - y) if y < 36 else (max(0, y - 44) if y > 44 else 0)
    return np.sqrt(y_distance**2 + x_distance**2)

def calculateAngle(x, y):
    """Angular width of the goal from the shot location (degrees)."""
    v0 = np.array([120, 44]) - np.array([x, y])
    v1 = np.array([120, 36]) - np.array([x, y])
    angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))

def calculateDistanceShooterGk(x1, y1, x2, y2):
    """Euclidean distance between shooter and goalkeeper."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
```

### Full feature set per model

| Feature | Description |
|---|---|
| `distance` | Shooter distance to goal |
| `angle` | Angular width of goal opening |
| `DistanceShooterGk` | Shooter-to-goalkeeper distance |
| `DistanceGk` | Goalkeeper distance to goal |
| `x_gk`, `y_gk` | Goalkeeper coordinates |
| `num_opposing_players` | Defenders in the shooting lane |
| `minus` | `x_shooter − x_goalkeeper` (goalkeeper depth offset) |
| `1on1` | Binary: 1v1 situation flag |
| `underPressure` | Binary: shot taken under pressure |

---

## 📈 Results & Insights

### Penalty conversion rate

Computed directly from StatsBomb data across all competitions:

```
Penalty conversion rate: 0.7405  (~74%)
```

### Key findings

- **Distance** is the dominant predictor — shots inside 10 yards score at 5–10× the rate of long-range efforts
- **Angle** is the second most important feature — central shots are 2–3× more dangerous than wide ones
- **Goalkeeper depth** (`x_gk`) significantly improves model performance over distance/angle alone
- **Defenders in the shooting lane** reduce xG by ~3–5% per player
- Splitting into separate models per shot category meaningfully outperforms a single global model

### Routing logic (from `xG.py`)

```python
if shot_category == "Penalty":
    return 0.74

if shot_category == "Head":
    return model_head.predict_proba([[angle, dist]])[0][1]

# Foot shot routing
if dist > 20 and outside_box_zone:
    # Outside box model
    return model_outside.predict(dm)[0]
elif is_1v1:
    # Inside box, 1v1 model
    return model_1v1.predict(dm)[0]
else:
    # Inside box, not 1v1 model
    return model_not_1v1.predict(dm)[0]
```

---

## 💻 Technologies Used

| Component | Technology |
|---|---|
| **Data extraction** | statsbombpy, Pandas |
| **Feature engineering** | NumPy, math |
| **Machine learning** | XGBoost, scikit-learn |
| **Visualisation** | mplsoccer, Matplotlib |
| **Web app** | Streamlit, streamlit-image-coordinates |
| **Serialisation** | joblib, pickle + gzip |
| **Notebooks** | Jupyter, IPython |
| **Data source** | StatsBomb Open Data |

---

## 🤝 Contributing

Contributions are welcome! Planned improvements:

- [ ] Add set-piece xG models (corners, free kicks)
- [ ] Improve outside-box model with more contextual features
- [ ] Add a player/team xG summary tab to the Streamlit app
- [ ] Export shot map as downloadable PNG
- [ ] Wyscout data integration as an alternative data source
- [ ] Add model calibration curves to the app

To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request against `master`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

**Created:** January 30, 2025  
**Last updated:** March 18, 2026  
**Author:** [Adam Saidane](https://github.com/adamsaidane)

> ⚽ *Turning raw event data into football intelligence — one shot at a time.*
