import streamlit as st
import numpy as np
import xgboost as xgb
from mplsoccer import Pitch
from joblib import load
import math

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Football xG Predictor", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        color: #00ff00 !important;
        background-color: #0e1117;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)


# --- CHARGEMENT DES MODÈLES (CACHE) ---
@st.cache_resource
def load_all_models():
    # Chemins à adapter selon ton dossier
    m1 = xgb.Booster();
    m1.load_model('models/xG_foot_openplay_inside_1on1.json')
    m2 = xgb.Booster();
    m2.load_model('models/xG_foot_openplay_inside_not_1on1.json')
    m3 = xgb.Booster();
    m3.load_model('models/xG_foot_openplay_outside.json')
    m4 = load('models/xG_head.joblib')
    return m1, m2, m3, m4


try:
    model_1v1, model_not_1v1, model_outside, model_head = load_all_models()
except Exception as e:
    st.error(f"Erreur de chargement des modèles : {e}. Vérifiez le dossier 'models/'.")


# --- FONCTIONS DE CALCUL (TES LOGIQUES) ---
def calculateDistance(x, y):
    x_distance = 120 - x
    y_distance = 0
    if y < 36:
        y_distance = 36 - y
    elif y > 44:
        y_distance = y - 44
    return np.sqrt(y_distance ** 2 + x_distance ** 2)


def calculateAngle(x, y):
    v0 = np.array([120, 44]) - np.array([x, y])
    v1 = np.array([120, 36]) - np.array([x, y])
    angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))


# --- BARRE LATÉRALE ---
st.sidebar.header("🛡️ Paramètres du Tir")

shot_category = st.sidebar.radio("Type de Tir", ["Pied", "Tête", "Penalty"])

# Paramètres dynamiques selon le type
if shot_category != "Penalty":
    st.sidebar.subheader("Positions")
    # Attaquant
    s_x = st.sidebar.slider("Attaquant : Distance (X)", 60.0, 120.0, 105.0)
    s_y = st.sidebar.slider("Attaquant : Latéral (Y)", 0.0, 80.0, 40.0)

    # Gardien (uniquement pour le pied)
    if shot_category == "Pied":
        st.sidebar.subheader("Environnement")
        gk_x = st.sidebar.slider("Gardien : Position X", 100.0, 120.0, 118.0)
        gk_y = st.sidebar.slider("Gardien : Position Y", 30.0, 50.0, 40.0)
        num_opp = st.sidebar.number_input("Nombre d'adversaires proches", 0, 10, 1)
        is_1v1 = st.sidebar.checkbox("Situation de 1v1", value=False)
    else:
        gk_x, gk_y, num_opp, is_1v1 = 118, 40, 0, False
else:
    s_x, s_y, gk_x, gk_y = 109.0, 40.0, 119.0, 40.0


# --- CALCUL xG ---
def get_prediction():
    dist = calculateDistance(s_x, s_y)
    angle = calculateAngle(s_x, s_y)

    if shot_category == "Penalty":
        return 0.74  # Valeur de ton code original

    if shot_category == "Tête":
        return model_head.predict_proba([[angle, dist]])[0][1]

    # Cas du Pied
    dist_shooter_gk = np.sqrt((s_x - gk_x) ** 2 + (s_y - gk_y) ** 2)

    if dist > 20 and s_x > 103.5 and (s_y < 20 or s_y > 60):  # Logique Outside
        minus = s_x - gk_x
        feat = np.array([[angle, s_y, dist_shooter_gk, dist, minus]])
        dm = xgb.DMatrix(feat, feature_names=['angle', 'y', 'DistanceShooterGk', 'distance', 'minus'])
        return model_outside.predict(dm)[0]
    else:
        dist_gk = calculateDistance(gk_x, gk_y)
        if is_1v1:
            feat = np.array([[angle, dist, dist_shooter_gk, dist_gk, num_opp]])
            dm = xgb.DMatrix(feat, feature_names=['angle', 'distance', 'DistanceShooterGk', 'DistanceGk',
                                                  'num_opposing_players'])
            return model_1v1.predict(dm)[0]
        else:
            feat = np.array([[angle, dist, gk_y, gk_x, dist_gk, num_opp]])
            dm = xgb.DMatrix(feat,
                             feature_names=['angle', 'distance', 'y_gk', 'x_gk', 'DistanceGk', 'num_opposing_players'])
            return model_not_1v1.predict(dm)[0]


xg_val = get_prediction()

# --- AFFICHAGE PRINCIPAL ---
st.title("⚽ Analyseur d'Expected Goals (xG)")

col1, col2 = st.columns([2, 1])

with col1:
    # Dessin du terrain avec mplsoccer
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80,
                  pitch_color='#224422', line_color='white', goal_type='box')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Dessiner les buts (green dans ton code)
    ax.scatter([120, 120], [36, 44], color="lime", s=100, zorder=5)

    # Shooter
    pitch.scatter(s_x, s_y, s=300, c='#e74c3c', edgecolors='white', marker='o', label='Buteur', ax=ax)
    # Gardien (si applicable)
    if shot_category == "Pied":
        pitch.scatter(gk_x, gk_y, s=250, c='#3498db', edgecolors='white', marker='s', label='Gardien', ax=ax)

    # Ligne de tir
    ax.plot([s_x, 120], [s_y, 40], color='white', linestyle='--', alpha=0.3)

    ax.legend(facecolor='#224422', edgecolor='white', labelcolor='white')
    st.pyplot(fig)

with col2:
    st.metric(label="Probabilité de but (xG)", value=f"{xg_val:.3f}")

    # Barre de progression pour le visuel
    st.progress(min(float(xg_val), 1.0))

    st.markdown("### 📊 Statistiques du tir")
    st.write(f"**Distance au but :** {calculateDistance(s_x, s_y):.1f}m")
    st.write(f"**Angle de tir :** {calculateAngle(s_x, s_y):.1f}°")

    if xg_val > 0.3:
        st.success("C'est une grosse occasion !")
    elif xg_val > 0.1:
        st.warning("Occasion dangereuse.")
    else:
        st.error("Tir très difficile.")