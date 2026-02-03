import streamlit as st
import numpy as np
import xgboost as xgb
from mplsoccer import Pitch
from joblib import load
import math
from streamlit_image_coordinates import streamlit_image_coordinates
import io
from PIL import Image

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

# --- INITIALISATION DES POSITIONS (SESSION STATE) ---
if "s_x" not in st.session_state:
    st.session_state.s_x, st.session_state.s_y = 105.0, 40.0
if "gk_x" not in st.session_state:
    st.session_state.gk_x, st.session_state.gk_y = 118.0, 40.0


# --- CHARGEMENT DES MODÈLES ---
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


# --- FONCTIONS DE CALCUL ---
def calculateDistance(x, y):
    x_dist = 120 - x
    y_dist = 0
    if y < 36:
        y_dist = 36 - y
    elif y > 44:
        y_dist = y - 44
    return np.sqrt(y_dist ** 2 + x_dist ** 2)


def calculateAngle(x, y):
    v0 = np.array([120, 44]) - np.array([x, y])
    v1 = np.array([120, 36]) - np.array([x, y])
    angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))


# --- BARRE LATÉRALE ---
st.sidebar.header("🛡️ Paramètres du Tir")

shot_category = st.sidebar.radio("Type de Tir", ["Pied", "Tête", "Penalty"])

selection_mode = "Le Buteur"
if shot_category != "Penalty":
    if shot_category == "Pied":
        st.sidebar.subheader("Sélection au clic")
        selection_mode = st.sidebar.selectbox("🎯 Cliquer pour placer :", ["Le Buteur", "Le Gardien"])
        num_opp = st.sidebar.number_input("Adversaires proches", 0, 10, 1)
        is_1v1 = st.sidebar.checkbox("Situation de 1v1", value=False)
    else:
        num_opp, is_1v1 = 0, False

    if st.sidebar.button("Réinitialiser positions"):
        st.session_state.s_x, st.session_state.s_y = 105.0, 40.0
        st.session_state.gk_x, st.session_state.gk_y = 118.0, 40.0
        st.rerun()
else:
    st.session_state.s_x, st.session_state.s_y, st.session_state.gk_x, st.session_state.gk_y = 109.0, 40.0, 119.0, 40.0

# --- LOGIQUE PRÉDICTION xG ---
def get_prediction():
    sx, sy = st.session_state.s_x, st.session_state.s_y
    gx, gy = st.session_state.gk_x, st.session_state.gk_y
    dist = calculateDistance(sx, sy)
    angle = calculateAngle(sx, sy)

    if shot_category == "Penalty": return 0.74
    if shot_category == "Tête" and model_head:
        return model_head.predict_proba([[angle, dist]])[0][1]

    dist_s_gk = np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)

    if dist > 20 and sx > 103.5 and (sy < 20 or sy > 60):
        if not model_outside: return 0.02
        dm = xgb.DMatrix(np.array([[angle, sy, dist_s_gk, dist, sx - gx]]),
                         feature_names=['angle', 'y', 'DistanceShooterGk', 'distance', 'minus'])
        return model_outside.predict(dm)[0]
    else:
        dist_gk = calculateDistance(gx, gy)
        if is_1v1 and model_1v1:
            dm = xgb.DMatrix(np.array([[angle, dist, dist_s_gk, dist_gk, num_opp]]),
                             feature_names=['angle', 'distance', 'DistanceShooterGk', 'DistanceGk',
                                            'num_opposing_players'])
            return model_1v1.predict(dm)[0]
        elif model_not_1v1:
            dm = xgb.DMatrix(np.array([[angle, dist, gy, gx, dist_gk, num_opp]]),
                             feature_names=['angle', 'distance', 'y_gk', 'x_gk', 'DistanceGk', 'num_opposing_players'])
            return model_not_1v1.predict(dm)[0]
    return 0.05


xg_val = get_prediction()

# --- AFFICHAGE PRINCIPAL ---
st.title("⚽ Analyseur d'Expected Goals (xG)")

col1, col2 = st.columns([2, 1])

with col1:
    # Dessin du terrain avec mplsoccer
    pitch = Pitch(pitch_type='custom', pitch_length=120, pitch_width=80,
                  pitch_color='#224422', line_color='white', goal_type='box')
    fig, ax = pitch.draw(figsize=(10, 7))

    # Dessin des joueurs
    pitch.scatter(st.session_state.s_x, st.session_state.s_y, s=300,
                  c='#e74c3c', edgecolors='white', marker='o', label='Buteur', ax=ax, zorder=3)

    if shot_category == "Pied":
        pitch.scatter(st.session_state.gk_x, st.session_state.gk_y, s=250,
                      c='#3498db', edgecolors='white', marker='s', label='Gardien', ax=ax, zorder=3)

    # Ligne de visée
    ax.plot([st.session_state.s_x, 120], [st.session_state.s_y, 40], color='white', linestyle='--', alpha=0.3)
    ax.legend(facecolor='#224422', edgecolor='white', labelcolor='white', loc='upper left')

    # Conversion en Image PIL
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)

    # Affichage du terrain cliquable
    coords = streamlit_image_coordinates(img, use_column_width=True)

    if coords:
        # CALCUL CORRIGÉ : x normal, y inversé
        st.session_state.s_x_last = st.session_state.s_x

        new_x = (coords["x"] / coords["width"]) * 120
        # Inversion de l'axe Y (1 - ratio) car l'image commence en haut
        new_y = (1 - (coords["y"] / coords["height"])) * 80

        if selection_mode == "Le Buteur":
            st.session_state.s_x, st.session_state.s_y = new_x, new_y
        else:
            st.session_state.gk_x, st.session_state.gk_y = new_x, new_y

        st.rerun()

with col2:
    st.metric(label="Probabilité de but (xG)", value=f"{xg_val:.3f}")
    st.progress(min(float(xg_val), 1.0))

    st.markdown("### 📊 Données du tir")
    st.write(f"**Distance au but :** {calculateDistance(st.session_state.s_x, st.session_state.s_y):.1f}m")
    st.write(f"**Angle de tir :** {calculateAngle(st.session_state.s_x, st.session_state.s_y):.1f}°")
    st.write(f"**Position Buteur :** X={st.session_state.s_x:.1f}, Y={st.session_state.s_y:.1f}")

    if xg_val > 0.3:
        st.success("C'est une grosse occasion !")
    elif xg_val > 0.1:
        st.warning("Occasion dangereuse.")
    else:
        st.error("Tir très difficile.")