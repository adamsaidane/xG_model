import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import Button,RadioButtons, TextBox
import numpy as np
import xgboost as xgb

def set_shot_type(event, type):
    global selected_type
    selected_type = type
    print(f"Selected shot type: {type}")
def calculateDistance(x,y):
    x_distance=120-x
    y_distance=0
    if (y<36):
        y_distance = 36-y
    elif (y>44):
        y_distance = y-44
    return np.sqrt(y_distance**2+x_distance**2)

def calculateAngle(x,y):
    g0 = [120, 44]
    p = [x, y]
    g1 = [120, 36]

    v0 = np.array(g0) - np.array(p)
    v1 = np.array(g1) - np.array(p)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return(abs(np.degrees(angle)))

def calculateDistanceShooterGk(x1,y1,x2,y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def Minus(x1,x2):
    return x1-x2

def calculate_xG_inside_one_v_one(x, y,x_gk,y_gk,num_opposing_players, model):
    distance = calculateDistance(x, y)
    angle = calculateAngle(x, y)
    DistanceShooterGk = calculateDistanceShooterGk(x,y,x_gk,y_gk)
    DistanceGk=calculateDistance(x_gk,y_gk)
    features = np.array([[angle,distance,DistanceShooterGk,DistanceGk,num_opposing_players]])
    dmatrix_x = xgb.DMatrix(features, feature_names=['angle','distance','DistanceShooterGk','DistanceGk','num_opposing_players'])
    return model.predict(dmatrix_x)

def calculate_xG_inside_not_one_v_one(x, y,x_gk,y_gk,num_opposing_players, model):
    distance = calculateDistance(x, y)
    angle = calculateAngle(x, y)
    DistanceGk=calculateDistance(x_gk,y_gk)
    features = np.array([[angle,distance,y_gk,x_gk,DistanceGk,num_opposing_players]])
    dmatrix_x = xgb.DMatrix(features, feature_names=['angle','distance','y_gk','x_gk','DistanceGk','num_opposing_players'])
    return model.predict(dmatrix_x)

def calculate_xG_outside(x, y,x_gk,y_gk, model):
    distance = calculateDistance(x, y)
    angle = calculateAngle(x, y)
    DistanceShooterGk = calculateDistanceShooterGk(x,y,x_gk,y_gk)
    minus=Minus(x,x_gk)
    features = np.array([[angle,y,DistanceShooterGk,distance,minus]])
    dmatrix_x = xgb.DMatrix(features, feature_names=['angle','y','DistanceShooterGk','distance','minus'])
    return model.predict(dmatrix_x)

def calculate_xG_head(x,y,model):
    distance = calculateDistance(x, y)
    angle = calculateAngle(x, y)
    x = [[angle, distance]]
    return model.predict_proba(x)[:, 1]

model_xG_inside_one_v_one= xgb.Booster()
model_xG_inside_one_v_one.load_model('models/xG_foot_openplay_inside_1on1.json')

model_xG_inside_not_one_v_one= xgb.Booster()
model_xG_inside_not_one_v_one.load_model('models/xG_foot_openplay_inside_not_1on1.json')

model_xG_outside= xgb.Booster()
model_xG_outside.load_model('models/xG_foot_openplay_outside.json')

from joblib import load
model_xG_head= load('models/xG_head.joblib')

def draw_pitch(ax):
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color="black", lw=2)

    ax.plot([120, 120, 114.5, 114.5, 120], [31, 49, 49, 31, 31], color="black", lw=2)

    ax.plot([120, 120, 103.5, 103.5, 120], [20, 60, 60, 20, 20], color="black", lw=2)

    ax.scatter(109, 40, color="black", s=20)

    center_circle = plt.Circle((60, 40), 10, color="black", fill=False, lw=2)
    ax.add_patch(center_circle)

    ax.plot([60, 60], [0, 80], color="black", lw=2)

    arc = Arc((109, 40), height=18.3, width=18.3, angle=180, theta1=308, theta2=52, color="black", lw=2)
    ax.add_patch(arc)

    ax.scatter(120, 36, color="green", s=50, zorder=5)  
    ax.scatter(120, 44, color="green", s=50, zorder=5) 

selected_type=None
selected_num_players=0
selected_one_v_one=False
step = "shooter"  
shooter_position = None
gk_position = None

def onclick(event):
    global step, shooter_position, gk_position, selected_num_players, selected_one_v_one

    x, y = event.xdata, event.ydata
    while x is None or y is None:
        x, y = event.xdata, event.ydata

    if step == "shooter" and selected_type == "Foot":
        gk_position = None
        shooter_position = (x, y)
        ax.scatter(x, y, color="red", s=50, zorder=5)
        fig.canvas.draw()
        print(f"Shooter position: x={x:.2f}, y={y:.2f}")
        step = "goalkeeper"
        print("Click on the pitch to select the goalkeeper's position.")

    elif step == "goalkeeper" and selected_type == "Foot":
        gk_position = (x, y)
        ax.scatter(x, y, color="blue", s=50, zorder=5)
        fig.canvas.draw()
        print(f"Goalkeeper position: x={x:.2f}, y={y:.2f}")
        step = "num_players" 
        print("Enter the number of opposing players in the text box.")

    elif step == "num_players" and selected_type == "Foot":
        print("Awaiting number of players from TextBox...")

    elif step == "1v1" and selected_type == "Foot":
        print("Awaiting 1v1 selection from RadioButtons...")
        step="shooter"

    elif selected_type == "Head":
        ax.scatter(x, y, color="red", s=50, zorder=5)
        fig.canvas.draw()
        xG = calculate_xG_head(x, y, model_xG_head)
        ax.text(x, y + 1, f"{xG[0]:.2f}", color="green", fontsize=10, zorder=6)
        fig.canvas.draw()
        step="shooter"

    elif selected_type == "Penalty":
        x, y = 109, 40
        ax.scatter(x, y, color="red", s=50, zorder=5)
        xG = [0.74045802]
        ax.text(x, y + 1, f"{xG[0]:.2f}", color="green", fontsize=10, zorder=6)
        fig.canvas.draw()
        step="shooter"

def update_num_players(val):
    global selected_num_players, step
    try:
        selected_num_players = int(val)
        print(f"Number of opposing players: {selected_num_players}")
        step = "1v1"  
        print("Select 1v1 status using the radio buttons.")
    except ValueError:
        print("Please enter a valid number.")

def update_one_v_one(label):
    global selected_one_v_one, step, shooter_position, gk_position

    selected_one_v_one = label == "True"
    print(f"1v1 selected: {selected_one_v_one}")

    x, y = shooter_position
    x_gk, y_gk = gk_position
    dist = calculateDistance(x, y)

    if dist > 20 and x > 103.5 and (y < 20 or y > 60):
        xG = calculate_xG_outside(x, y, x_gk, y_gk, model_xG_outside)
    else:
        if selected_one_v_one:
            xG = calculate_xG_inside_one_v_one(x, y, x_gk, y_gk, selected_num_players, model_xG_inside_one_v_one)
        else:
            xG = calculate_xG_inside_not_one_v_one(x, y, x_gk, y_gk, selected_num_players, model_xG_inside_not_one_v_one)

    print(f"xG for foot shot: {xG[0]:.2f}")
    ax.text(x, y, f"{xG[0]:.2f}", color="green", fontsize=10, zorder=6)
    fig.canvas.draw()


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 120)
ax.set_ylim(0, 80)
draw_pitch(ax)

ax_head = plt.axes([0.1, 0.92, 0.1, 0.05])
ax_foot = plt.axes([0.3, 0.92, 0.1, 0.05])
ax_penalty = plt.axes([0.5, 0.92, 0.1, 0.05])
ax_text = plt.axes([0.7, 0.92, 0.1, 0.05])
ax_radio = plt.axes([0.85, 0.92, 0.1, 0.05])

btn_head = Button(ax_head, "Head")
btn_foot = Button(ax_foot, "Foot")
btn_penalty = Button(ax_penalty, "Penalty")
text_box = TextBox(ax_text, "Players")
radio = RadioButtons(ax_radio, ["True", "False"])

btn_head.on_clicked(lambda event: set_shot_type(event, "Head"))
btn_foot.on_clicked(lambda event: set_shot_type(event, "Foot"))
btn_penalty.on_clicked(lambda event: set_shot_type(event, "Penalty"))
text_box.on_submit(update_num_players)
radio.on_clicked(update_one_v_one)

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()