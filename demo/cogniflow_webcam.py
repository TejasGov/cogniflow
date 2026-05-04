import cv2
import numpy as np
import pandas as pd
import pickle
import shap
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

MODEL_PKL   = 'cogniflow_xgboost.pkl'
YOLO_MODEL  = 'yolov8n-pose.pt'  
WINDOW_SEC  = 5            
FEATURE_NAMES = [
    'complexity_score', 'estimated_steps', 'priority_encoded',
    'avg_gaze', 'avg_head_pose', 'avg_eye_openness', 'gaze_variance'
]

TASK_FEATURES = {
    'complexity_score': 5,
    'estimated_steps': 12,
    'priority_encoded': 2
}

print('Loading XGBoost model...')
with open(MODEL_PKL, 'rb') as f:
    xgb_model = pickle.load(f)

print('Loading YOLOv8-Pose...')
yolo = YOLO(YOLO_MODEL)

explainer = shap.TreeExplainer(xgb_model)
print('Models ready. Starting webcam...')

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def gaze_score(kp):
    nose, le, re = kp[0], kp[1], kp[2]
    ec = (le + re) / 2
    ew = dist(le, re)
    if ew < 1e-3: return 0.5
    return float(max(0, min(1, 1 - abs(nose[0] - ec[0]) / ew)))

def head_pose_score(kp):
    nose, la, ra = kp[0], kp[3], kp[4]
    ec = (la + ra) / 2
    ew = dist(la, ra)
    if ew < 1e-3: return 0.5
    return float(max(0, min(1, 1 - abs(nose[0] - ec[0]) / ew)))

def eye_openness(kp):
    nose, le, re = kp[0], kp[1], kp[2]
    ey = (le[1] + re[1]) / 2
    return float(max(0.005, min(0.05, abs(nose[1] - ey) / 1000.0)))

def draw_shap_overlay(frame, shap_vals, feature_vals, risk_score):
    h, w = frame.shape[:2]
    panel_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, 'CogniFlow', (w - panel_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'SHAP Feature Analysis', (w - panel_w + 10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    risk_color = (0, 80, 220) if risk_score < 0.5 else (0, 60, 220)
    if risk_score >= 0.7:
        risk_color = (0, 30, 200)
    label = 'AT RISK' if risk_score >= 0.5 else 'SAFE'
    label_color = (50, 50, 255) if risk_score >= 0.5 else (50, 200, 50)
    cv2.putText(frame, f'Risk Score: {risk_score:.3f}  [{label}]',
                (w - panel_w + 10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    bar_x = w - panel_w + 10
    bar_y = 92
    bar_w = panel_w - 20
    bar_h = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * min(risk_score, 1.0))
    bar_color = (50, 200, 50) if risk_score < 0.5 else (50, 50, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.line(frame, (w - panel_w + 10, 115), (w - 10, 115), (80, 80, 80), 1)
    cv2.putText(frame, 'SHAP Values (impact on risk)', (w - panel_w + 10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    feat_labels = [
        'complexity', 'est_steps', 'priority',
        'avg_gaze', 'avg_head_pose', 'eye_open', 'gaze_var'
    ]
    max_shap = max(abs(shap_vals).max(), 0.01)
    bar_area_w = 120
    zero_x = w - panel_w + 10 + bar_area_w // 2 + 30

    for i, (fname, fval, sval) in enumerate(zip(feat_labels, feature_vals, shap_vals)):
        y = 148 + i * 38
        is_task = i < 3
        name_color = (100, 180, 255) if is_task else (100, 255, 180)
        cv2.putText(frame, f'{fname}', (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, name_color, 1)
        cv2.putText(frame, f'{fval:.3f}', (w - panel_w + 10, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        bar_len = int((abs(sval) / max_shap) * (bar_area_w // 2))
        bar_color = (50, 50, 255) if sval > 0 else (50, 200, 50)
        if sval > 0:
            cv2.rectangle(frame, (zero_x, y - 8), (zero_x + bar_len, y + 4), bar_color, -1)
        else:
            cv2.rectangle(frame, (zero_x - bar_len, y - 8), (zero_x, y + 4), bar_color, -1)
        cv2.line(frame, (zero_x, y - 10), (zero_x, y + 6), (120, 120, 120), 1)
        cv2.putText(frame, f'{sval:+.3f}', (zero_x + bar_area_w // 2 + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (50, 50, 255) if sval > 0 else (50, 200, 50), 1)

    ly = h - 60
    cv2.rectangle(frame, (w - panel_w + 10, ly), (w - panel_w + 20, ly + 10), (50, 50, 255), -1)
    cv2.putText(frame, '= increases risk', (w - panel_w + 25, ly + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
    cv2.rectangle(frame, (w - panel_w + 10, ly + 16), (w - panel_w + 20, ly + 26), (50, 200, 50), -1)
    cv2.putText(frame, '= decreases risk', (w - panel_w + 25, ly + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
    cv2.putText(frame, 'Blue=task  Green=behavioral', (w - panel_w + 10, ly + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1)
    cv2.putText(frame, "Press 'S' to screenshot  'Q' to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: Could not open webcam.')
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gaze_buf, head_buf, eye_buf = [], [], []
risk_score = 0.0
shap_vals  = np.zeros(7)
feat_vals  = np.zeros(7)
last_update = time.time()
screenshot_count = 0

print("Webcam running. Press 'S' to save screenshot, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo(frame, verbose=False)
    if results[0].keypoints is not None:
        kp_data = results[0].keypoints.xy.cpu().numpy()
        if len(kp_data) > 0 and kp_data[0].shape[0] >= 7:
            kp = kp_data[0]
            gaze_buf.append(gaze_score(kp))
            head_buf.append(head_pose_score(kp))
            eye_buf.append(eye_openness(kp))
            if risk_score < 0.4:
                mesh_color  = (0, 255, 120)   
                point_color = (0, 255, 180)
            elif risk_score < 0.65:
                mesh_color  = (0, 165, 255)   
                point_color = (0, 200, 255)
            else:
                mesh_color  = (60, 60, 255)    
                point_color = (80, 80, 255)
            CONNECTIONS = [
                (0, 1), (0, 2),          # nose to eyes
                (1, 3), (2, 4),          # eyes to ears
                (1, 2),                  # eye to eye
                (3, 4),                  # ear to ear
                (5, 6),                  # shoulder to shoulder
                (3, 5), (4, 6),          # ears to shoulders
            ]

            pts = {}
            for idx in range(min(len(kp), 7)):
                x, y = int(kp[idx][0]), int(kp[idx][1])
                if x > 0 and y > 0:
                    pts[idx] = (x, y)

            for (a, b) in CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b],
                             tuple(c // 3 for c in mesh_color), 4, cv2.LINE_AA)
                    cv2.line(frame, pts[a], pts[b], mesh_color, 2, cv2.LINE_AA)
            POINT_LABELS = ['NOSE', 'L.EYE', 'R.EYE', 'L.EAR', 'R.EAR', 'L.SHD', 'R.SHD']
            for idx, pt in pts.items():
                cv2.circle(frame, pt, 10,
                           tuple(c // 4 for c in point_color), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6,
                           tuple(c // 2 for c in point_color), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 3, point_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 1, (255, 255, 255), -1, cv2.LINE_AA)
            face_pts = [pts[i] for i in [0,1,2,3,4] if i in pts]
            if len(face_pts) >= 3:
                xs = [p[0] for p in face_pts]
                ys = [p[1] for p in face_pts]
                pad = 60
                x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad*2)
                x2, y2 = min(frame.shape[1], max(xs)+pad), min(frame.shape[0], max(ys)+pad*3)
                blen = 25
                thick = 2
                cv2.line(frame, (x1, y1), (x1+blen, y1), mesh_color, thick)
                cv2.line(frame, (x1, y1), (x1, y1+blen), mesh_color, thick)
                cv2.line(frame, (x2, y1), (x2-blen, y1), mesh_color, thick)
                cv2.line(frame, (x2, y1), (x2, y1+blen), mesh_color, thick)
                cv2.line(frame, (x1, y2), (x1+blen, y2), mesh_color, thick)
                cv2.line(frame, (x1, y2), (x1, y2-blen), mesh_color, thick)
                cv2.line(frame, (x2, y2), (x2-blen, y2), mesh_color, thick)
                cv2.line(frame, (x2, y2), (x2, y2-blen), mesh_color, thick)
                cv2.putText(frame, 'FACE DETECTED', (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, mesh_color, 1, cv2.LINE_AA)

    if time.time() - last_update >= WINDOW_SEC and len(gaze_buf) >= 5:
        avg_gaze        = float(np.mean(gaze_buf))
        avg_head_pose   = float(np.mean(head_buf))
        avg_eye_open    = float(np.mean(eye_buf))
        gaze_var        = float(np.var(gaze_buf))

        feat_vals = np.array([
            TASK_FEATURES['complexity_score'],
            TASK_FEATURES['estimated_steps'],
            TASK_FEATURES['priority_encoded'],
            avg_gaze, avg_head_pose, avg_eye_open, gaze_var
        ])

        X_row = pd.DataFrame([feat_vals], columns=FEATURE_NAMES)
        risk_score = float(xgb_model.predict_proba(X_row)[0][1])
        sv = explainer(X_row)
        shap_vals = sv.values[0]

        gaze_buf.clear()
        head_buf.clear()
        eye_buf.clear()
        last_update = time.time()
        
    frame = draw_shap_overlay(frame, shap_vals, feat_vals, risk_score)

    cv2.imshow('CogniFlow — Live SHAP Analysis', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        screenshot_count += 1
        fname = f'cogniflow_screenshot_{screenshot_count}.png'
        cv2.imwrite(fname, frame)
        print(f'Screenshot saved: {fname}')

cap.release()
cv2.destroyAllWindows()
print('Done.')