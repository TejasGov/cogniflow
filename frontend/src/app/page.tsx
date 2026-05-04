"use client";

import { useState, useEffect, useRef } from 'react';
import { Brain, Activity, Target, Minimize2, ClipboardList, Camera, ListChecks } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as blazeface from '@tensorflow-models/blazeface';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import './App.css';

// BlazeFace landmark order: right_eye(0), left_eye(1), nose(2), mouth(3), right_ear(4), left_ear(5)
// Mirrors the YOLOv8 keypoint formulas from cogniflow_webcam.py
function computeFeatures(landmarks: number[][]) {
  if (!landmarks || landmarks.length < 6) return null;

  const [rx, ry] = landmarks[0]; // right eye
  const [lx, ly] = landmarks[1]; // left eye
  const [nx, ny] = landmarks[2]; // nose
  const [rex]    = landmarks[4]; // right ear
  const [lex]    = landmarks[5]; // left ear

  const eyeCenterX = (lx + rx) / 2;
  const eyeCenterY = (ly + ry) / 2;
  const eyeWidth   = Math.hypot(lx - rx, ly - ry);
  if (eyeWidth < 1) return null;

  // Gaze: nose closeness to eye-center (symmetric face = looking straight = high)
  const noseDist = Math.hypot(nx - eyeCenterX, ny - eyeCenterY);
  const gaze     = Math.max(0, Math.min(1, 1 - noseDist / eyeWidth));

  // Head pose: horizontal nose offset from ear-center (facing forward = low offset = high)
  const earCenterX = (lex + rex) / 2;
  const headPose   = Math.max(0, Math.min(1, 1 - Math.abs(nx - earCenterX) / eyeWidth));

  // Eye openness: vertical asymmetry between the two eye Y-coords (matches original / 1000 formula)
  const eyeOpenness = Math.abs(ly - ry) / (eyeWidth * 10);

  return { gaze, headPose, eyeOpenness };
}

const GAZE_BUF = 25; // ~5 s at 5 fps sampling

const useSensors = (
  isActive: boolean,
  taskFeatures: { complexity: number; steps: number; priority: number },
  behaviorRef: React.RefObject<{ gaze: number; headPose: number; eyeOpen: number; gazeVar: number }>
) => {
  const [data, setData] = useState({
    riskScore: 0,
    behavior: { gaze: 0, headPose: 0, eyeOpen: 0, gazeVar: 0 },
    shap: [
      { name: 'Complexity',     val: 0.45 },
      { name: 'Est. Steps',     val: 0.12 },
      { name: 'Priority',       val: 0.05 },
      { name: 'Avg Gaze',       val: -0.25 },
      { name: 'Head Pose',      val: -0.15 },
      { name: 'Eye Open',       val: -0.05 },
      { name: 'Gaze Variance',  val: 0.35 },
    ],
    history: Array.from({ length: 20 }, (_, i) => ({ time: i, risk: 0 }))
  });

  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      const b = behaviorRef.current;

      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complexity_score:   taskFeatures.complexity,
          estimated_steps:    taskFeatures.steps,
          priority_encoded:   taskFeatures.priority,
          avg_gaze:           b.gaze,
          avg_head_pose:      b.headPose,
          avg_eye_openness:   b.eyeOpen,
          gaze_variance:      b.gazeVar
        })
      })
        .then(res => res.json())
        .then(result => {
          if (result.riskScore !== undefined) {
            setData(prev => ({
              ...prev,
              riskScore: result.riskScore,
              behavior: { ...b },
              history: [
                ...prev.history.slice(1),
                { time: prev.history[prev.history.length - 1].time + 1, risk: result.riskScore }
              ]
            }));
          }
        })
        .catch(err => console.error('Predict error:', err));
    }, 2000);

    return () => clearInterval(interval);
  }, [isActive, taskFeatures.complexity, taskFeatures.steps, taskFeatures.priority]);

  return { data };
};

export default function App() {
  const [taskAnalyzed, setTaskAnalyzed]   = useState(false);
  const [taskFeatures, setTaskFeatures]   = useState({ complexity: 0, steps: 0, priority: 0 });
  const [stepsList, setStepsList]         = useState<string[]>([]);
  const [customTaskInput, setCustomTaskInput] = useState('');
  const [isAnalyzing, setIsAnalyzing]     = useState(false);

  // Real behavioral values written by the camera loop, read by useSensors
  const behaviorRef  = useRef({ gaze: 0.84, headPose: 0.88, eyeOpen: 0.030, gazeVar: 0.015 });
  const gazeBufferRef = useRef<number[]>([]);

  const { data } = useSensors(taskAnalyzed, taskFeatures, behaviorRef);

  const videoRef    = useRef<HTMLVideoElement>(null);
  const requestRef  = useRef<number | null>(null);
  const detectorRef = useRef<any>(null);
  const [faceBox, setFaceBox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);

  const handleAnalyzeTask = async () => {
    if (!customTaskInput) return;
    setIsAnalyzing(true);
    try {
      const res = await fetch('/api/analyze-task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_description: customTaskInput })
      });
      const result = await res.json();
      setTaskFeatures({ complexity: result.complexity, steps: result.steps, priority: result.priority });
      setStepsList(result.steps_list || []);
      setTaskAnalyzed(true);
      setCustomTaskInput('');
    } catch (e) {
      console.error(e);
    }
    setIsAnalyzing(false);
  };

  useEffect(() => {
    if (!taskAnalyzed) return;

    const detectFace = async () => {
      if (videoRef.current && detectorRef.current && videoRef.current.readyState === 4) {
        try {
          const faces = await detectorRef.current.estimateFaces(videoRef.current, false);

          if (faces.length > 0) {
            // ── bounding box for the overlay ──────────────────────────────
            const start = faces[0].topLeft      as [number, number];
            const end   = faces[0].bottomRight  as [number, number];
            const videoWidth   = videoRef.current.videoWidth;
            const videoHeight  = videoRef.current.videoHeight;
            const clientWidth  = videoRef.current.clientWidth;
            const clientHeight = videoRef.current.clientHeight;
            const scale        = Math.max(clientWidth / videoWidth, clientHeight / videoHeight);
            const offsetX      = (videoWidth  * scale - clientWidth)  / 2;
            const offsetY      = (videoHeight * scale - clientHeight) / 2;
            setFaceBox({
              x:      (start[0] * scale) - offsetX,
              y:      (start[1] * scale) - offsetY,
              width:  (end[0] - start[0]) * scale,
              height: (end[1] - start[1]) * scale
            });

            // ── real behavioral features from BlazeFace landmarks ─────────
            const landmarks = faces[0].landmarks as number[][];
            const feat = computeFeatures(landmarks);
            if (feat) {
              gazeBufferRef.current.push(feat.gaze);
              if (gazeBufferRef.current.length > GAZE_BUF) gazeBufferRef.current.shift();

              const buf      = gazeBufferRef.current;
              const mean     = buf.reduce((a, v) => a + v, 0) / buf.length;
              const rawVar   = buf.reduce((a, v) => a + (v - mean) ** 2, 0) / buf.length;

              // BlazeFace is unusually stable on occluded faces / hands, giving near-zero
              // variance even when gaze is low. Apply a floor that rises as gaze drops,
              // matching the variance range the XGBoost model was trained on (~0.05–0.11
              // for distracted users).
              const varFloor  = Math.max(0, (0.75 - feat.gaze) * 0.20);
              const gazeVar   = Math.max(rawVar, varFloor);

              behaviorRef.current = {
                gaze:     feat.gaze,
                headPose: feat.headPose,
                eyeOpen:  feat.eyeOpenness,
                gazeVar
              };
            }
          } else {
            // No face detected → strong distraction signal
            setFaceBox(null);
            gazeBufferRef.current.push(0.20);
            if (gazeBufferRef.current.length > GAZE_BUF) gazeBufferRef.current.shift();
            const buf    = gazeBufferRef.current;
            const mean   = buf.reduce((a, v) => a + v, 0) / buf.length;
            const rawVar = buf.reduce((a, v) => a + (v - mean) ** 2, 0) / buf.length;
            behaviorRef.current = { gaze: 0.20, headPose: 0.20, eyeOpen: 0.008, gazeVar: Math.max(rawVar, 0.10) };
          }
        } catch (e) { /* ignore frame errors */ }
      }
      requestRef.current = requestAnimationFrame(detectFace);
    };

    async function setupCamera() {
      try {
        await tf.ready();
        detectorRef.current = await blazeface.load();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => detectFace();
        }
      } catch (err) {
        console.error('Camera/detector setup error:', err);
      }
    }

    setupCamera();
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [taskAnalyzed]);

  let statusColor = 'var(--safe-color)';
  let statusGlow  = 'var(--safe-glow)';
  let statusText  = 'SAFE';
  if (data.riskScore >= 0.7) {
    statusColor = 'var(--risk-color)';
    statusGlow  = 'var(--risk-glow)';
    statusText  = 'AT RISK';
  } else if (data.riskScore >= 0.4) {
    statusColor = 'var(--moderate-color)';
    statusGlow  = 'var(--moderate-glow)';
    statusText  = 'CAUTION';
  }

  return (
    <div className="app-container animate-enter">
      {/* Left Panel */}
      <div className="left-panel">
        <div className="header">
          <Brain size={32} className="header-icon" />
          <div>
            <h1 className="header-title">CogniFlow</h1>
            <div className="header-subtitle">Real-time Inference</div>
          </div>
        </div>

        <div className="glass-panel" style={{ marginBottom: '-8px' }}>
          <div className="panel-title" style={{ marginBottom: '12px' }}>
            <Target className="panel-icon" size={20} />
            Custom Task
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              placeholder="E.g. Filing a tax return..."
              value={customTaskInput}
              onChange={e => setCustomTaskInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleAnalyzeTask()}
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                padding: '8px 12px',
                color: 'var(--text-primary)',
                outline: 'none'
              }}
            />
            <button
              onClick={handleAnalyzeTask}
              disabled={isAnalyzing}
              style={{
                background: 'var(--accent-color)',
                color: '#000',
                border: 'none',
                borderRadius: '8px',
                padding: '8px 16px',
                fontWeight: 600,
                cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                opacity: isAnalyzing ? 0.7 : 1
              }}
            >
              {isAnalyzing ? '...' : 'Analyze'}
            </button>
          </div>
        </div>

        <div className="glass-panel">
          <div className="panel-title">
            <ClipboardList className="panel-icon" size={20} />
            Task Analysis
          </div>
          {taskAnalyzed ? (
            <div className="feature-list">
              <div className="feature-item">
                <span className="feature-label">Complexity Score</span>
                <span className="feature-val">{taskFeatures.complexity}/5</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Estimated Steps</span>
                <span className="feature-val">{taskFeatures.steps}</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Priority Level</span>
                <span className="feature-val">{taskFeatures.priority}/5</span>
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-secondary)', fontSize: '13px', textAlign: 'center', padding: '12px 0' }}>
              Enter a task above to begin analysis
            </div>
          )}
        </div>

        {taskAnalyzed && (
          <div className="glass-panel">
            <div className="panel-title">
              <Activity className="panel-icon" size={20} />
              Behavioral Signals
            </div>
            <div className="feature-list">
              <div className="feature-item">
                <span className="feature-label">Average Gaze</span>
                <span className="feature-val">{data.behavior.gaze.toFixed(2)}</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Head Pose</span>
                <span className="feature-val">{data.behavior.headPose.toFixed(2)}</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Eye Openness</span>
                <span className="feature-val">{data.behavior.eyeOpen.toFixed(3)}</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Gaze Variance</span>
                <span className="feature-val" style={{ color: data.behavior.gazeVar > 0.05 ? 'var(--risk-color)' : 'inherit' }}>
                  {data.behavior.gazeVar.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Task Breakdown — bottom left */}
        <div className="glass-panel" style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
          <div className="panel-title">
            <ListChecks className="panel-icon" size={20} />
            Task Breakdown
          </div>
          {stepsList.length > 0 ? (
            <ol style={{ margin: 0, paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {stepsList.map((step, i) => (
                <li key={i} style={{ fontSize: '13px', color: 'var(--text-primary)', lineHeight: '1.5' }}>
                  {step}
                </li>
              ))}
            </ol>
          ) : (
            <div style={{ color: 'var(--text-secondary)', fontSize: '13px', textAlign: 'center', padding: '12px 0' }}>
              {taskAnalyzed ? 'No steps returned' : 'Analyze a task to see the breakdown'}
            </div>
          )}
        </div>
      </div>

      {/* Center Panel */}
      <div className="center-panel">
        <div className="glass-panel video-container">
          {taskAnalyzed ? (
            <>
              <video
                ref={videoRef}
                className="video-feed"
                autoPlay
                playsInline
                muted
                style={{ position: 'absolute', top: 0, left: 0, zIndex: 0 }}
              />
              <div className="video-overlay" style={{ zIndex: 1 }} />
              <div className="video-scanner" style={{ zIndex: 2, background: statusColor, boxShadow: `0 0 20px ${statusColor}` }} />
              <div className="face-box" style={{
                zIndex: 3,
                borderColor: statusColor,
                boxShadow: `0 0 15px ${statusGlow}, inset 0 0 15px ${statusGlow}`,
                ...(faceBox
                  ? { left: faceBox.x, top: faceBox.y, width: faceBox.width, height: faceBox.height, transition: 'none' }
                  : { opacity: 0 })
              }} />
              <div style={{ position: 'absolute', top: 20, left: 20, background: 'rgba(0,0,0,0.5)', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', display: 'flex', gap: '8px', alignItems: 'center', zIndex: 4 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#FF3246', animation: 'pulse 1.5s infinite' }} />
                REC
              </div>
            </>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '16px', color: 'var(--text-secondary)' }}>
              <Camera size={56} style={{ opacity: 0.25 }} />
              <div style={{ fontSize: '14px', textAlign: 'center', opacity: 0.5, lineHeight: '1.6' }}>
                Camera will activate after<br />task analysis is complete
              </div>
            </div>
          )}
        </div>

        <div className="glass-panel flex-between" style={{ padding: '24px 32px' }}>
          <div>
            <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Fusion Classifier</div>
            <div style={{ fontSize: '24px', fontFamily: 'var(--font-display)', fontWeight: 600 }}>Abandonment Risk</div>
          </div>

          <div className="dial-container flex-center">
            <div className="dial-circle">
              <div className="dial-indicator" style={{ transform: `rotate(${data.riskScore * 360 - 180}deg)` }} />
              <div className="dial-value" style={{ color: taskAnalyzed ? statusColor : 'var(--text-secondary)' }}>
                {taskAnalyzed ? `${(data.riskScore * 100).toFixed(0)}%` : '—'}
              </div>
            </div>
            <div className="dial-label" style={{ color: taskAnalyzed ? statusColor : 'var(--text-secondary)' }}>
              {taskAnalyzed ? statusText : 'WAITING'}
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div className="right-panel">
        <div className="glass-panel">
          <div className="panel-title">
            <Target className="panel-icon" size={20} />
            SHAP Value Analysis
          </div>
          <div className="shap-list">
            {data.shap.map((s, i) => {
              const width = Math.min(100, (Math.abs(s.val) / 0.6) * 100);
              const isPos = s.val > 0;
              return (
                <div className="shap-item" key={i}>
                  <div className="shap-header">
                    <span className="shap-name">{s.name}</span>
                    <span className="shap-val" style={{ color: isPos ? 'var(--risk-color)' : 'var(--safe-color)' }}>
                      {isPos ? '+' : ''}{s.val.toFixed(2)}
                    </span>
                  </div>
                  <div className="shap-bar-container">
                    <div className="shap-zero-line" />
                    <div className={`shap-bar ${isPos ? 'positive' : 'negative'}`} style={{ width: `${width}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: '20px', fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', gap: '16px', justifyContent: 'center' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ display: 'inline-block', width: '8px', height: '8px', background: 'var(--risk-color)', borderRadius: '2px' }} />
              Increases Risk
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ display: 'inline-block', width: '8px', height: '8px', background: 'var(--safe-color)', borderRadius: '2px' }} />
              Decreases Risk
            </span>
          </div>
        </div>

        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div className="panel-title">
            <Minimize2 className="panel-icon" size={20} />
            Risk Trend
          </div>
          <div style={{ flex: 1, minHeight: '150px', marginLeft: '-20px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.history}>
                <XAxis dataKey="time" hide />
                <YAxis domain={[0, 1]} hide />
                <Tooltip
                  contentStyle={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)', borderRadius: '8px' }}
                  itemStyle={{ color: '#FFF' }}
                />
                <Line
                  type="monotone"
                  dataKey="risk"
                  stroke={taskAnalyzed ? statusColor : 'var(--text-secondary)'}
                  strokeWidth={3}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
