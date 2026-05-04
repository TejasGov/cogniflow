import { useState, useEffect } from 'react';
import { Brain, Activity, Target, Minimize2, ClipboardList, Camera } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import './App.css';

// Mock Data Generator for visual demonstration
const useSensors = () => {
  const [data, setData] = useState({
    riskScore: 0.35,
    task: { complexity: 5, steps: 12, priority: 2 },
    behavior: { gaze: 0.8, headPose: 0.9, eyeOpen: 0.03, gazeVar: 0.1 },
    shap: [
      { name: 'Complexity', val: 0.45 },
      { name: 'Est. Steps', val: 0.12 },
      { name: 'Priority', val: 0.05 },
      { name: 'Avg Gaze', val: -0.25 },
      { name: 'Head Pose', val: -0.15 },
      { name: 'Eye Open', val: -0.05 },
      { name: 'Gaze Variance', val: 0.35 },
    ],
    history: Array.from({ length: 20 }, (_, i) => ({ time: i, risk: 0.3 }))
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        // Random walk for risk score bounded [0.1, 0.95]
        let newRisk = prev.riskScore + (Math.random() - 0.5) * 0.15;
        newRisk = Math.max(0.1, Math.min(0.95, newRisk));
        
        const newHistory = [...prev.history.slice(1), { time: prev.history[prev.history.length-1].time + 1, risk: newRisk }];
        
        return {
          ...prev,
          riskScore: newRisk,
          behavior: {
            gaze: Math.max(0.2, Math.min(1.0, prev.behavior.gaze + (Math.random() - 0.5) * 0.1)),
            headPose: Math.max(0.3, Math.min(1.0, prev.behavior.headPose + (Math.random() - 0.5) * 0.1)),
            eyeOpen: Math.max(0.01, Math.min(0.05, prev.behavior.eyeOpen + (Math.random() - 0.5) * 0.005)),
            gazeVar: Math.max(0.0, Math.min(1.0, prev.behavior.gazeVar + (Math.random() - 0.5) * 0.15)),
          },
          shap: prev.shap.map(s => ({
            ...s,
            // Vary behavioral SHAP values based on risk trend
            val: s.name.includes('Gaze') || s.name.includes('Head') 
              ? s.val + (Math.random() - 0.5) * 0.1
              : s.val
          })),
          history: newHistory
        };
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return data;
};

function App() {
  const data = useSensors();
  
  // Determine color based on risk
  let statusColor = 'var(--safe-color)';
  let statusGlow = 'var(--safe-glow)';
  let statusText = 'SAFE';
  if (data.riskScore >= 0.7) {
    statusColor = 'var(--risk-color)';
    statusGlow = 'var(--risk-glow)';
    statusText = 'AT RISK';
  } else if (data.riskScore >= 0.4) {
    statusColor = 'var(--moderate-color)';
    statusGlow = 'var(--moderate-glow)';
    statusText = 'CAUTION';
  }

  return (
    <div className="app-container animate-enter">
      {/* Left Panel: Task & Behaviors */}
      <div className="left-panel">
        <div className="header">
          <Brain size={32} className="header-icon" />
          <div>
            <h1 className="header-title">CogniFlow</h1>
            <div className="header-subtitle">Real-time Inference</div>
          </div>
        </div>

        <div className="glass-panel">
          <div className="panel-title">
            <ClipboardList className="panel-icon" size={20} />
            Task Analysis
          </div>
          <div className="feature-list">
            <div className="feature-item">
              <span className="feature-label">Complexity Score</span>
              <span className="feature-val">{data.task.complexity}/10</span>
            </div>
            <div className="feature-item">
              <span className="feature-label">Estimated Steps</span>
              <span className="feature-val">{data.task.steps}</span>
            </div>
            <div className="feature-item">
              <span className="feature-label">Priority Level</span>
              <span className="feature-val">{data.task.priority}</span>
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ flex: 1 }}>
          <div className="panel-title">
            <Activity className="panel-icon" size={20} />
            YOLOv8 Behavioral Signals
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
              <span className="feature-val" style={{ color: data.behavior.gazeVar > 0.4 ? 'var(--risk-color)' : 'inherit' }}>
                {data.behavior.gazeVar.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Center Panel: Camera & Risk Score */}
      <div className="center-panel">
        <div className="glass-panel video-container">
          <div className="video-overlay"></div>
          <div className="video-scanner" style={{ background: statusColor, boxShadow: `0 0 20px ${statusColor}` }}></div>
          <div className="face-box" style={{ borderColor: statusColor, boxShadow: `0 0 15px ${statusGlow}, inset 0 0 15px ${statusGlow}` }}></div>
          
          {/* We use a placeholder camera icon for the design. 
              In real implementation, connect to a WebRTC feed or react-webcam */}
          <div className="flex-center" style={{ flexDirection: 'column', gap: '16px', color: 'rgba(255,255,255,0.2)' }}>
             <Camera size={64} />
             <span style={{ fontFamily: 'var(--font-display)', letterSpacing: '2px' }}>LIVE SENSOR FEED</span>
          </div>

          <div style={{ position: 'absolute', top: 20, left: 20, background: 'rgba(0,0,0,0.5)', padding: '6px 12px', borderRadius: '20px', fontSize: '12px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#FF3246', animation: 'pulse 1.5s infinite' }}></div>
            REC
          </div>
        </div>

        <div className="glass-panel flex-between" style={{ padding: '24px 32px' }}>
          <div>
            <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Fusion Classifier</div>
            <div style={{ fontSize: '24px', fontFamily: 'var(--font-display)', fontWeight: 600 }}>Abandonment Risk</div>
          </div>
          
          <div className="dial-container flex-center">
            <div className="dial-circle">
               <div className="dial-indicator" style={{ transform: `rotate(${((data.riskScore) * 360) - 180}deg)` }}></div>
               <div className="dial-value" style={{ color: statusColor }}>
                 {(data.riskScore * 100).toFixed(0)}%
               </div>
            </div>
            <div className="dial-label" style={{ color: statusColor }}>{statusText}</div>
          </div>
        </div>
      </div>

      {/* Right Panel: SHAP & History */}
      <div className="right-panel">
        <div className="glass-panel">
          <div className="panel-title">
            <Target className="panel-icon" size={20} />
            SHAP Value Analysis
          </div>
          <div className="shap-list">
            {data.shap.map((s, i) => {
              const maxVal = 0.6; // Scale for bars
              const width = Math.min(100, (Math.abs(s.val) / maxVal) * 100);
              const isPos = s.val > 0;
              
              return (
                <div className="shap-item" key={i}>
                  <div className="shap-header">
                    <span className="shap-name">{s.name}</span>
                    <span className="shap-val" style={{ color: isPos ? 'var(--risk-color)' : 'var(--safe-color)' }}>
                      {s.val > 0 ? '+' : ''}{s.val.toFixed(2)}
                    </span>
                  </div>
                  <div className="shap-bar-container">
                    <div className="shap-zero-line"></div>
                    <div 
                      className={`shap-bar ${isPos ? 'positive' : 'negative'}`} 
                      style={{ width: `${width}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: '20px', fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', gap: '16px', justifyContent: 'center' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ display: 'inline-block', width: '8px', height: '8px', background: 'var(--risk-color)', borderRadius: '2px' }}></span>
              Increases Risk
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ display: 'inline-block', width: '8px', height: '8px', background: 'var(--safe-color)', borderRadius: '2px' }}></span>
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
                  stroke={statusColor} 
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

export default App;
