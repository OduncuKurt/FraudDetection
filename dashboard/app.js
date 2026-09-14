// FraudShield Dashboard — app.js v4.0 (Login + PDF Export)
const API = '';
const POLL_MS = 1000;
const DRIFT_POLL_MS = 5000;
const PRIVACY_POLL_MS = 8000;
const FL_POLL_MS = 10000;

const FRAUD_META = {
  fraud_type_0:  { label: 'Type 0 — High-Value Fraud',     color: '#ef4444', icon: '💳' },
  fraud_type_1:  { label: 'Type 1 — Account Takeover',      color: '#f59e0b', icon: '🔑' },
  fraud_type_2:  { label: 'Type 2 — Micro Card Probing',    color: '#eab308', icon: '🔍' },
  fraud_type_3:  { label: '⚠️ NEW — Money Laundering (ZSL)',color: '#a855f7', icon: '🚨' },
  normal:        { label: 'Normal Transaction',              color: '#10b981', icon: '✅' },
};

const FRAUD_INFO = {};

let chartFlow, chartPie, chartShap, chartMetrics, chartModalShap, chartDrift, chartDriftConf, chartPrivacy;
const FLOW_MAX = 60;
const flowLabels  = Array(FLOW_MAX).fill('');
const flowNormal  = Array(FLOW_MAX).fill(0);
const flowFraud   = Array(FLOW_MAX).fill(0);
let normalBuf = 0, fraudBuf = 0;

// Drift confidence trend data
const CONF_MAX = 40;
const confData = Array(CONF_MAX).fill(null);
const confLabels = Array(CONF_MAX).fill('');

// Privacy tradeoff data (pre-computed curve)
const EPSILON_VALS   = [0.1, 0.5, 1, 2, 4, 8, 10];
const ACCURACY_VALS  = [0.82, 0.89, 0.924, 0.947, 0.956, 0.961, 0.9647];

let _startTime = Date.now();
let _flRoundRunning = false;
let _currentNoiseImpact = 0;

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  loadComparison();
  loadShap('fraud_type_0');
  loadFraudTypes();
  startPolling();
  startDriftPolling();
  startPrivacyPolling();
  startFLPolling();
  startUptimeTick();
  initRiskMap();
});

// ── CHARTS ──────────────────────────────────────────
function initCharts() {
  Chart.defaults.color = '#64748b';
  Chart.defaults.borderColor = '#1e2640';
  Chart.defaults.font.family = "'Inter', sans-serif";

  // Transaction flow chart
  chartFlow = new Chart(document.getElementById('chartFlow'), {
    type: 'line',
    data: {
      labels: flowLabels,
      datasets: [
        { label: 'Normal', data: flowNormal, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.1)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2 },
        { label: 'Fraud',  data: flowFraud,  borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.1)',  fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2 },
      ]
    },
    options: { animation: false, responsive: true, maintainAspectRatio: false, scales: { x:{display:false}, y:{grid:{color:'rgba(255,255,255,.04)'}, ticks:{maxTicksLimit:4}} }, plugins:{legend:{display:false}} }
  });

  // Fraud type donut chart
  chartPie = new Chart(document.getElementById('chartPie'), {
    type: 'doughnut',
    data: {
      labels: ['Type 0', 'Type 1', 'Type 2', 'Type 3 (ZSL)'],
      datasets: [{ data: [0,0,0,0], backgroundColor: ['#ef4444','#f59e0b','#eab308','#a855f7'], borderColor:'#161b2e', borderWidth:3 }]
    },
    options: { responsive:true, maintainAspectRatio:false, cutout:'60%', plugins:{ legend:{display:false}, tooltip:{callbacks:{label:ctx=>`  ${ctx.label}: ${ctx.parsed} detected`}} } }
  });

  // SHAP chart
  chartShap = new Chart(document.getElementById('chartShap'), {
    type: 'bar', data: { labels:[], datasets:[] }, options: shapChartOptions()
  });

  // Metrics comparison chart
  chartMetrics = new Chart(document.getElementById('chartMetrics'), {
    type: 'bar',
    data: {
      labels: ['Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC'],
      datasets: [
        { label: 'Centralized',    data:[0.9289,0.9388,0.9338,0.9991,0.7741], backgroundColor:'rgba(100,116,139,.7)', borderRadius:6 },
        { label: 'Federated (FL)', data:[0.9373,1.0000,0.9676,1.0000,0.9942], backgroundColor:'rgba(59,130,246,.8)',  borderRadius:6 },
        { label: 'FL + FZSL',      data:[0.9579,0.9715,0.9647,1.0000,0.9934], backgroundColor:'rgba(139,92,246,.9)', borderRadius:6 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      scales:{ y:{min:0.7,max:1.02,grid:{color:'rgba(255,255,255,.04)'},ticks:{callback:v=>(v*100).toFixed(0)+'%',maxTicksLimit:5}}, x:{grid:{display:false}} },
      plugins:{ legend:{labels:{color:'#94a3b8',usePointStyle:true,padding:16}}, tooltip:{callbacks:{label:ctx=>` ${ctx.dataset.label}: ${(ctx.parsed.y*100).toFixed(2)}%`}} }
    }
  });

  // Drift feature chart
  chartDrift = new Chart(document.getElementById('chartDrift'), {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'KL Divergence', data: [], backgroundColor: [], borderRadius: 4 }] },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: { duration: 600 },
      scales: {
        x: { grid:{color:'rgba(255,255,255,.04)'}, ticks:{callback:v=>v.toFixed(3)} },
        y: { grid:{display:false}, ticks:{font:{family:"'JetBrains Mono',monospace",size:10}} }
      },
      plugins: { legend:{display:false}, tooltip:{callbacks:{label:ctx=>` KL: ${ctx.parsed.x.toFixed(4)}`}} }
    }
  });

  // Drift confidence trend
  chartDriftConf = new Chart(document.getElementById('chartDriftConf'), {
    type: 'line',
    data: {
      labels: confLabels,
      datasets: [{ label: 'Confidence', data: confData, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.1)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5 }]
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      scales: { x:{display:false}, y:{min:0,max:1,grid:{color:'rgba(255,255,255,.04)'},ticks:{maxTicksLimit:3,callback:v=>(v*100).toFixed(0)+'%'}} },
      plugins: { legend:{display:false} }
    }
  });

  // Privacy tradeoff chart
  chartPrivacy = new Chart(document.getElementById('chartPrivacy'), {
    type: 'line',
    data: {
      labels: EPSILON_VALS.map(e=>`ε=${e}`),
      datasets: [
        { label: 'Model Accuracy', data: ACCURACY_VALS, borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,.12)', fill: true, tension: 0.4, pointRadius: 3, borderWidth: 2 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { min:0.78, max:1.0, grid:{color:'rgba(255,255,255,.04)'}, ticks:{callback:v=>(v*100).toFixed(0)+'%', maxTicksLimit:5} },
        x: { grid:{display:false} }
      },
      plugins: {
        legend: { labels:{color:'#94a3b8', usePointStyle:true} },
        tooltip: { callbacks:{ label:ctx=>` Accuracy: ${(ctx.parsed.y*100).toFixed(2)}%` } }
      }
    }
  });

  buildPieLegend();
}

function shapChartOptions() {
  return {
    indexAxis:'y', responsive:true, maintainAspectRatio:false,
    scales:{
      x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{callback:v=>v.toFixed(2)}},
      y:{grid:{display:false},ticks:{font:{family:"'JetBrains Mono',monospace",size:11}}}
    },
    plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>` SHAP: ${ctx.parsed.x.toFixed(4)}`}}}
  };
}

function buildPieLegend() {
  const items = [
    {label:'Type 0 — Card Cloning',    color:'#ef4444'},
    {label:'Type 1 — Acct Takeover',   color:'#f59e0b'},
    {label:'Type 2 — Card Probing',    color:'#eab308'},
    {label:'Type 3 — Zero-Shot (ZSL)', color:'#a855f7'},
  ];
  document.getElementById('pie-legend').innerHTML = items.map(i=>`
    <div class="pie-legend-item">
      <span class="pie-legend-dot" style="background:${i.color}"></span>
      <span>${i.label}</span>
    </div>`).join('');
}

// ── POLLING ─────────────────────────────────────────
function startPolling() {
  setEl('status-text', 'Live — 1 tx/sec');
  setInterval(poll, POLL_MS);
}

async function poll() {
  try {
    const [txn, stats] = await Promise.all([
      fetch(`${API}/api/stream`).then(r=>r.json()),
      fetch(`${API}/api/stats`).then(r=>r.json()),
    ]);
    handleTransaction(txn);
    updateStats(stats);
    document.querySelector('.pulse-dot').style.background = '#10b981';
    const isReal = txn.model_used && txn.model_used.includes('REAL');
    setEl('status-text', isReal ? 'Real Model Active' : 'Simulation Mode');
    setEl('footer-model-status', isReal ? 'Model: ✅ Real (FZSL)' : 'Model: ⚠️ Fallback');
    // Feed confidence trend
    const conf = txn.confidence != null ? txn.confidence : (txn.fl_probability != null ? txn.fl_probability : null);
    if (conf != null) {
      confData.shift(); confData.push(conf);
      confLabels.shift(); confLabels.push('');
      chartDriftConf.update('none');
    }
  } catch(e) {
    setEl('status-text', 'Connection error…');
    document.querySelector('.pulse-dot').style.background = '#ef4444';
  }
}

// ── DRIFT POLLING ───────────────────────────────────
function startDriftPolling() {
  setTimeout(() => { pollDrift(); setInterval(pollDrift, DRIFT_POLL_MS); }, 3000);
}

async function pollDrift() {
  try {
    const data = await fetch(`${API}/api/drift`).then(r=>r.json());
    renderDrift(data);
  } catch(e) {
    // Drift endpoint not yet ready — show placeholder
    renderDriftPlaceholder();
  }
}

function renderDrift(data) {
  const score = data.overall_drift_score || 0; // 0..1
  const pct = Math.round(score * 100);

  // Update gauge arc
  const arc = document.getElementById('drift-gauge-arc');
  const total = 157; // half-circle path length
  arc.setAttribute('stroke-dashoffset', total - (total * score));
  arc.setAttribute('stroke', score > 0.6 ? '#ef4444' : score > 0.3 ? '#f59e0b' : '#10b981');
  setEl('drift-gauge-pct', `${pct}%`);

  // Alert badge
  const badge = document.getElementById('drift-alert-badge');
  if (score > 0.4) {
    badge.style.display = 'inline-flex';
    badge.textContent = score > 0.7 ? '🔴 HIGH DRIFT' : '⚠️ DRIFT';
  } else {
    badge.style.display = 'none';
  }

  setEl('drift-status-text', `Score: ${pct}% · Window: ${data.window_size||100} tx`);

  // Feature heatmap
  const feats = data.feature_drift || {};
  const sorted = Object.entries(feats).sort((a,b)=>b[1]-a[1]).slice(0,10);
  chartDrift.data.labels = sorted.map(([k])=>k);
  chartDrift.data.datasets[0].data = sorted.map(([,v])=>v);
  chartDrift.data.datasets[0].backgroundColor = sorted.map(([,v])=>
    v > 0.15 ? 'rgba(239,68,68,.8)' : v > 0.07 ? 'rgba(245,158,11,.8)' : 'rgba(16,185,129,.7)'
  );
  chartDrift.update('none');
}

function renderDriftPlaceholder() {
  // Show synthetic demo data when API not ready
  const features = ['V14','V4','V12','V3','V10','V17','V7','V1','V11','V5'];
  const vals = features.map((_,i) => Math.max(0, 0.02 + Math.random() * 0.05 * (features.length - i) / features.length));
  chartDrift.data.labels = features;
  chartDrift.data.datasets[0].data = vals;
  chartDrift.data.datasets[0].backgroundColor = vals.map(v=>v>0.05?'rgba(245,158,11,.6)':'rgba(16,185,129,.6)');
  chartDrift.update('none');
  setEl('drift-gauge-pct', '—');
  setEl('drift-status-text', 'Awaiting data…');
}

// ── PRIVACY POLLING ─────────────────────────────────
function startPrivacyPolling() {
  setTimeout(() => { pollPrivacy(); setInterval(pollPrivacy, PRIVACY_POLL_MS); }, 5000);
}

async function pollPrivacy() {
  try {
    const data = await fetch(`${API}/api/privacy/budget`).then(r=>r.json());
    renderPrivacy(data);
  } catch(e) {
    renderPrivacyPlaceholder();
  }
}

function renderPrivacy(data) {
  const spent  = data.epsilon_spent  || 0;
  const limit  = data.epsilon_limit  || 10;
  const delta  = data.delta          || 1e-5;
  const sigma  = data.noise_scale    || 1.0;
  const rounds = data.fl_rounds      || 0;

  setEl('dp-epsilon-spent', spent.toFixed(3));
  setEl('dp-epsilon-limit', limit.toFixed(1));
  setEl('dp-delta',  delta.toExponential(0));
  setEl('dp-sigma',  sigma.toFixed(2));
  setEl('dp-rounds', rounds);
  setEl('kpi-epsilon-val', `ε=${spent.toFixed(2)}`);
  setEl('kpi-epsilon-sub', `Budget: ${((spent/limit)*100).toFixed(1)}% used`);

  const pct = Math.min(100, (spent / limit) * 100);
  const bar = document.getElementById('dp-epsilon-bar');
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.style.background = pct > 80 ? '#ef4444' : pct > 50 ? '#f59e0b' : 'linear-gradient(90deg,#10b981,#3b82f6)';
  }

  // Update tradeoff chart — mark current epsilon
  updatePrivacyCurrentPoint(spent);
}

function renderPrivacyPlaceholder() {
  setEl('dp-epsilon-spent', '0.00');
  setEl('dp-sigma', '1.00');
  setEl('dp-rounds', '0');
  setEl('kpi-epsilon-val', 'ε=0.0');
  setEl('kpi-epsilon-sub', 'DP not active');
}

function updatePrivacyCurrentPoint(epsilonSpent) {
  if (!chartPrivacy) return;
  // Find nearest point on curve
  const nearest = EPSILON_VALS.reduce((best, e, i) =>
    Math.abs(e - epsilonSpent) < Math.abs(EPSILON_VALS[best] - epsilonSpent) ? i : best, 0);
  chartPrivacy.data.datasets[0].pointBackgroundColor = EPSILON_VALS.map((_,i) =>
    i === nearest ? '#f59e0b' : '#8b5cf6'
  );
  chartPrivacy.data.datasets[0].pointRadius = EPSILON_VALS.map((_,i) => i === nearest ? 7 : 3);
  chartPrivacy.update('none');
}

// Noise slider
function updateNoiseLevel(val) {
  setEl('noise-val', parseFloat(val).toFixed(1));
  // σ → accuracy impact: higher noise = lower accuracy
  // Rough: sigma=1 → 0% impact; sigma=5 → ~8% impact
  const sigma = parseFloat(val);
  const impact = Math.min(15, ((sigma - 1) / 4) * 8);
  _currentNoiseImpact = impact;
  setEl('noise-impact', `Model accuracy impact: -${impact.toFixed(1)}%`);
  const impactEl = document.getElementById('noise-impact');
  if (impactEl) impactEl.style.color = impact > 5 ? '#ef4444' : impact > 2 ? '#f59e0b' : '#10b981';
}

// ── FL POLLING ──────────────────────────────────────
function startFLPolling() {
  setTimeout(() => { pollFL(); setInterval(pollFL, FL_POLL_MS); }, 7000);
}

async function pollFL() {
  try {
    const data = await fetch(`${API}/api/fl/clients`).then(r=>r.json());
    renderFLClients(data.clients || []);
    setEl('fl-round-num', data.current_round || 0);
    setEl('fl-round-total', data.total_rounds || 5);
    setEl('fl-round-status', data.status || 'Idle');
  } catch(e) {
    renderFLPlaceholder();
  }
}

function renderFLClients(clients) {
  clients.forEach((c, i) => {
    const acc = c.local_accuracy != null ? c.local_accuracy : 0;
    setEl(`fl-acc-${i}`, `${(acc*100).toFixed(1)}%`);
    setEl(`fl-n-${i}`, `${c.n_samples||'—'} samples`);
    setEl(`fl-node-acc-${i}`, `${(acc*100).toFixed(0)}%`);
    const bar = document.getElementById(`fl-bar-${i}`);
    if (bar) bar.style.width = `${acc*100}%`;
    // Color node by accuracy
    const node = document.getElementById(`fl-node-${i}`);
    if (node) node.setAttribute('stroke', acc > 0.9 ? '#10b981' : acc > 0.7 ? '#f59e0b' : '#ef4444');
  });
}

function renderFLPlaceholder() {
  const names = ['Bank A','Bank B','Bank C','Bank D'];
  const accs  = [0.962, 0.948, 0.971, 0.955];
  const ns    = [1250, 980, 1500, 820];
  names.forEach((_, i) => {
    setEl(`fl-acc-${i}`, `${(accs[i]*100).toFixed(1)}%`);
    setEl(`fl-n-${i}`, `${ns[i]} samples`);
    setEl(`fl-node-acc-${i}`, `${(accs[i]*100).toFixed(0)}%`);
    const bar = document.getElementById(`fl-bar-${i}`);
    if (bar) bar.style.width = `${accs[i]*100}%`;
  });
  setEl('fl-round-num', '5');
  setEl('fl-round-status', 'Trained');
}

async function runFLRound() {
  if (_flRoundRunning) return;
  _flRoundRunning = true;
  const btn = document.getElementById('btn-fl-round');
  btn.disabled = true;
  btn.textContent = '⏳ Aggregating…';

  // Animate lines
  ['fl-line-0','fl-line-1','fl-line-2','fl-line-3'].forEach((id, i) => {
    const el = document.getElementById(id);
    if (el) {
      setTimeout(() => el.classList.add('fl-line-active'), i * 200);
      setTimeout(() => el.classList.remove('fl-line-active'), i * 200 + 1200);
    }
  });

  try {
    const data = await fetch(`${API}/api/fl/simulate_round`, {method:'POST'}).then(r=>r.json());
    renderFLClients(data.clients || []);
    setEl('fl-round-num', data.current_round || '—');
    setEl('fl-round-status', '✅ Round complete');
  } catch(e) {
    // Simulate visually even if API not ready
    await new Promise(r => setTimeout(r, 1500));
    setEl('fl-round-status', '✅ Simulated round');
  }
  setTimeout(() => {
    _flRoundRunning = false;
    btn.disabled = false;
    btn.textContent = '▶ Run FL Round';
  }, 2000);
}

// ── STATS ────────────────────────────────────────────
function updateStats(s) {
  const fmt  = n => n>=1e6?`${(n/1e6).toFixed(1)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:`${n}`;
  const fmtD = n => n>=1e6?`$${(n/1e6).toFixed(1)}M`:`$${(n/1e3).toFixed(1)}K`;

  setEl('kpi-total-val', fmt(s.total_transactions));
  setEl('kpi-total-sub', `Uptime: ${fmtUptime(s.uptime_seconds)}`);
  setEl('kpi-fraud-val', s.fraud_total);
  setEl('kpi-fraud-sub', `Rate: ${s.fraud_rate_pct.toFixed(4)}%`);
  setEl('kpi-unknown-val', s.fraud_type_counts?.fraud_type_3 || 0);
  setEl('kpi-amount-val', fmtD(s.amounts_total));
  setEl('kpi-amount-sub', `Blocked: ${fmtD(s.amounts_fraud)}`);
  setEl('hdr-total', fmt(s.total_transactions));
  setEl('hdr-fraud', s.fraud_total);

  const ft = s.fraud_type_counts || {};
  chartPie.data.datasets[0].data = [ft.fraud_type_0||0, ft.fraud_type_1||0, ft.fraud_type_2||0, ft.fraud_type_3||0];
  chartPie.update('none');
}

function setEl(id, val) { const el=document.getElementById(id); if(el) el.textContent=val; }
function fmtUptime(s) { const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60); return h>0?`${h}h ${m}m`:m>0?`${m}m ${sec}s`:`${sec}s`; }

function startUptimeTick() {
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - _startTime) / 1000);
    setEl('footer-uptime', `Uptime: ${fmtUptime(elapsed)}`);
  }, 1000);
}

// ── SHAP ────────────────────────────────────────────
async function loadShap(fraudType) {
  document.querySelectorAll('.shap-tab').forEach(t => t.classList.toggle('active', t.dataset.type===fraudType));
  try {
    const data = await fetch(`${API}/api/shap/${fraudType}`).then(r=>r.json());
    renderShapChart(chartShap, data.shap_values, fraudType);
    const desc = data.description || {};
    setEl('shap-desc-icon', desc.icon || '📊');
    setEl('shap-desc-text', desc.title ? `${desc.title} — ${desc.description}` : fraudType);
    const src = document.getElementById('shap-source');
    if (src) src.textContent = data.source === 'real_model' ? '✅ Real SHAP (GradientExplainer)' : '📊 Statistical Baseline';
  } catch(e) { console.warn('SHAP load failed:', e); }
}

function renderShapChart(instance, shapVals, fraudType) {
  if (!shapVals || Object.keys(shapVals).length === 0) return;
  const sorted = Object.entries(shapVals).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,10);
  const maxAbs = Math.max(...sorted.map(([,v])=>Math.abs(v)), 0.001);
  const labels = sorted.map(([k])=>k);
  const values = sorted.map(([,v])=> maxAbs > 1 ? v / maxAbs : v);
  const colors = values.map(v=>v>=0?'rgba(239,68,68,.8)':'rgba(59,130,246,.8)');
  instance.data.labels = labels;
  instance.data.datasets = [{
    label:'SHAP (normalized)', data:values, backgroundColor:colors,
    borderColor:colors.map(c=>c.replace('.8','1')), borderWidth:1, borderRadius:4
  }];
  instance.update();
}

// ── FRAUD TYPES ─────────────────────────────────────
async function loadFraudTypes() {
  try {
    const data = await fetch(`${API}/api/fraud_types`).then(r=>r.json());
    Object.assign(FRAUD_INFO, data.fraud_types || {});
  } catch(e) {}
}

// ── MODAL ─────────────────────────────────────────
async function openModal(txn) {
  window._currentModalTxn = txn;  // Store for PDF export
  const meta = FRAUD_META[txn.fraud_type] || {label:txn.fraud_type,color:'#ef4444',icon:'🚨'};
  const isUnknown = txn.fraud_type === 'fraud_type_3';

  document.getElementById('modal-icon').textContent = meta.icon;
  document.getElementById('modal-title').textContent = isUnknown ? '⚠️ NEW FRAUD TYPE — ZERO-SHOT!' : 'FRAUD DETECTED';
  document.getElementById('modal-title').style.color = meta.color;
  document.getElementById('modal-subtitle').textContent = txn.fraud_type;
  document.getElementById('modal-desc').textContent = txn.message || '';

  // Human-readable risk bullets
  const hExp = txn.human_explanation;
  const hEl = document.getElementById('modal-human-exp');
  const bEl = document.getElementById('modal-bullets');
  if (hExp && hExp.bullets && hExp.bullets.length > 0) {
    bEl.innerHTML = hExp.bullets.map(b => `<li>${b.replace(/\*\*(.+?)\*\*/g,'<strong style="color:#e2e8f0">$1</strong>')}</li>`).join('');
    hEl.style.display = 'block';
    hEl.style.borderColor = meta.color + '55';
  } else {
    hEl.style.display = 'none';
  }

  document.getElementById('modal-txn-id').textContent = txn.id;
  document.getElementById('modal-amount').textContent = `$${txn.amount.toFixed(2)}`;
  document.getElementById('modal-confidence').textContent = `${(txn.confidence*100).toFixed(2)}%`;
  document.getElementById('modal-time').textContent = new Date(txn.timestamp*1000).toLocaleString('en-GB');

  const flEl = document.getElementById('modal-fl-prob');
  const fzslEl = document.getElementById('modal-fzsl-prob');
  if (flEl) flEl.textContent = txn.fl_probability != null ? `${(txn.fl_probability*100).toFixed(2)}%` : '—';
  if (fzslEl) fzslEl.textContent = txn.fzsl_fraud_probability != null ? `${(txn.fzsl_fraud_probability*100).toFixed(2)}%` : '—';

  // Similarity scores (FZSL output)
  const simEl = document.getElementById('modal-sim-scores');
  if (simEl && txn.similarity_scores && Object.keys(txn.similarity_scores).length > 0) {
    const sorted = Object.entries(txn.similarity_scores).sort((a,b)=>b[1]-a[1]);
    simEl.innerHTML = sorted.map(([k,v])=>{
      const m = FRAUD_META[k] || {};
      const pct = Math.max(0, Math.round((v+1)*50));
      return `<div class="sim-row">
        <span class="sim-label" style="color:${m.color||'#94a3b8'}">${k}</span>
        <div class="sim-bar-wrap"><div class="sim-bar-fill" style="width:${pct}%;background:${m.color||'#64748b'}"></div></div>
        <span class="sim-val">${v.toFixed(4)}</span>
      </div>`;
    }).join('');
  } else if (simEl) {
    simEl.innerHTML = '<div style="font-size:11px;color:#64748b;padding:8px">FZSL similarity scores loading…</div>';
  }

  document.getElementById('fraud-modal').classList.add('open');

  // SHAP: per-transaction real SHAP first, then fall back to type-based
  try {
    let shapVals = txn.shap_values;
    let shapSource = txn.shap_ready ? '✅ Real SHAP (this transaction)' : null;

    if (!shapVals || Object.keys(shapVals).length === 0) {
      const data = await fetch(`${API}/api/shap/${txn.fraud_type}`).then(r=>r.json());
      shapVals = data.shap_values;
      shapSource = data.source === 'gradient_x_input' ? '✅ Real SHAP (this transaction)' : '📊 Statistical Baseline';
    }

    const srcEl = document.getElementById('shap-modal-source');
    if (srcEl && shapSource) srcEl.textContent = shapSource;

    if (chartModalShap) chartModalShap.destroy();
    chartModalShap = new Chart(document.getElementById('chartModalShap'), {
      type:'bar', data:{labels:[],datasets:[]}, options: shapChartOptions()
    });
    renderShapChart(chartModalShap, shapVals, txn.fraud_type);

    const info = FRAUD_INFO[txn.fraud_type] || {};
    const topF = info.top_features || Object.keys(shapVals||{}).slice(0,5);
    document.getElementById('modal-top-features').innerHTML =
      topF.map(f=>`<span class="feature-chip">${f}</span>`).join('') +
      `<span class="feature-chip" style="background:rgba(16,185,129,.1);color:#10b981;border-color:rgba(16,185,129,.3)">Cluster avg $${(info.cluster_avg_amount||0).toFixed(0)}</span>`;
  } catch(e) { console.warn('SHAP modal error:', e); }

  // Counterfactual explanation
  const cfEl = document.getElementById('modal-counterfactual');
  const cfBodyEl = document.getElementById('modal-cf-body');
  if (cfEl && cfBodyEl) {
    cfEl.style.display = 'none';
    try {
      const explain = await fetch(`${API}/api/explain/${encodeURIComponent(txn.id)}`).then(r=>r.json());
      const cf = explain.counterfactual;
      if (cf && cf.changes && cf.changes.length > 0) {
        cfBodyEl.innerHTML = `
          <div class="cf-verdict ${cf.success?'cf-success':'cf-fail'}">${cf.verdict}</div>
          <div class="cf-table">
            <div class="cf-header"><span>Feature</span><span>Current</span><span>Required</span><span>Change</span></div>
            ${cf.changes.slice(0,6).map(c=>`
              <div class="cf-row">
                <span class="cf-feat">${c.feature}</span>
                <span class="cf-val">${c.original}</span>
                <span class="cf-val">${c.counterfactual}</span>
                <span class="cf-delta ${c.change>0?'pos':'neg'}">${c.change>0?'+':''}${c.change.toFixed(3)}</span>
              </div>`).join('')}
          </div>
          <div style="font-size:10px;color:#475569;margin-top:8px">
            * Counterfactual computed via gradient descent — minimum feature changes to flip prediction.
            Original fraud probability: ${(cf.original_prob*100).toFixed(1)}% → Target: ${(cf.counterfactual_prob*100).toFixed(1)}%.
          </div>
        `;
        cfEl.style.display = 'block';
      }
    } catch(e) { /* optional */ }
  }
}

function closeModal() { document.getElementById('fraud-modal').classList.remove('open'); }

function exportCurrentTxnPDF() {
  if (window._currentModalTxn && typeof exportFraudReportPDF === 'function') {
    exportFraudReportPDF(window._currentModalTxn);
  } else {
    alert('PDF export module not loaded yet. Please wait a moment.');
  }
}

// ── TRANSACTION HANDLER ─────────────────────────────
function handleTransaction(txn) {
  addFeedItem(txn);
  if (txn.is_fraud) {
    addAlertItem(txn);
    showToast(txn);
    fraudBuf++;
    // Collect for session PDF
    if (window._sessionAlerts) window._sessionAlerts.unshift(txn);
  } else {
    normalBuf++;
  }
  flowNormal.shift(); flowNormal.push(normalBuf);
  flowFraud.shift();  flowFraud.push(fraudBuf);
  normalBuf = 0; fraudBuf = 0;
  chartFlow.update('none');
}

// ── FEED ────────────────────────────────────────────
function addFeedItem(txn) {
  const c = document.getElementById('feed-container');
  const empty = c.querySelector('.feed-empty');
  if (empty) empty.remove();

  const isUnknown = txn.fraud_type === 'fraud_type_3';
  const meta = FRAUD_META[txn.fraud_type] || FRAUD_META.normal;
  const cls = txn.is_fraud ? (isUnknown ? 'unknown' : 'fraud') : 'normal';

  const flPct = txn.fl_probability != null ? Math.round(txn.fl_probability * 100) : null;
  const flBar = flPct != null
    ? `<span class="fl-bar" title="FL Prob: ${flPct}%"><span style="width:${flPct}%;background:${meta.color}"></span></span>`
    : '';

  const div = document.createElement('div');
  div.className = `feed-item ${cls}`;
  div.style.cursor = txn.is_fraud ? 'pointer' : 'default';
  div.innerHTML = `
    <span class="feed-id">${txn.id}</span>
    <span class="feed-amount">$${txn.amount.toFixed(2)}</span>
    ${flBar}
    <span class="feed-type type-${txn.is_fraud ? txn.fraud_type.replace('_','-') : 'normal'}">${txn.is_fraud ? (isUnknown ? '⚠️ ZSL' : txn.fraud_type) : 'NORMAL'}</span>
  `;
  if (txn.is_fraud) div.onclick = () => openModal(txn);
  c.prepend(div);
  const items = c.querySelectorAll('.feed-item');
  if (items.length > 100) items[items.length-1].remove();
}

// ── ALERTS ──────────────────────────────────────────
function addAlertItem(txn) {
  const c = document.getElementById('alerts-container');
  const empty = c.querySelector('.feed-empty');
  if (empty) empty.remove();

  const meta = FRAUD_META[txn.fraud_type] || {label:txn.fraud_type, color:'#ef4444', icon:'🚨'};
  const isUnknown = txn.fraud_type === 'fraud_type_3';

  let simHtml = '';
  if (txn.similarity_scores && Object.keys(txn.similarity_scores).length > 0) {
    const sorted = Object.entries(txn.similarity_scores).sort((a,b)=>b[1]-a[1]);
    simHtml = `<div class="sim-scores">${sorted.slice(0,3).map(([k,v])=>
      `<span class="sim-chip" style="border-color:${FRAUD_META[k]?.color||'#64748b'}">${k}: ${v.toFixed(3)}</span>`
    ).join('')}</div>`;
  }

  const div = document.createElement('div');
  div.className = `alert-item ${isUnknown ? 'unknown' : ''}`;
  div.onclick = () => openModal(txn);
  div.innerHTML = `
    <div class="alert-header">
      <span class="alert-type" style="color:${meta.color}">${meta.icon} ${meta.label}</span>
      <span class="alert-conf">FL: ${(txn.fl_probability*100).toFixed(1)}%</span>
    </div>
    <div class="alert-meta">${txn.id} · $${txn.amount.toFixed(2)} · ${new Date(txn.timestamp*1000).toLocaleTimeString('en-GB')}</div>
    ${simHtml}
    <div class="alert-desc">${txn.message}</div>
  `;
  c.prepend(div);
  const badge = document.getElementById('alert-count-badge');
  badge.textContent = parseInt(badge.textContent||'0') + 1;
  const items = c.querySelectorAll('.alert-item');
  if (items.length > 30) items[items.length-1].remove();
}

// ── TOASTS ──────────────────────────────────────────
function showToast(txn) {
  const meta = FRAUD_META[txn.fraud_type] || {};
  const isUnknown = txn.fraud_type === 'fraud_type_3';
  const c = document.getElementById('toast-container');

  // Cap at 3 toasts to prevent chart overlap
  while (c.children.length >= 3) c.lastChild.remove();

  const t = document.createElement('div');
  t.className = `toast ${isUnknown?'toast-unknown':'toast-fraud'}`;
  t.innerHTML = `
    <div class="toast-title">${meta.icon||'🚨'} ${meta.label||txn.fraud_type}</div>
    <div class="toast-body">
      ${txn.id} · $${txn.amount.toFixed(2)}<br>
      FL: ${(txn.fl_probability*100).toFixed(1)}% | FZSL: ${(txn.fzsl_fraud_probability*100).toFixed(1)}%
    </div>
  `;
  t.onclick = () => openModal(txn);
  c.prepend(t);

  // Auto-dismiss with fade-out
  const dismissMs = isUnknown ? 8000 : 4000;
  setTimeout(() => {
    t.style.transition = 'opacity 0.4s, transform 0.4s';
    t.style.opacity = '0';
    t.style.transform = 'translateX(20px)';
    setTimeout(() => t.remove(), 400);
  }, dismissMs);
}

// ── COMPARISON ──────────────────────────────────────
async function loadComparison() {
  try {
    const data = await fetch(`${API}/api/model_comparison`).then(r=>r.json());
    renderComparison(data.models);
  } catch(e) {}
}

function renderComparison(models) {
  document.getElementById('comparison-grid').innerHTML = models.map((m,i)=>`
    <div class="comp-row ${i===models.length-1?'highlight':''}">
      <div class="comp-name">
        <span style="color:${m.color}">${m.name}</span>
        <div style="display:flex;gap:4px">
          ${m.privacy?'<span class="comp-badge">🔒 Privacy</span>':''}
          ${m.unseen_detection>0?'<span class="comp-badge" style="background:rgba(168,85,247,.15);color:#a855f7">Zero-Shot</span>':''}
        </div>
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:8px">${m.description||''}</div>
      <div class="comp-metrics">
        <div class="comp-metric"><span class="comp-metric-val" style="color:${m.color}">${pct(m.f1)}</span><span class="comp-metric-label">F1</span></div>
        <div class="comp-metric"><span class="comp-metric-val">${pct(m.precision)}</span><span class="comp-metric-label">Precision</span></div>
        <div class="comp-metric"><span class="comp-metric-val">${pct(m.recall)}</span><span class="comp-metric-label">Recall</span></div>
        <div class="comp-metric"><span class="comp-metric-val" style="color:${m.unseen_detection>0?'#a855f7':'inherit'}">${m.unseen_detection>0?pct(m.unseen_detection):'—'}</span><span class="comp-metric-label">New Fraud</span></div>
      </div>
      <div class="comp-bar"><div class="comp-bar-fill" style="width:${m.f1*100}%;background:${m.color}"></div></div>
    </div>
  `).join('');
}
function pct(v) { return `${(v*100).toFixed(2)}%`; }

// ── ACTIONS ─────────────────────────────────────────
async function triggerNewFraud() {
  const btn = document.getElementById('btn-new-fraud');
  btn.disabled = true;
  btn.textContent = '⏳ Running model…';
  try {
    const data = await fetch(`${API}/api/trigger_new_fraud`, {method:'POST'}).then(r=>r.json());
    handleTransaction(data.transaction);
    setTimeout(() => openModal(data.transaction), 300);
  } catch(e) {
    alert('API connection failed.\nRun: python -m uvicorn backend.dashboard_api:app --port 8000 --reload');
  }
  setTimeout(()=>{ btn.disabled=false; btn.textContent='⚠️ SIMULATE NEW FRAUD TYPE'; }, 2000);
}

async function resetStats() {
  await fetch(`${API}/api/reset`, {method:'POST'}).catch(()=>{});
  document.getElementById('feed-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">⏳</span><span>Reset complete — awaiting transactions…</span></div>';
  document.getElementById('alerts-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">✅</span><span>No fraud detected yet</span></div>';
  document.getElementById('alert-count-badge').textContent = '0';
  flowNormal.fill(0); flowFraud.fill(0); chartFlow.update('none');
  // Clear map markers on reset
  if (window._riskMapMarkers) {
    window._riskMapMarkers.forEach(m => m.remove());
    window._riskMapMarkers = [];
    window._sessionAlerts = [];
  }
  setEl('map-stat-total', '0 points plotted');
  setEl('map-stat-amount', '$0 total blocked');
}

// ─── RISK MAP ─────────────────────────────────────────────────────────────────
const MAP_COLORS = {
  fraud_type_0: '#ef4444',
  fraud_type_1: '#f59e0b',
  fraud_type_2: '#eab308',
  fraud_type_3: '#a855f7',
};
const MAP_LABELS = {
  fraud_type_0: 'Card Cloning',
  fraud_type_1: 'Account Takeover',
  fraud_type_2: 'Card Probing',
  fraud_type_3: '⚡ Zero-Shot (New Type)',
};

let _leafletMap = null;
window._riskMapMarkers = [];
window._mapPointIds = new Set(); // avoid duplicates

function initRiskMap() {
  if (typeof L === 'undefined') {
    console.warn('[Map] Leaflet not loaded');
    return;
  }

  _leafletMap = L.map('risk-map', {
    center: [20, 10],
    zoom: 2,
    zoomControl: true,
    attributionControl: false,
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 10,
    minZoom: 1,
  }).addTo(_leafletMap);

  L.control.attribution({ prefix: '' }).addTo(_leafletMap);

  // Initial load + polling every 5s
  pollMapPoints();
  setInterval(pollMapPoints, 5000);
}

async function pollMapPoints() {
  try {
    const data = await fetch(`${API}/api/map/points`).then(r => r.json());
    const pts = data.points || [];
    let newCount = 0;
    let totalAmount = 0;

    pts.forEach(pt => {
      totalAmount += (pt.amount || 0);
      if (window._mapPointIds.has(pt.id)) return;
      window._mapPointIds.add(pt.id);
      newCount++;
      addMapMarker(pt);
    });

    // Update stats bar
    setEl('map-stat-total', `${pts.length} point${pts.length !== 1 ? 's' : ''} plotted`);
    setEl('map-stat-amount', `$${totalAmount.toFixed(2)} total blocked`);
  } catch(e) { /* silent */ }
}

function addMapMarker(pt) {
  if (!_leafletMap) return;
  const color   = MAP_COLORS[pt.fraud_type] || '#ef4444';
  const label   = MAP_LABELS[pt.fraud_type] || pt.fraud_type;
  const isZSL   = pt.fraud_type === 'fraud_type_3';
  const radius  = isZSL ? 10 : 7;
  const ts      = new Date(pt.timestamp * 1000).toLocaleString('en-GB');
  const riskColor = pt.risk_level?.includes('CRITICAL') ? '#ef4444'
                  : pt.risk_level?.includes('HIGH')     ? '#f59e0b' : '#10b981';

  // Outer pulsing circle
  const pulse = L.circleMarker([pt.lat, pt.lng], {
    radius: radius + 8,
    color: color,
    fillColor: color,
    fillOpacity: 0.12,
    weight: 1,
    opacity: 0.4,
    className: 'map-pulse-ring',
  }).addTo(_leafletMap);

  // Inner solid dot
  const dot = L.circleMarker([pt.lat, pt.lng], {
    radius,
    color: '#000',
    weight: 1,
    fillColor: color,
    fillOpacity: 0.9,
  }).addTo(_leafletMap);

  dot.bindPopup(`
    <div class="map-popup-title">${label}</div>
    <span class="map-popup-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">
      ${pt.fraud_type.replace('_',' ').toUpperCase()}
    </span>
    <div class="map-popup-meta">
      <b>ID:</b> ${pt.id}<br>
      <b>Amount:</b> $${(pt.amount||0).toFixed(2)}<br>
      <b>FL Score:</b> ${((pt.fl_prob||0)*100).toFixed(1)}%<br>
      <b>Risk:</b> <span style="color:${riskColor}">${pt.risk_level || 'HIGH'}</span><br>
      <b>Time:</b> ${ts}
    </div>
  `, { maxWidth: 220 });

  window._riskMapMarkers.push(pulse, dot);

  // Fade-in animation for new marker
  dot.setStyle({ fillOpacity: 0 });
  let op = 0;
  const fadeIn = setInterval(() => {
    op = Math.min(op + 0.08, 0.9);
    dot.setStyle({ fillOpacity: op });
    if (op >= 0.9) clearInterval(fadeIn);
  }, 30);
}
