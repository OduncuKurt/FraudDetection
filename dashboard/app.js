// FraudDetection Dashboard — app.js (Gerçek Model Entegrasyonu)
const API = 'http://localhost:8001';
const POLL_MS = 1000;

const FRAUD_META = {
  fraud_type_0:  { label: 'Tip 0 — Yüksek Değerli Fraud', color: '#ef4444', icon: '💳' },
  fraud_type_1:  { label: 'Tip 1 — Hesap Ele Geçirme',    color: '#f59e0b', icon: '🔑' },
  fraud_type_2:  { label: 'Tip 2 — Mikro Kart Testi',      color: '#eab308', icon: '🔍' },
  fraud_type_3:  { label: '⚠️ YENİ — Para Aklama (ZSL)',   color: '#a855f7', icon: '🚨' },
  normal:        { label: 'Normal İşlem',                   color: '#10b981', icon: '✅' },
};

let chartFlow, chartPie, chartShap, chartMetrics, chartModalShap;
const FLOW_MAX = 60;
const flowLabels  = Array(FLOW_MAX).fill('');
const flowNormal  = Array(FLOW_MAX).fill(0);
const flowFraud   = Array(FLOW_MAX).fill(0);
let normalBuf = 0, fraudBuf = 0;

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  loadComparison();
  loadShap('fraud_type_0');
  startPolling();
});

// ── CHARTS ──────────────────────────────────────────
function initCharts() {
  Chart.defaults.color = '#64748b';
  Chart.defaults.borderColor = '#1e2640';
  Chart.defaults.font.family = "'Inter', sans-serif";

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

  chartPie = new Chart(document.getElementById('chartPie'), {
    type: 'doughnut',
    data: {
      labels: ['Tip 0', 'Tip 1', 'Tip 2', 'Tip 3 (ZSL)'],
      datasets: [{ data: [0,0,0,0], backgroundColor: ['#ef4444','#f59e0b','#eab308','#a855f7'], borderColor:'#161b2e', borderWidth:3 }]
    },
    options: { responsive:true, maintainAspectRatio:false, cutout:'60%', plugins:{ legend:{display:false}, tooltip:{callbacks:{label:ctx=>` ${ctx.label}: ${ctx.parsed} adet`}} } }
  });

  chartShap = new Chart(document.getElementById('chartShap'), {
    type: 'bar', data: { labels:[], datasets:[] }, options: shapChartOptions()
  });

  chartMetrics = new Chart(document.getElementById('chartMetrics'), {
    type: 'bar',
    data: {
      labels: ['Precision', 'Recall', 'F1 Skoru', 'ROC-AUC', 'PR-AUC'],
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
    {label:'Tip 0 — Yüksek Değerli',color:'#ef4444'},
    {label:'Tip 1 — Hesap Ele Geçirme',color:'#f59e0b'},
    {label:'Tip 2 — Mikro Kart Testi',color:'#eab308'},
    {label:'Tip 3 — ZSL Yeni Fraud',color:'#a855f7'},
  ];
  document.getElementById('pie-legend').innerHTML = items.map(i=>`
    <div class="pie-legend-item">
      <span class="pie-legend-dot" style="background:${i.color}"></span>
      <span>${i.label}</span>
    </div>`).join('');
}

// ── POLLING ─────────────────────────────────────────
function startPolling() {
  setEl('status-text', 'Canlı — 1 işlem/sn');
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
    setEl('status-text', txn.model_used === 'REAL — FL + FZSL' ? '🟢 Gerçek Model Aktif' : '🟡 Simülasyon Modu');
  } catch(e) {
    setEl('status-text', '🔴 Bağlantı hatası…');
    document.querySelector('.pulse-dot').style.background = '#ef4444';
  }
}

// ── TRANSACTION HANDLER ─────────────────────────────
function handleTransaction(txn) {
  addFeedItem(txn);
  if (txn.is_fraud) {
    addAlertItem(txn);
    showToast(txn);
    fraudBuf++;
  } else {
    normalBuf++;
  }
  // Flow chart
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

  // FL olasılık çubuğu — gerçek model bilgisi
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

  // Similarity skorları (gerçek model çıktısı)
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
    <div class="alert-meta">${txn.id} · $${txn.amount.toFixed(2)} · ${new Date(txn.timestamp*1000).toLocaleTimeString('tr')}</div>
    ${simHtml}
    <div class="alert-desc">${txn.message}</div>
  `;
  c.prepend(div);
  const badge = document.getElementById('alert-count-badge');
  badge.textContent = parseInt(badge.textContent||'0') + 1;
  const items = c.querySelectorAll('.alert-item');
  if (items.length > 30) items[items.length-1].remove();
}

// ── STATS ────────────────────────────────────────────
function updateStats(s) {
  const fmt = n => n>=1e6?`${(n/1e6).toFixed(1)}M`:n>=1e3?`${(n/1e3).toFixed(1)}K`:`${n}`;
  const fmtD = n => n>=1e6?`$${(n/1e6).toFixed(1)}M`:`$${(n/1e3).toFixed(1)}K`;

  setEl('kpi-total-val', fmt(s.total_transactions));
  setEl('kpi-total-sub', `Uptime: ${fmtUptime(s.uptime_seconds)}`);
  setEl('kpi-fraud-val', s.fraud_total);
  setEl('kpi-fraud-sub', `Oran: ${s.fraud_rate_pct.toFixed(4)}%`);
  setEl('kpi-unknown-val', s.fraud_type_counts?.fraud_type_3 || 0);
  setEl('kpi-amount-val', fmtD(s.amounts_total));
  setEl('kpi-amount-sub', `Engellenen: ${fmtD(s.amounts_fraud)}`);
  setEl('hdr-total', fmt(s.total_transactions));
  setEl('hdr-fraud', s.fraud_total);

  const ft = s.fraud_type_counts || {};
  chartPie.data.datasets[0].data = [ft.fraud_type_0||0, ft.fraud_type_1||0, ft.fraud_type_2||0, ft.fraud_type_3||0];
  chartPie.update('none');
}

function setEl(id, val) { const el=document.getElementById(id); if(el) el.textContent=val; }
function fmtUptime(s) { const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60); return h>0?`${h}s ${m}d`:m>0?`${m}d ${sec}s`:`${sec}s`; }

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
    if (src) src.textContent = data.source === 'real_model' ? '✅ Gerçek Model SHAP' : '⚠️ Fallback';
  } catch(e) { console.warn('SHAP yüklenemedi:', e); }
}

function renderShapChart(instance, shapVals, fraudType) {
  const sorted = Object.entries(shapVals).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,10);
  const labels = sorted.map(([k])=>k);
  const values = sorted.map(([,v])=>v);
  const colors = values.map(v=>v>=0?'rgba(239,68,68,.8)':'rgba(59,130,246,.8)');
  instance.data.labels = labels;
  instance.data.datasets = [{
    label:'SHAP',data:values,backgroundColor:colors,borderColor:colors.map(c=>c.replace('.8','1')),borderWidth:1,borderRadius:4
  }];
  instance.update();
}

// ── MODAL ─────────────────────────────────────────
async function openModal(txn) {
  const meta = FRAUD_META[txn.fraud_type] || {label:txn.fraud_type,color:'#ef4444',icon:'🚨'};
  const isUnknown = txn.fraud_type === 'fraud_type_3';

  document.getElementById('modal-icon').textContent = meta.icon;
  document.getElementById('modal-title').textContent = isUnknown ? '⚠️ YENİ FRAUD TİPİ — ZERO-SHOT!' : 'FRAUD TESPİT EDİLDİ';
  document.getElementById('modal-title').style.color = meta.color;
  document.getElementById('modal-subtitle').textContent = txn.fraud_type;
  document.getElementById('modal-desc').textContent = txn.message;
  document.getElementById('modal-txn-id').textContent = txn.id;
  document.getElementById('modal-amount').textContent = `$${txn.amount.toFixed(2)}`;
  document.getElementById('modal-confidence').textContent = `${(txn.confidence*100).toFixed(2)}%`;
  document.getElementById('modal-time').textContent = new Date(txn.timestamp*1000).toLocaleString('tr');

  // Gerçek model skorları
  const flEl = document.getElementById('modal-fl-prob');
  const fzslEl = document.getElementById('modal-fzsl-prob');
  if (flEl) flEl.textContent = txn.fl_probability != null ? `${(txn.fl_probability*100).toFixed(2)}%` : '—';
  if (fzslEl) fzslEl.textContent = txn.fzsl_fraud_probability != null ? `${(txn.fzsl_fraud_probability*100).toFixed(2)}%` : '—';

  // Similarity skorları
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
  }

  document.getElementById('fraud-modal').classList.add('open');

  // Gerçek SHAP
  try {
    const data = await fetch(`${API}/api/shap/${txn.fraud_type}`).then(r=>r.json());
    if (chartModalShap) chartModalShap.destroy();
    chartModalShap = new Chart(document.getElementById('chartModalShap'), {
      type:'bar', data:{labels:[],datasets:[]}, options: shapChartOptions()
    });
    renderShapChart(chartModalShap, data.shap_values, txn.fraud_type);

    const desc = data.description || {};
    const topF = desc.top_features || Object.keys(data.shap_values).slice(0,5);
    document.getElementById('modal-top-features').innerHTML = topF.map(f=>`<span class="feature-chip">${f}</span>`).join('');
  } catch(e) {}
}

function closeModal() { document.getElementById('fraud-modal').classList.remove('open'); }

// ── TOASTS ──────────────────────────────────────────
function showToast(txn) {
  const meta = FRAUD_META[txn.fraud_type] || {};
  const isUnknown = txn.fraud_type === 'fraud_type_3';
  const c = document.getElementById('toast-container');
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
  setTimeout(()=>t.remove(), isUnknown ? 9000 : 4500);
  if (c.children.length > 5) c.lastChild.remove();
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
          ${m.privacy?'<span class="comp-badge">🔒 Gizlilik</span>':''}
          ${m.unseen_detection>0?'<span class="comp-badge" style="background:rgba(168,85,247,.15);color:#a855f7">Zero-Shot</span>':''}
        </div>
      </div>
      <div style="font-size:11px;color:#64748b;margin-bottom:8px">${m.description||''}</div>
      <div class="comp-metrics">
        <div class="comp-metric"><span class="comp-metric-val" style="color:${m.color}">${pct(m.f1)}</span><span class="comp-metric-label">F1</span></div>
        <div class="comp-metric"><span class="comp-metric-val">${pct(m.precision)}</span><span class="comp-metric-label">Precision</span></div>
        <div class="comp-metric"><span class="comp-metric-val">${pct(m.recall)}</span><span class="comp-metric-label">Recall</span></div>
        <div class="comp-metric"><span class="comp-metric-val" style="color:${m.unseen_detection>0?'#a855f7':'inherit'}">${m.unseen_detection>0?pct(m.unseen_detection):'—'}</span><span class="comp-metric-label">Yeni Fraud</span></div>
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
  btn.textContent = '⏳ Model çalışıyor…';
  try {
    const data = await fetch(`${API}/api/trigger_new_fraud`, {method:'POST'}).then(r=>r.json());
    handleTransaction(data.transaction);
    setTimeout(() => openModal(data.transaction), 300);
  } catch(e) {
    alert('API bağlantısı kurulamadı.\nTerminalde: python -m uvicorn backend.dashboard_api:app --port 8001 --reload');
  }
  setTimeout(()=>{ btn.disabled=false; btn.textContent='⚠️ YENİ FRAUD TİPİ SİMÜLE ET'; }, 2000);
}

async function resetStats() {
  await fetch(`${API}/api/reset`, {method:'POST'}).catch(()=>{});
  document.getElementById('feed-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">⏳</span><span>Sıfırlandı…</span></div>';
  document.getElementById('alerts-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">✅</span><span>Henüz fraud tespit edilmedi</span></div>';
  document.getElementById('alert-count-badge').textContent = '0';
  flowNormal.fill(0); flowFraud.fill(0); chartFlow.update('none');
}
