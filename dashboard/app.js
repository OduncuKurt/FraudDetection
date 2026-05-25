// ─────────────────────────────────────────────
//  FraudDetection Dashboard — app.js
// ─────────────────────────────────────────────
const API = 'http://localhost:8001';
const POLL_MS = 1000;

// ── Fraud tipi meta (fallback — API'den de alınıyor)
const FRAUD_META = {
  fraud_type_0:      { label: 'Tip 0 — Yüksek Değerli', color: '#ef4444', icon: '💳' },
  fraud_type_1:      { label: 'Tip 1 — Hesap Ele Geçirme', color: '#f59e0b', icon: '🔑' },
  fraud_type_2:      { label: 'Tip 2 — Mikro İşlem', color: '#eccc68', icon: '🔍' },
  UNKNOWN_NEW_FRAUD: { label: '⚠️ YENİ FRAUD TİPİ',       color: '#a855f7', icon: '🚨' },
  normal:            { label: 'Normal', color: '#10b981', icon: '✅' },
};

// ── State
let fraudTypes = {};
let modalShapChart = null;
let currentAlerts = [];

// ── Chart instances
let chartFlow, chartPie, chartShap, chartMetrics;
const FLOW_MAX = 60; // son N veri noktası
const flowLabels = Array.from({ length: FLOW_MAX }, (_, i) => '');
const flowNormal = Array(FLOW_MAX).fill(0);
const flowFraud  = Array(FLOW_MAX).fill(0);

// ─────────────────────────────
//  INIT
// ─────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  await loadFraudTypes();
  await loadComparison();
  loadShap('fraud_type_0');
  startPolling();
});

// ─────────────────────────────
//  CHARTS INIT
// ─────────────────────────────
function initCharts() {
  Chart.defaults.color = '#64748b';
  Chart.defaults.borderColor = '#1e2640';
  Chart.defaults.font.family = "'Inter', sans-serif";

  // Flow chart
  chartFlow = new Chart(document.getElementById('chartFlow'), {
    type: 'line',
    data: {
      labels: flowLabels,
      datasets: [
        { label: 'Normal', data: flowNormal, borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.1)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2 },
        { label: 'Fraud',  data: flowFraud,  borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.1)',  fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2 },
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { maxTicksLimit: 4 } }
      },
      plugins: { legend: { display: false } }
    }
  });

  // Pie chart
  chartPie = new Chart(document.getElementById('chartPie'), {
    type: 'doughnut',
    data: {
      labels: ['Tip 0', 'Tip 1', 'Tip 2', 'Yeni Fraud'],
      datasets: [{
        data: [0, 0, 0, 0],
        backgroundColor: ['#ef4444', '#f59e0b', '#eccc68', '#a855f7'],
        borderColor: '#161b2e',
        borderWidth: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed} adet`
          }
        }
      }
    }
  });
  buildPieLegend();

  // SHAP chart (initialized in loadShap)
  chartShap = new Chart(document.getElementById('chartShap'), {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: shapOptions()
  });

  // Metrics chart
  chartMetrics = new Chart(document.getElementById('chartMetrics'), {
    type: 'bar',
    data: {
      labels: ['Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'],
      datasets: [
        { label: 'Centralized',     data: [0.9289, 0.9388, 0.9338, 0.9991, 0.7741], backgroundColor: 'rgba(100,116,139,.7)', borderRadius: 6 },
        { label: 'Federated (FL)',  data: [0.9373, 1.0000, 0.9676, 1.0000, 0.9942], backgroundColor: 'rgba(59,130,246,.8)',  borderRadius: 6 },
        { label: 'FL + FZSL',       data: [0.9579, 0.9715, 0.9647, 1.0000, 0.9934], backgroundColor: 'rgba(139,92,246,.9)', borderRadius: 6 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0.7, max: 1.02, grid: { color: 'rgba(255,255,255,.04)' }, ticks: { callback: v => (v*100).toFixed(0)+'%', maxTicksLimit: 5 } },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'rectRounded', padding: 16 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y*100).toFixed(2)}%` } }
      }
    }
  });
}

function shapOptions() {
  return {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { callback: v => v.toFixed(2) } },
      y: { grid: { display: false }, ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } } }
    },
    plugins: { legend: { display: false } }
  };
}

// ─────────────────────────────
//  POLLING
// ─────────────────────────────
let pollTimer = null;
let normalBatch = 0, fraudBatch = 0;
let flowTick = 0;

function startPolling() {
  document.getElementById('status-text').textContent = 'Canlı — 1 işlem/sn';
  pollTimer = setInterval(poll, POLL_MS);
}

async function poll() {
  try {
    const [txn, stats] = await Promise.all([
      fetch(`${API}/api/stream`).then(r => r.json()),
      fetch(`${API}/api/stats`).then(r => r.json()),
    ]);
    handleTransaction(txn);
    updateStats(stats);
    document.getElementById('header-status').querySelector('.pulse-dot').style.background = '#10b981';
  } catch (e) {
    document.getElementById('status-text').textContent = 'Bağlantı hatası — yeniden deneniyor…';
    document.getElementById('header-status').querySelector('.pulse-dot').style.background = '#ef4444';
  }
}

// ─────────────────────────────
//  TRANSACTION HANDLER
// ─────────────────────────────
function handleTransaction(txn) {
  if (txn.is_fraud) {
    fraudBatch++;
    addFeedItem(txn);
    addAlertItem(txn);
    showToast(txn);
    if (txn.fraud_type === 'UNKNOWN_NEW_FRAUD') {
      showUnknownFraudBanner();
    }
  } else {
    normalBatch++;
    addFeedItem(txn);
  }

  // Flow chart update
  flowTick++;
  if (flowTick % 1 === 0) {
    flowNormal.shift(); flowNormal.push(normalBatch);
    flowFraud.shift();  flowFraud.push(fraudBatch);
    normalBatch = 0; fraudBatch = 0;
    chartFlow.update('none');
  }
}

// ─────────────────────────────
//  FEED
// ─────────────────────────────
function addFeedItem(txn) {
  const container = document.getElementById('feed-container');
  const empty = container.querySelector('.feed-empty');
  if (empty) empty.remove();

  const typeClass = txnTypeClass(txn.fraud_type);
  const meta = FRAUD_META[txn.fraud_type] || FRAUD_META.normal;
  const div = document.createElement('div');
  div.className = `feed-item ${txn.is_fraud ? (txn.fraud_type === 'UNKNOWN_NEW_FRAUD' ? 'unknown' : 'fraud') : 'normal'}`;
  div.innerHTML = `
    <span class="feed-id">${txn.id}</span>
    <span class="feed-amount">$${txn.amount.toFixed(2)}</span>
    <span class="feed-type ${typeClass}">${txn.fraud_type === 'UNKNOWN_NEW_FRAUD' ? '⚠️ YENİ' : txn.is_fraud ? txn.fraud_type : 'NORMAL'}</span>
  `;
  container.prepend(div);
  // Keep max 80 items
  const items = container.querySelectorAll('.feed-item');
  if (items.length > 80) items[items.length-1].remove();
}

function txnTypeClass(ft) {
  const m = { fraud_type_0:'type-fraud0', fraud_type_1:'type-fraud1', fraud_type_2:'type-fraud2', UNKNOWN_NEW_FRAUD:'type-unknown', normal:'type-normal' };
  return m[ft] || 'type-normal';
}

// ─────────────────────────────
//  ALERTS
// ─────────────────────────────
function addAlertItem(txn) {
  const container = document.getElementById('alerts-container');
  const empty = container.querySelector('.feed-empty');
  if (empty) empty.remove();

  const meta = FRAUD_META[txn.fraud_type] || { label: txn.fraud_type, color: '#ef4444', icon: '🚨' };
  const isUnknown = txn.fraud_type === 'UNKNOWN_NEW_FRAUD';
  const div = document.createElement('div');
  div.className = `alert-item ${isUnknown ? 'unknown' : ''}`;
  div.setAttribute('data-txn', JSON.stringify(txn));
  div.onclick = () => openModal(txn);
  div.innerHTML = `
    <div class="alert-header">
      <span class="alert-type" style="color:${meta.color}">${meta.icon} ${meta.label}</span>
      <span class="alert-conf">${(txn.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="alert-meta">${txn.id} · $${txn.amount.toFixed(2)} · ${new Date(txn.timestamp*1000).toLocaleTimeString('tr')}</div>
    <div class="alert-desc">${txn.message}</div>
  `;
  container.prepend(div);
  const badge = document.getElementById('alert-count-badge');
  badge.textContent = parseInt(badge.textContent||'0') + 1;
  const items = container.querySelectorAll('.alert-item');
  if (items.length > 30) items[items.length-1].remove();

  currentAlerts.unshift(txn);
}

// ─────────────────────────────
//  STATS
// ─────────────────────────────
function updateStats(stats) {
  const fmt = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : n;
  const fmtDollar = n => n >= 1e6 ? '$'+(n/1e6).toFixed(1)+'M' : '$'+(n/1e3).toFixed(1)+'K';

  setEl('kpi-total-val', fmt(stats.total_transactions));
  setEl('kpi-total-sub', `Uptime: ${fmtUptime(stats.uptime_seconds)}`);
  setEl('kpi-fraud-val', stats.fraud_total);
  setEl('kpi-fraud-sub', `Oran: ${stats.fraud_rate_pct.toFixed(4)}%`);
  setEl('kpi-unknown-val', stats.fraud_unknown_count);
  setEl('kpi-amount-val', fmtDollar(stats.amounts_processed));
  setEl('kpi-amount-sub', `Engellenen: ${fmtDollar(stats.fraud_amounts)}`);
  setEl('hdr-total', fmt(stats.total_transactions));
  setEl('hdr-fraud', stats.fraud_total);
  setEl('feed-rate', `~${(stats.total_transactions / Math.max(1, stats.uptime_seconds)).toFixed(1)} işlem/sn`);

  // Pie chart
  const ft = stats.fraud_type_counts || {};
  chartPie.data.datasets[0].data = [
    ft.fraud_type_0 || 0,
    ft.fraud_type_1 || 0,
    ft.fraud_type_2 || 0,
    ft.UNKNOWN_NEW_FRAUD || 0,
  ];
  chartPie.update('none');
}

function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function fmtUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
  return h > 0 ? `${h}s ${m}d` : m > 0 ? `${m}d ${sec}s` : `${sec}s`;
}

// ─────────────────────────────
//  PIE LEGEND
// ─────────────────────────────
function buildPieLegend() {
  const items = [
    { label: 'Tip 0 — Yüksek Değerli', color: '#ef4444' },
    { label: 'Tip 1 — Hesap Ele Geçirme', color: '#f59e0b' },
    { label: 'Tip 2 — Mikro İşlem', color: '#eccc68' },
    { label: 'Yeni Fraud (ZSL)', color: '#a855f7' },
  ];
  const leg = document.getElementById('pie-legend');
  leg.innerHTML = items.map(i => `
    <div class="pie-legend-item">
      <span class="pie-legend-dot" style="background:${i.color}"></span>
      <span>${i.label}</span>
    </div>
  `).join('');
}

// ─────────────────────────────
//  SHAP
// ─────────────────────────────
async function loadShap(fraudType) {
  // Tab highlight
  document.querySelectorAll('.shap-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.type === fraudType);
  });

  try {
    const res = await fetch(`${API}/api/shap/${fraudType}`);
    const data = await res.json();
    renderShapChart(chartShap, data.shap_values, fraudType);

    const desc = data.description || {};
    document.getElementById('shap-desc-icon').textContent = desc.icon || '📊';
    document.getElementById('shap-desc-text').textContent = desc.title ? `${desc.title} — ${desc.description}` : fraudType;
  } catch(e) {
    console.warn('SHAP fetch failed, using local data');
  }
}

function renderShapChart(chartInstance, shapValues, fraudType) {
  const sorted = Object.entries(shapValues).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10);
  const labels = sorted.map(([k]) => k);
  const values = sorted.map(([,v]) => v);
  const colors = values.map(v => v >= 0 ? 'rgba(239,68,68,.8)' : 'rgba(59,130,246,.8)');

  chartInstance.data.labels = labels;
  chartInstance.data.datasets = [{
    label: 'SHAP Değeri',
    data: values,
    backgroundColor: colors,
    borderColor: colors.map(c => c.replace('.8', '1')),
    borderWidth: 1,
    borderRadius: 4,
  }];
  chartInstance.update();
}

// ─────────────────────────────
//  MODAL
// ─────────────────────────────
async function openModal(txn) {
  const modal = document.getElementById('fraud-modal');
  const meta = FRAUD_META[txn.fraud_type] || { label: txn.fraud_type, color: '#ef4444', icon: '🚨' };

  document.getElementById('modal-icon').textContent = meta.icon;
  document.getElementById('modal-title').textContent = txn.fraud_type === 'UNKNOWN_NEW_FRAUD' ? '⚠️ YENİ FRAUD TİPİ TESPİT EDİLDİ!' : 'FRAUD TESPİT EDİLDİ';
  document.getElementById('modal-title').style.color = meta.color;
  document.getElementById('modal-subtitle').textContent = txn.fraud_type;
  document.getElementById('modal-desc').textContent = txn.message;
  document.getElementById('modal-txn-id').textContent = txn.id;
  document.getElementById('modal-amount').textContent = `$${txn.amount.toFixed(2)}`;
  document.getElementById('modal-confidence').textContent = `${(txn.confidence*100).toFixed(2)}%`;
  document.getElementById('modal-time').textContent = new Date(txn.timestamp*1000).toLocaleString('tr');

  modal.classList.add('open');

  // Load SHAP for this transaction
  try {
    const res = await fetch(`${API}/api/shap/${txn.fraud_type}`);
    const data = await res.json();

    if (modalShapChart) modalShapChart.destroy();
    modalShapChart = new Chart(document.getElementById('chartModalShap'), {
      type: 'bar',
      data: { labels: [], datasets: [] },
      options: shapOptions()
    });
    renderShapChart(modalShapChart, data.shap_values, txn.fraud_type);

    // Top features chips
    const desc = data.description || {};
    const features = desc.top_features || Object.keys(data.shap_values).slice(0,5);
    document.getElementById('modal-top-features').innerHTML =
      features.map(f => `<span class="feature-chip">${f}</span>`).join('');
  } catch(e) {}
}

function closeModal() {
  document.getElementById('fraud-modal').classList.remove('open');
}

// ─────────────────────────────
//  TOASTS
// ─────────────────────────────
function showToast(txn) {
  const meta = FRAUD_META[txn.fraud_type] || {};
  const isUnknown = txn.fraud_type === 'UNKNOWN_NEW_FRAUD';
  const container = document.getElementById('toast-container');

  const toast = document.createElement('div');
  toast.className = `toast ${isUnknown ? 'toast-unknown' : 'toast-fraud'}`;
  toast.innerHTML = `
    <div class="toast-title">${meta.icon || '🚨'} ${meta.label || txn.fraud_type}</div>
    <div class="toast-body">${txn.id} · $${txn.amount.toFixed(2)} · Güven: ${(txn.confidence*100).toFixed(1)}%</div>
  `;
  toast.onclick = () => openModal(txn);
  container.prepend(toast);
  setTimeout(() => toast.remove(), isUnknown ? 8000 : 4000);
  if (container.children.length > 5) container.lastChild.remove();
}

// ─────────────────────────────
//  UNKNOWN FRAUD BANNER
// ─────────────────────────────
let bannerShown = false;
function showUnknownFraudBanner() {
  if (bannerShown) return;
  bannerShown = true;
  const toast = document.createElement('div');
  toast.className = 'toast toast-unknown';
  toast.style.cssText = 'border-width:2px; padding: 16px 20px;';
  toast.innerHTML = `
    <div class="toast-title" style="font-size:15px; color:#a855f7">⚠️ YENİ FRAUD TİPİ TESPİT EDİLDİ!</div>
    <div class="toast-body" style="color:#c4b5fd">Zero-Shot Learning aktive oldu. Eğitimde görülmemiş yeni bir dolandırıcılık paterni!</div>
  `;
  document.getElementById('toast-container').prepend(toast);
  setTimeout(() => { toast.remove(); bannerShown = false; }, 10000);
}

// ─────────────────────────────
//  COMPARISON TABLE
// ─────────────────────────────
async function loadComparison() {
  try {
    const data = await fetch(`${API}/api/model_comparison`).then(r => r.json());
    renderComparison(data.models);
  } catch (e) {
    renderComparison([
      { name: 'Centralized MLP', precision: 0.9289, recall: 0.9388, f1: 0.9338, unseen_detection: 0, color: '#64748b', privacy: false },
      { name: 'Federated (FL)',  precision: 0.9373, recall: 1.0000, f1: 0.9676, unseen_detection: 0, color: '#3b82f6', privacy: true  },
      { name: 'FL + FZSL',       precision: 0.9579, recall: 0.9715, f1: 0.9647, unseen_detection: 0.9831, color: '#8b5cf6', privacy: true },
    ]);
  }
}

function renderComparison(models) {
  const grid = document.getElementById('comparison-grid');
  grid.innerHTML = models.map((m, i) => `
    <div class="comp-row ${i === models.length-1 ? 'highlight' : ''}">
      <div class="comp-name">
        <span style="color:${m.color}">${m.name}</span>
        ${m.privacy ? '<span class="comp-badge">🔒 Gizlilik</span>' : ''}
        ${m.unseen_detection > 0 ? '<span class="comp-badge" style="background:rgba(168,85,247,.15);color:#a855f7">Zero-Shot</span>' : ''}
      </div>
      <div class="comp-metrics">
        <div class="comp-metric">
          <span class="comp-metric-val" style="color:${m.color}">${pct(m.f1)}</span>
          <span class="comp-metric-label">F1</span>
        </div>
        <div class="comp-metric">
          <span class="comp-metric-val">${pct(m.precision)}</span>
          <span class="comp-metric-label">Precision</span>
        </div>
        <div class="comp-metric">
          <span class="comp-metric-val" style="color:${m.unseen_detection>0?'#a855f7':'inherit'}">${m.unseen_detection > 0 ? pct(m.unseen_detection) : '—'}</span>
          <span class="comp-metric-label">Yeni Fraud</span>
        </div>
      </div>
      <div class="comp-bar"><div class="comp-bar-fill" style="width:${m.f1*100}%; background:${m.color}"></div></div>
    </div>
  `).join('');
}

function pct(v) { return `${(v*100).toFixed(2)}%`; }

// ─────────────────────────────
//  FRAUD TYPES LOAD
// ─────────────────────────────
async function loadFraudTypes() {
  try {
    const data = await fetch(`${API}/api/fraud_types`).then(r => r.json());
    fraudTypes = data.fraud_types || {};
  } catch(e) {}
}

// ─────────────────────────────
//  ACTIONS
// ─────────────────────────────
async function triggerNewFraud() {
  const btn = document.getElementById('btn-new-fraud');
  btn.disabled = true;
  btn.textContent = '⏳ Simüle ediliyor…';
  try {
    const res = await fetch(`${API}/api/trigger_new_fraud`, { method: 'POST' });
    const data = await res.json();
    handleTransaction(data.transaction);
    showUnknownFraudBanner();
    openModal(data.transaction);
  } catch(e) {
    alert('API bağlantısı kurulamadı. Backend çalışıyor mu?\nuvicorn backend.dashboard_api:app --port 8001');
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = '⚠️ YENİ FRAUD TİPİ SİMÜLE ET'; }, 2000);
}

async function resetStats() {
  await fetch(`${API}/api/reset`, { method: 'POST' }).catch(() => {});
  document.getElementById('feed-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">⏳</span><span>Sıfırlandı…</span></div>';
  document.getElementById('alerts-container').innerHTML = '<div class="feed-empty"><span class="feed-empty-icon">✅</span><span>Henüz fraud tespit edilmedi</span></div>';
  document.getElementById('alert-count-badge').textContent = '0';
  flowNormal.fill(0); flowFraud.fill(0);
  chartFlow.update('none');
}
