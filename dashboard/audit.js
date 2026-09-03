// ─── FraudShield Audit Log JS ────────────────────────────────────────────────

const API = '';
const PAGE_SIZE = 25;

let _allAlerts  = [];
let _filtered   = [];
let _sortKey    = 'timestamp';
let _sortDir    = -1;       // -1 = desc, 1 = asc
let _page       = 1;
let _liveTimer  = null;
let _liveOn     = true;

const TYPE_META = {
  fraud_type_0: { label: 'Type 0 — Card Cloning',       cls: 'type-0', color: '#ef4444' },
  fraud_type_1: { label: 'Type 1 — Acct Takeover',      cls: 'type-1', color: '#f59e0b' },
  fraud_type_2: { label: 'Type 2 — Card Probing',       cls: 'type-2', color: '#eab308' },
  fraud_type_3: { label: 'Type 3 — Zero-Shot',          cls: 'type-3', color: '#a855f7' },
};

// ── Session ─────────────────────────────────────────────────────────────────
function getSession() {
  try { return JSON.parse(sessionStorage.getItem('fraudshield_session') || '{}'); }
  catch(_) { return {}; }
}
function logout() {
  sessionStorage.removeItem('fraudshield_session');
  window.location.replace('login.html');
}

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const session = getSession();

  // Populate user bar
  const badge = document.getElementById('audit-role-badge');
  const nameEl = document.getElementById('audit-user-name');
  if (badge) {
    badge.textContent = (session.role || 'user').charAt(0).toUpperCase() + (session.role || 'user').slice(1);
    badge.setAttribute('data-role', session.role || 'analyst');
  }
  if (nameEl) nameEl.textContent = session.name || 'Unknown';

  loadAuditLog();
  startLive();
  startFooterTimer();
});

// ── Load from API ────────────────────────────────────────────────────────────
async function loadAuditLog() {
  try {
    const res = await fetch(`${API}/api/audit/log?limit=100`);
    const data = await res.json();
    _allAlerts = data.alerts || [];
    applyFilters();
    updateStats();
  } catch(e) {
    document.getElementById('audit-tbody').innerHTML =
      '<tr><td colspan="9" class="table-loading">⚠️ Could not connect to API. Is the backend running?</td></tr>';
  }
}

// ── Stats strip ───────────────────────────────────────────────────────────────
function updateStats() {
  const total    = _allAlerts.length;
  const critical = _allAlerts.filter(a => (a.risk_level||'').includes('HIGH') || (a.risk_level||'').includes('CRITICAL')).length;
  const zsl      = _allAlerts.filter(a => a.fraud_type === 'fraud_type_3').length;
  const amount   = _allAlerts.reduce((s, a) => s + (a.amount || 0), 0);
  document.getElementById('stat-total').textContent    = total;
  document.getElementById('stat-critical').textContent = critical;
  document.getElementById('stat-zsl').textContent      = zsl;
  document.getElementById('stat-amount').textContent   = `$${amount.toFixed(2)}`;
}

// ── Filters ──────────────────────────────────────────────────────────────────
function applyFilters() {
  const type   = document.getElementById('filter-type').value;
  const risk   = document.getElementById('filter-risk').value;
  const search = document.getElementById('filter-search').value.trim().toLowerCase();

  _filtered = _allAlerts.filter(a => {
    if (type !== 'all' && a.fraud_type !== type) return false;
    if (risk !== 'all' && !(a.risk_level || '').toUpperCase().includes(risk)) return false;
    if (search) {
      const hay = `${a.id} ${a.amount}`.toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });

  sortAlerts();
  _page = 1;
  renderTable();
}

function clearFilters() {
  document.getElementById('filter-type').value   = 'all';
  document.getElementById('filter-risk').value   = 'all';
  document.getElementById('filter-search').value = '';
  applyFilters();
}

// ── Sort ─────────────────────────────────────────────────────────────────────
function sortBy(key) {
  if (_sortKey === key) { _sortDir *= -1; }
  else { _sortKey = key; _sortDir = -1; }
  // Update arrow indicators
  document.querySelectorAll('.sort-arrow').forEach(el => el.textContent = '');
  const arrow = document.getElementById(`sort-${key}`);
  if (arrow) arrow.textContent = _sortDir === -1 ? '↓' : '↑';
  sortAlerts();
  renderTable();
}

function sortAlerts() {
  _filtered.sort((a, b) => {
    let va = a[_sortKey] ?? 0;
    let vb = b[_sortKey] ?? 0;
    if (typeof va === 'string') return va.localeCompare(vb) * _sortDir;
    return (va - vb) * _sortDir;
  });
}

// ── Render table ─────────────────────────────────────────────────────────────
function renderTable() {
  const tbody = document.getElementById('audit-tbody');
  const total = _filtered.length;

  document.getElementById('audit-count').textContent = `${total} event${total !== 1 ? 's' : ''} found`;

  if (total === 0) {
    tbody.innerHTML = `
      <tr><td colspan="9">
        <div class="audit-empty">
          <span class="audit-empty-icon">✅</span>
          <div class="audit-empty-msg">No fraud events match your filters</div>
          <div class="audit-empty-sub">Try clearing the filters or wait for new detections.</div>
        </div>
      </td></tr>`;
    renderPagination(0);
    return;
  }

  const start = (_page - 1) * PAGE_SIZE;
  const slice = _filtered.slice(start, start + PAGE_SIZE);

  tbody.innerHTML = slice.map(a => {
    const meta   = TYPE_META[a.fraud_type] || { label: a.fraud_type, cls: 'type-0', color: '#ef4444' };
    const isZSL  = a.fraud_type === 'fraud_type_3';
    const ts     = new Date((a.timestamp || 0) * 1000).toLocaleString('en-GB', {
      day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit', second:'2-digit'
    });
    const flPct  = ((a.fl_probability || 0) * 100).toFixed(1);
    const fzPct  = ((a.fzsl_fraud_probability || 0) * 100).toFixed(1);
    const flBar  = Math.round(a.fl_probability * 100);
    const fzBar  = Math.round((a.fzsl_fraud_probability || 0) * 100);
    const flCol  = flPct > 80 ? '#ef4444' : flPct > 60 ? '#f59e0b' : '#10b981';
    const fzCol  = fzPct > 80 ? '#ef4444' : fzPct > 60 ? '#f59e0b' : '#10b981';

    const riskRaw = (a.risk_level || 'HIGH').toUpperCase();
    const riskCls = riskRaw.includes('CRITICAL') ? 'risk-critical' : riskRaw.includes('HIGH') ? 'risk-high' : 'risk-medium';
    const riskShort = riskRaw.split(' ')[0];

    const geo = a.geo_location
      ? `<span class="geo-flag">${a.geo_location.lat.toFixed(1)}, ${a.geo_location.lng.toFixed(1)}</span>`
      : '<span class="geo-flag" style="opacity:.3">—</span>';

    return `
      <tr class="${isZSL ? 'zsl-row' : ''}" onclick="openDetail(${JSON.stringify(a).replace(/"/g,'&quot;')})">
        <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b">${ts}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:${meta.color}">${a.id}</td>
        <td><span class="type-badge ${meta.cls}">${isZSL ? '⚡ ' : ''}${meta.label}</span></td>
        <td style="font-weight:700;color:#e2e8f0">$${(a.amount || 0).toFixed(2)}</td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar"><div class="score-bar-fill" style="width:${flBar}%;background:${flCol}"></div></div>
            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;color:${flCol}">${flPct}%</span>
          </div>
        </td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar"><div class="score-bar-fill" style="width:${fzBar}%;background:${fzCol}"></div></div>
            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;color:${fzCol}">${fzPct}%</span>
          </div>
        </td>
        <td><span class="risk-badge ${riskCls}">${riskShort}</span></td>
        <td style="font-size:10px;color:#64748b">${a.model_used || 'FL+FZSL'}</td>
        <td>${geo}</td>
      </tr>`;
  }).join('');

  const footerEnd = Math.min(start + PAGE_SIZE, total);
  document.getElementById('audit-footer-info').textContent =
    `Showing ${start + 1}–${footerEnd} of ${total} events`;
  renderPagination(total);
}

// ── Pagination ────────────────────────────────────────────────────────────────
function renderPagination(total) {
  const pages = Math.ceil(total / PAGE_SIZE);
  const el = document.getElementById('pagination');
  if (pages <= 1) { el.innerHTML = ''; return; }
  let html = '';
  for (let i = 1; i <= pages; i++) {
    html += `<button class="page-btn${i === _page ? ' active' : ''}" onclick="goPage(${i})">${i}</button>`;
  }
  el.innerHTML = html;
}
function goPage(p) { _page = p; renderTable(); window.scrollTo(0, 0); }

// ── Detail slide-in panel ─────────────────────────────────────────────────────
function openDetail(a) {
  if (typeof a === 'string') { try { a = JSON.parse(a); } catch(_) { return; } }

  const meta  = TYPE_META[a.fraud_type] || { label: a.fraud_type, color: '#ef4444' };
  const ts    = new Date((a.timestamp || 0) * 1000).toLocaleString('en-GB');
  const geo   = a.geo_location ? `${a.geo_location.lat.toFixed(4)}, ${a.geo_location.lng.toFixed(4)}` : '—';
  const bullets = (a.human_explanation?.bullets || []).slice(0, 5);

  document.getElementById('detail-title').textContent = a.id;
  document.getElementById('detail-body').innerHTML = `
    <div class="detail-section">
      <div class="detail-section-title">Classification</div>
      <div class="detail-kv"><span class="detail-key">Fraud Type</span><span class="detail-val" style="color:${meta.color}">${meta.label}</span></div>
      <div class="detail-kv"><span class="detail-key">Risk Level</span><span class="detail-val">${a.risk_level || 'HIGH'}</span></div>
      <div class="detail-kv"><span class="detail-key">Verdict</span><span class="detail-val">${a.human_explanation?.verdict || 'FRAUD'}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Transaction</div>
      <div class="detail-kv"><span class="detail-key">Transaction ID</span><span class="detail-val">${a.id}</span></div>
      <div class="detail-kv"><span class="detail-key">Amount</span><span class="detail-val" style="color:#ef4444">$${(a.amount||0).toFixed(2)}</span></div>
      <div class="detail-kv"><span class="detail-key">Timestamp</span><span class="detail-val">${ts}</span></div>
      <div class="detail-kv"><span class="detail-key">Geo Location</span><span class="detail-val">${geo}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Model Scores</div>
      <div class="detail-kv"><span class="detail-key">FL Probability</span><span class="detail-val" style="color:#ef4444">${((a.fl_probability||0)*100).toFixed(2)}%</span></div>
      <div class="detail-kv"><span class="detail-key">FZSL Score</span><span class="detail-val" style="color:#f59e0b">${((a.fzsl_fraud_probability||0)*100).toFixed(2)}%</span></div>
      <div class="detail-kv"><span class="detail-key">Confidence</span><span class="detail-val">${((a.confidence||0)*100).toFixed(2)}%</span></div>
      <div class="detail-kv"><span class="detail-key">Model Used</span><span class="detail-val">${a.model_used || 'FL+FZSL'}</span></div>
    </div>
    ${bullets.length > 0 ? `
    <div class="detail-section">
      <div class="detail-section-title">AI Risk Narrative</div>
      <ul style="list-style:none;padding:0;display:flex;flex-direction:column;gap:6px">
        ${bullets.map(b => `<li style="font-size:11px;color:#94a3b8;padding:6px 8px;background:rgba(255,255,255,.02);border-radius:6px;border-left:2px solid #6366f1">${b.replace(/\*\*(.+?)\*\*/g,'<b style="color:#e2e8f0">$1</b>')}</li>`).join('')}
      </ul>
    </div>` : ''}
  `;
  document.getElementById('detail-overlay').classList.add('open');
  document.getElementById('detail-panel').classList.add('open');
}
function closeDetail() {
  document.getElementById('detail-overlay').classList.remove('open');
  document.getElementById('detail-panel').classList.remove('open');
}

// ── CSV Export ─────────────────────────────────────────────────────────────────
function exportCSV() {
  const headers = ['Timestamp','Transaction ID','Fraud Type','Amount','FL Score','FZSL Score','Risk Level','Model','Lat','Lng'];
  const rows = _filtered.map(a => [
    new Date((a.timestamp||0)*1000).toISOString(),
    a.id,
    a.fraud_type,
    (a.amount||0).toFixed(2),
    ((a.fl_probability||0)*100).toFixed(2)+'%',
    ((a.fzsl_fraud_probability||0)*100).toFixed(2)+'%',
    a.risk_level || 'HIGH',
    a.model_used || 'FL+FZSL',
    a.geo_location?.lat ?? '',
    a.geo_location?.lng ?? '',
  ]);
  const csv = [headers, ...rows].map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `FraudShield_AuditLog_${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Live refresh ──────────────────────────────────────────────────────────────
function toggleLive(on) {
  _liveOn = on;
  if (on) startLive(); else stopLive();
}
function startLive() {
  stopLive();
  _liveTimer = setInterval(async () => {
    await loadAuditLog();
  }, 4000);
}
function stopLive() {
  if (_liveTimer) { clearInterval(_liveTimer); _liveTimer = null; }
}

// ── Footer uptime timer ────────────────────────────────────────────────────────
function startFooterTimer() {
  const start = Date.now();
  setInterval(() => {
    const sec = Math.floor((Date.now() - start) / 1000);
    const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
    const str = h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
    const el = document.getElementById('audit-uptime-footer');
    if (el) el.textContent = `Session active: ${str}`;
  }, 1000);
}
