// ─── FraudShield Login JS ─────────────────────────────────────────────────────

const USERS = {
  analyst: { password: 'analyst123', role: 'analyst', name: 'Sarah Chen',      title: 'Fraud Analyst'         },
  admin:   { password: 'admin123',   role: 'admin',   name: 'Alex Morgan',     title: 'System Administrator'  },
  auditor: { password: 'audit123',   role: 'auditor', name: 'David Williams',  title: 'Compliance Auditor'    },
};

const ROLE_PERMS = {
  analyst: { canTriggerFraud: true,  canRunFL: false, canExport: true,  canReset: false, canViewAudit: false },
  admin:   { canTriggerFraud: true,  canRunFL: true,  canExport: true,  canReset: true,  canViewAudit: true  },
  auditor: { canTriggerFraud: false, canRunFL: false, canExport: true,  canReset: false, canViewAudit: true  },
};

let _selectedRole = 'analyst';

// ── Role selector ──────────────────────────────────────────────────────────────
function selectRole(role) {
  _selectedRole = role;
  document.querySelectorAll('.role-pill').forEach(p => {
    p.classList.toggle('active', p.dataset.role === role);
  });
  // Pre-fill username for convenience
  const u = document.getElementById('username');
  if (u.value === '' || USERS[Object.keys(USERS).find(k => USERS[k].name.split(' ')[0].toLowerCase() === u.value.toLowerCase())]) {
    u.value = role === 'admin' ? 'admin' : role === 'auditor' ? 'auditor' : 'analyst';
  }
}

// ── Fill credential from hint ─────────────────────────────────────────────────
function fillCredential(role, password) {
  selectRole(role);
  document.getElementById('username').value = role;
  document.getElementById('password').value = password;
  document.getElementById('username').focus();
}

// ── Toggle password visibility ─────────────────────────────────────────────────
function togglePassword() {
  const pw = document.getElementById('password');
  const btn = document.getElementById('show-pw-btn');
  if (pw.type === 'password') {
    pw.type = 'text';
    btn.textContent = '🙈';
  } else {
    pw.type = 'password';
    btn.textContent = '👁';
  }
}

// ── Handle login ───────────────────────────────────────────────────────────────
function handleLogin(event) {
  event.preventDefault();
  const username = document.getElementById('username').value.trim().toLowerCase();
  const password = document.getElementById('password').value;
  const errEl    = document.getElementById('login-error');
  const errText  = document.getElementById('login-error-text');
  const btn      = document.getElementById('btn-login');
  const btnText  = document.getElementById('btn-login-text');
  const spinner  = document.getElementById('btn-login-spinner');

  // Hide previous error
  errEl.style.display = 'none';

  // Loading state
  btn.disabled = true;
  btnText.textContent = 'Authenticating…';
  spinner.style.display = 'inline';

  // Simulate network delay (makes it feel real)
  setTimeout(() => {
    const user = USERS[username];
    if (!user || user.password !== password) {
      errText.textContent = username in USERS
        ? 'Incorrect password. Try the demo credentials below.'
        : `User "${username}" not found. Use demo credentials below.`;
      errEl.style.display = 'flex';
      btn.disabled = false;
      btnText.textContent = 'Sign In';
      spinner.style.display = 'none';
      document.getElementById('password').value = '';
      return;
    }

    // ── Success — store session ─────────────────────────────────────────────
    const sessionData = {
      username,
      role:        user.role,
      name:        user.name,
      title:       user.title,
      permissions: ROLE_PERMS[user.role],
      // Fake JWT — for visual demo
      token: btoa(JSON.stringify({ sub: username, role: user.role, exp: Date.now() + 3600000 })),
      loginTime: new Date().toISOString(),
    };
    sessionStorage.setItem('fraudshield_session', JSON.stringify(sessionData));

    btnText.textContent = '✅ Authenticated!';
    spinner.style.display = 'none';
    document.querySelector('.login-card-inner').classList.add('success-anim');

    // Redirect to dashboard
    setTimeout(() => {
      window.location.href = 'index.html';
    }, 800);
  }, 900);
}

// ── Keyboard shortcut: Enter to select next field ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();

  // If already logged in, skip to dashboard
  const existing = sessionStorage.getItem('fraudshield_session');
  if (existing) {
    try {
      const s = JSON.parse(existing);
      if (s.token) {
        window.location.href = 'index.html';
        return;
      }
    } catch(_) {}
  }
});

// ── Particle background ────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  const COLORS = ['rgba(99,102,241,', 'rgba(139,92,246,', 'rgba(59,130,246,', 'rgba(16,185,129,'];

  function createParticles(n) {
    return Array.from({ length: n }, () => ({
      x:    Math.random() * W,
      y:    Math.random() * H,
      r:    Math.random() * 1.8 + 0.4,
      vx:   (Math.random() - 0.5) * 0.3,
      vy:   (Math.random() - 0.5) * 0.3,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * 0.5 + 0.1,
    }));
  }

  function drawLine(a, b, dist, maxDist) {
    const alpha = (1 - dist / maxDist) * 0.15;
    ctx.strokeStyle = `rgba(99,102,241,${alpha})`;
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function animate() {
    ctx.clearRect(0, 0, W, H);
    const MAX_DIST = 130;

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;

      // Draw dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + p.alpha + ')';
      ctx.fill();

      // Connect nearby particles
      for (let j = i + 1; j < particles.length; j++) {
        const q   = particles[j];
        const dx  = p.x - q.x;
        const dy  = p.y - q.y;
        const d   = Math.sqrt(dx * dx + dy * dy);
        if (d < MAX_DIST) drawLine(p, q, d, MAX_DIST);
      }
    }
    requestAnimationFrame(animate);
  }

  resize();
  particles = createParticles(80);
  animate();
  window.addEventListener('resize', () => { resize(); particles = createParticles(80); });
}
