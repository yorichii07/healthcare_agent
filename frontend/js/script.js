// ── CONFIG ───────────────────────────────────────────────────
const DEFAULT_API_BASE = window.location.protocol === 'file:'
  ? 'http://127.0.0.1:8000/api'
  : '/api';
const API_BASE = window.__API_BASE__ || DEFAULT_API_BASE;

// ── BACKEND HEALTH CHECK ─────────────────────────────────────
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/`, {
      signal: AbortSignal.timeout(3000)
    });
    if (res.ok) {
      setStatus('online', '✓ Agent is Live');
    } else {
      setStatus('offline', '✗ Backend returned an error. Restart the server.');
    }
  } catch {
    setStatus('offline', '✗ Backend not reachable. Run the FastAPI server and open the app at http://127.0.0.1:8000/');
  }
}

function setStatus(state, msg) {
  document.getElementById('statusBanner').className = `status-banner ${state}`;
  document.getElementById('statusDot').className    = `status-dot ${state}`;
  document.getElementById('statusText').textContent = msg;
}

// ── QUICK GOAL SETTER ────────────────────────────────────────
function setGoal(g) {
  document.getElementById('goalInput').value = g;
}

function updateModeBadge() {
  const modeSelect = document.getElementById('modeSelect');
  const modeBadge = document.getElementById('modeBadge');
  const modeButtons = document.querySelectorAll('.mode-option');
  const value = modeSelect.value;
  const labels = {
    auto: 'Auto',
    ai: 'AI Only',
    fallback: 'Fallback'
  };
  const classes = {
    auto: 'mode-auto',
    ai: 'mode-ai',
    fallback: 'mode-fallback'
  };

  modeBadge.textContent = labels[value] || 'Auto';
  modeBadge.className = `mode-badge ${classes[value] || 'mode-auto'}`;

  modeButtons.forEach(button => {
    button.classList.toggle('active', button.dataset.mode === value);
  });
}

// ── ICON & BADGE HELPERS ─────────────────────────────────────
function typeIcons(type) {
  const icons = { consultation: '👨‍⚕️', lab_test: '🔬', medication: '💊', followup: '📞' };
  return icons[type] || '📋';
}

function badgeClass(status) {
  const classes = {
    validated:         'badge-validated',
    alternative_found: 'badge-alternative_found',
    unavailable:       'badge-unavailable',
    scheduled:         'badge-scheduled'
  };
  return classes[status] || 'badge-pending';
}

function badgeLabel(status) {
  const labels = {
    validated:         '✓ Validated',
    alternative_found: '⚠ Alternative',
    unavailable:       '✗ Unavailable',
    scheduled:         '📅 Scheduled'
  };
  return labels[status] || status;
}

// ── MAIN: CALL PYTHON BACKEND API ────────────────────────────
async function runAgent() {
  const goal = document.getElementById('goalInput').value.trim();
  if (!goal) { showError('Please enter a healthcare goal.'); return; }

  const selectedMode = document.getElementById('modeSelect').value;
  let fallbackNotice = '';

  hideError();
  hideInfo();
  clearResults();
  setLoading(true);

  try {
    let response = await fetch(`${API_BASE}/plan`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ goal, mode: selectedMode })
    });

    if (response.status === 503 && selectedMode === 'ai') {
      // Gracefully degrade when AI provider/runtime is temporarily unavailable.
      fallbackNotice = 'AI mode was temporarily unavailable. Plan generated using Auto mode.';
      response = await fetch(`${API_BASE}/plan`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ goal, mode: 'auto' })
      });
    }

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Backend error');
    }

    const data = await response.json();
    renderResults(data);
    if (fallbackNotice) {
      showInfo(`ℹ ${fallbackNotice}`);
    }

  } catch (err) {
    if (err.message.includes('fetch') || err.message.includes('Failed')) {
      showError('❌ Cannot reach backend.\n\nMake sure the FastAPI server is running and open the app at http://127.0.0.1:8000/');
    } else {
      showError('❌ Error: ' + err.message);
    }
  } finally {
    setLoading(false);
  }
}

// ── RENDER RESULTS ───────────────────────────────────────────
function renderResults(data) {
  const { plan, condition, description } = data;
  const tasks    = plan.tasks;
  const timeline = plan.timeline;

  const nVal = tasks.filter(t => t.status === 'validated').length;
  const nUna = tasks.filter(t => t.status === 'unavailable').length;
  const nAlt = tasks.filter(t => t.status === 'alternative_found').length;

  renderSummary(plan, description, tasks.length, nVal, nUna, nAlt);
  renderTimeline(timeline, tasks);
  renderTaskCards(tasks);
}

function renderSummary(plan, description, total, nVal, nUna, nAlt) {
  document.getElementById('planTitle').textContent   = plan.goal;
  document.getElementById('planDesc').textContent    = description;
  document.getElementById('planSummary').textContent = plan.summary;
  document.getElementById('statRow').innerHTML = `
    <div class="stat">
      <div class="num">${total}</div>
      <div class="lbl">Total Tasks</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--accent3)">${nVal}</div>
      <div class="lbl">Validated</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--warn)">${nAlt}</div>
      <div class="lbl">Alternatives</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--danger)">${nUna}</div>
      <div class="lbl">Unavailable</div>
    </div>
  `;
  document.getElementById('summaryCard').classList.add('visible');
}

function renderTimeline(timeline, tasks) {
  const tlEl = document.getElementById('timeline');
  tlEl.innerHTML = '';

  timeline.forEach((step, i) => {
    const task = tasks.find(t => t.id === step.task_id) || tasks[i];
    const deps = task.dependencies && task.dependencies.length
      ? `Depends on: Task ${task.dependencies.join(', ')}` : null;

    tlEl.innerHTML += `
      <div class="timeline-item" style="animation-delay:${i * 60}ms">
        <div class="tl-dot ${step.type}">${typeIcons(step.type)}</div>
        <div class="tl-content">
          <div class="tl-header">
            <span class="tl-title">${step.description}</span>
            <span class="tl-badge ${badgeClass(task.status)}">${badgeLabel(task.status)}</span>
          </div>
          <div class="tl-meta">
            <span>🕐 ${step.scheduled_time}</span>
            <span>⏱ ${step.duration}</span>
            <span>📍 ${task.resource}</span>
          </div>
          ${task.notes ? `<div class="tl-notes">${task.notes}</div>` : ''}
          ${deps       ? `<div class="tl-dep">🔗 ${deps}</div>` : ''}
        </div>
      </div>`;
  });

  document.getElementById('timelineSection').classList.add('visible');
}

function renderTaskCards(tasks) {
  const gridEl = document.getElementById('tasksGrid');
  gridEl.innerHTML = '';

  tasks.forEach((t, i) => {
    gridEl.innerHTML += `
      <div class="task-card" style="animation-delay:${i * 50}ms">
        <div class="task-card-header">
          <div class="task-num">${t.id}</div>
          <div class="task-title">${t.description}</div>
        </div>
        <div class="task-resource">${typeIcons(t.task_type)} ${t.resource}</div>
        <div class="task-info">
          <span>⏱ ${t.estimated_duration}</span>
          <span>⭐ Priority: ${'★'.repeat(t.priority)}${'☆'.repeat(3 - t.priority)}</span>
          <span class="tl-badge ${badgeClass(t.status)}"
                style="align-self:flex-start;margin-top:4px">
            ${badgeLabel(t.status)}
          </span>
        </div>
      </div>`;
  });

  document.getElementById('tasksSection').classList.add('visible');
}

// ── UI HELPERS ───────────────────────────────────────────────
function clearResults() {
  ['summaryCard', 'timelineSection', 'tasksSection']
    .forEach(id => document.getElementById(id).classList.remove('visible'));
}

function setLoading(on) {
  document.getElementById('btnIcon').innerHTML    = on ? '<div class="spinner"></div>' : '▶';
  document.getElementById('btnText').textContent  = on ? 'Agent Running…' : 'Generate Plan';
  document.getElementById('runBtn').disabled      = on;
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg;
  el.classList.add('visible');
}

function hideError() {
  document.getElementById('errorMsg').classList.remove('visible');
}

function showInfo(msg) {
  const el = document.getElementById('infoMsg');
  el.textContent = msg;
  el.classList.add('visible');
}

function hideInfo() {
  document.getElementById('infoMsg').classList.remove('visible');
}

// ── INIT ─────────────────────────────────────────────────────
document.getElementById('goalInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAgent();
});

document.getElementById('modeSelect').addEventListener('change', updateModeBadge);
document.querySelectorAll('.mode-option').forEach(button => {
  button.addEventListener('click', () => {
    document.getElementById('modeSelect').value = button.dataset.mode;
    updateModeBadge();
  });
});

updateModeBadge();

checkBackend();
