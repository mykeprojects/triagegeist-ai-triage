/**
 * Capa de UI y Gestión del DOM
 */
window.TriageUi = {
  ESI_LABELS: {
    1: 'IMMEDIATE — Life threat',
    2: 'EMERGENT — High risk',
    3: 'URGENT — Needs resources',
    4: 'LESS URGENT',
    5: 'NON-URGENT'
  },

  updateApiUrlDisplay(url) {
    const display = document.getElementById('api-url-display');
    if (display) {
      display.textContent = url;
    }
  },

  updateApiStatus(isOnline) {
    const dot = document.getElementById('api-status-dot');
    if (!dot) return;
    
    const container = dot.parentElement;
    const text = document.getElementById('api-status-text');

    if (isOnline) {
      container.classList.add('api-status-online');
      container.classList.remove('api-status-offline');
      if (text) text.textContent = 'API ONLINE';
    } else {
      container.classList.add('api-status-offline');
      container.classList.remove('api-status-online');
      if (text) text.textContent = 'API OFFLINE';
    }
  },

  showWaitingState() {
    document.getElementById('waiting-state').classList.remove('hidden');
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('result-state').classList.add('hidden');
  },

  showLoadingState() {
    document.getElementById('waiting-state').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');
    document.getElementById('result-state').classList.add('hidden');
  },

  showResultState() {
    document.getElementById('waiting-state').classList.add('hidden');
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('result-state').classList.remove('hidden');
  },

  showError(msg) {
    const el = document.getElementById('error-msg');
    if (el) {
      el.textContent = msg;
      el.classList.remove('hidden');
    }
  },

  hideError() {
    const el = document.getElementById('error-msg');
    if (el) {
      el.classList.add('hidden');
    }
  },

  setSubmitButtonDisabled(disabled) {
    const btn = document.querySelector('.submit-btn');
    if (btn) {
      btn.disabled = disabled;
    }
  },

  getFormData() {
    return {
      systolic_bp: this.getVal('systolic_bp'),
      diastolic_bp: this.getVal('diastolic_bp'),
      heart_rate: this.getVal('heart_rate'),
      respiratory_rate: this.getVal('respiratory_rate'),
      temperature_c: this.getVal('temperature_c'),
      spo2: this.getVal('spo2'),
      pain_score: this.getVal('pain_score'),
      age: this.getVal('age'),
      mental_status_triage: this.getVal('mental_status_triage'),
      arrival_mode: this.getVal('arrival_mode'),
      sex: this.getVal('sex'),
      chief_complaint: this.getVal('chief_complaint')
    };
  },

  getVal(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const v = el.value.trim();
    return v === '' ? null : (el.type === 'number' ? parseFloat(v) : v);
  },

  renderResult(data, payload, clinicalFlags) {
    this.showResultState();

    const esiRaw = Number(data.esi_level);
    const esi = Number.isFinite(esiRaw) ? Math.min(5, Math.max(1, Math.round(esiRaw))) : 1;

    // ESI Número y Label
    const esiNumberEl = document.getElementById('esi-number');
    const esiLabelEl = document.getElementById('esi-label');
    
    // Limpiar clases ESI previas
    for (let i = 1; i <= 5; i++) {
      esiNumberEl.classList.remove(`esi-color-${i}`);
      esiLabelEl.classList.remove(`esi-color-${i}`);
    }

    esiNumberEl.textContent = esi;
    esiNumberEl.classList.add(`esi-color-${esi}`);
    
    esiLabelEl.textContent = this.ESI_LABELS[esi];
    esiLabelEl.classList.add(`esi-color-${esi}`);

    // Barra de confianza
    const conf = Number(data.confidence) || 0;
    const confFillEl = document.getElementById('conf-fill');
    
    for (let i = 1; i <= 5; i++) {
      confFillEl.classList.remove(`esi-bg-${i}`);
    }
    confFillEl.classList.add(`esi-bg-${esi}`);
    confFillEl.style.setProperty('--progress', `${conf}%`);
    
    document.getElementById('conf-text').textContent = `Model confidence: ${conf}%`;

    // Barras de probabilidad por nivel ESI
    const probContainer = document.getElementById('prob-bars');
    probContainer.innerHTML = '<div class="section-label section-label-spaced-14">Probability by ESI level</div>';

    for (let i = 1; i <= 5; i++) {
      const pct = Number(data.probabilities[`ESI_${i}`]) || 0;
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-label">ESI ${i}</span>
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill esi-bg-${i}" style="--progress: ${pct}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      `;
      probContainer.appendChild(row);
    }

    // Flags clínicos
    const notesContainer = document.getElementById('clinical-notes');
    notesContainer.innerHTML = '<div class="section-label section-label-spaced-12">Clinical flags</div>';

    clinicalFlags.forEach(flag => {
      const item = document.createElement('div');
      item.className = 'note-item';
      item.innerHTML = `
        <div class="note-dot ${flag.className}"></div>
        <div class="note-text">${flag.text}</div>
      `;
      notesContainer.appendChild(item);
    });
  }
};
