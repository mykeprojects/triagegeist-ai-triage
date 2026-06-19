/**
 * Capa de Aplicación, Orquestación y Lógica de Negocio
 */
window.TriageApp = {
  async init() {
    // Inicializar visualmente la URL de la API activa
    window.TriageUi.updateApiUrlDisplay(window.TriageApi.activeUrl);

    // Suscribir eventos sin atributos inline en HTML
    const submitBtn = document.getElementById('submit-btn');
    if (submitBtn) {
      submitBtn.addEventListener('click', () => this.predict());
    }

    // Inicializar estados iniciales de paneles y errores con clases
    window.TriageUi.hideError();
    window.TriageUi.showWaitingState();

    // Validar salud de la API al iniciar
    await this.checkApiHealth();
  },

  async checkApiHealth() {
    const health = await window.TriageApi.checkApiHealth();
    window.TriageUi.updateApiStatus(health.online);
  },

  withDefaultNumber(value, fallback) {
    return value === null || Number.isNaN(value) ? fallback : value;
  },

  getClinicalFlags(data, esi) {
    const flags = [];
    
    if (data.systolic_bp !== null && data.systolic_bp < 90) {
      flags.push({ className: 'esi-bg-1', text: `Systolic BP ${data.systolic_bp} mmHg — hypotension` });
    }
    if (data.spo2 !== null && data.spo2 < 94) {
      flags.push({ className: 'esi-bg-2', text: `SpO2 ${data.spo2}% — hypoxemia` });
    }
    if (data.respiratory_rate !== null && data.respiratory_rate >= 22) {
      flags.push({ className: 'esi-bg-2', text: `RR ${data.respiratory_rate} rpm — tachypnea` });
    }
    if (data.temperature_c !== null && data.temperature_c >= 38.5) {
      flags.push({ className: 'esi-bg-3', text: `Temp ${data.temperature_c}°C — fever` });
    }
    if (data.mental_status_triage === 'unresponsive') {
      flags.push({ className: 'esi-bg-1', text: 'Unresponsive — immediate attention required' });
    }
    if (data.pain_score !== null && data.pain_score >= 8) {
      flags.push({ className: 'esi-bg-2', text: `Pain ${data.pain_score}/10 — severe` });
    }
    
    if (flags.length === 0) {
      flags.push({ className: 'esi-bg-4', text: 'No critical flags detected' });
    }
    
    return flags;
  },

  async predict() {
    window.TriageUi.hideError();

    const data = window.TriageUi.getFormData();

    // Lógica de negocio: Validar campos obligatorios
    if (
      data.systolic_bp === null ||
      data.diastolic_bp === null ||
      data.heart_rate === null ||
      data.respiratory_rate === null ||
      data.temperature_c === null ||
      data.spo2 === null ||
      data.age === null
    ) {
      window.TriageUi.showError('Systolic BP, diastolic BP, heart rate, respiratory rate, temperature, SpO2, and age are required fields.');
      return;
    }

    const payload = {
      heart_rate:           this.withDefaultNumber(data.heart_rate, 80),
      respiratory_rate:     this.withDefaultNumber(data.respiratory_rate, 16),
      spo2:                 this.withDefaultNumber(data.spo2, 98),
      systolic_bp:          this.withDefaultNumber(data.systolic_bp, 120),
      diastolic_bp:         this.withDefaultNumber(data.diastolic_bp, 75),
      temperature_c:        this.withDefaultNumber(data.temperature_c, 37.0),
      pain_score:           data.pain_score,
      age:                  this.withDefaultNumber(data.age, 45),
      mental_status_triage: data.mental_status_triage,
      arrival_mode:         data.arrival_mode,
      sex:                  data.sex,
      chief_complaint:      data.chief_complaint || ''
    };

    window.TriageUi.showLoadingState();
    window.TriageUi.setSubmitButtonDisabled(true);

    try {
      const prediction = await window.TriageApi.sendPredictRequest(payload);
      const clinicalFlags = this.getClinicalFlags(payload, prediction.esi_level);
      
      // Delegar renderizado de resultados a la capa UI
      window.TriageUi.renderResult(prediction, payload, clinicalFlags);
    } catch (err) {
      await this.checkApiHealth();
      window.TriageUi.showWaitingState();
      window.TriageUi.showError(`Connection error: ${err.message}. Make sure the API is running.`);
    } finally {
      window.TriageUi.setSubmitButtonDisabled(false);
    }
  }
};

// Auto inicializar al terminar de cargar el DOM
document.addEventListener('DOMContentLoaded', () => {
  window.TriageApp.init();
});
