/**
 * Capa de API y Comunicación de Red
 */
window.TriageApi = {
  candidates: [],
  activeUrl: '',

  init() {
    this.candidates = this.getApiCandidates();
    this.activeUrl = this.candidates[0] || 'http://localhost:8000';
  },

  normalizeBaseUrl(url) {
    return String(url || '').trim().replace(/\/+$/, '');
  },

  getApiCandidates() {
    const candidates = [];
    const urlParams = new URLSearchParams(window.location.search);
    const apiFromQuery = this.normalizeBaseUrl(urlParams.get('api'));
    const apiFromStorage = this.normalizeBaseUrl(localStorage.getItem('triage_api_url'));

    if (apiFromQuery) candidates.push(apiFromQuery);
    if (apiFromStorage) candidates.push(apiFromStorage);

    if (window.location.protocol.startsWith('http') && window.location.port === '8000') {
      candidates.push(this.normalizeBaseUrl(window.location.origin));
    }

    candidates.push('http://127.0.0.1:8000', 'http://localhost:8000');

    return [...new Set(candidates.filter(Boolean))];
  },

  setActiveApiUrl(url) {
    this.activeUrl = this.normalizeBaseUrl(url);
    localStorage.setItem('triage_api_url', this.activeUrl);
    
    // Notifica a la UI si está cargada
    if (window.TriageUi && typeof window.TriageUi.updateApiUrlDisplay === 'function') {
      window.TriageUi.updateApiUrlDisplay(this.activeUrl);
    }
  },

  async apiFetch(path, options) {
    let lastError = new Error('No API endpoint available');

    for (const baseUrl of this.candidates) {
      try {
        const response = await fetch(`${baseUrl}${path}`, options);
        this.setActiveApiUrl(baseUrl);
        return response;
      } catch (err) {
        lastError = err;
      }
    }

    throw lastError;
  },

  async checkApiHealth() {
    try {
      const res = await this.apiFetch('/health');
      if (!res.ok) throw new Error(`status ${res.status}`);
      return { online: true, url: this.activeUrl };
    } catch (err) {
      return { online: false, url: this.activeUrl };
    }
  },

  async sendPredictRequest(payload) {
    const res = await this.apiFetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return await res.json();
  }
};

window.TriageApi.init();
