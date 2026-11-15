// Общие функции для всех страниц.
const COLORS = {
  pine: '#0E4439', mist: '#F4F7F1', amber: '#FFB224', sage: '#A9C4B6',
  band: 'rgba(255,178,36,.28)', grid: 'rgba(244,247,241,.12)',
};
const STATUS = { free: 'Свободно', moderate: 'Умеренная загрузка', busy: 'Занято', unknown: 'Нет данных' };

async function api(url, options = {}) {
  if (options.body && typeof options.body !== 'string') {
    options.body = JSON.stringify(options.body);
    options.headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Ошибка ${response.status}`;
    try {
      const data = await response.json();
      if (typeof data.detail === 'string') detail = data.detail;
      else if (Array.isArray(data.detail)) detail = 'Проверьте заполнение полей';
    } catch (_) { /* тело не JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function pct(value, digits = 0) {
  if (value === null || value === undefined) return '—';
  return `${num(value * 100, digits)} %`;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function setMessage(node, text, kind = '') {
  node.textContent = text || '';
  node.className = `msg ${kind}`.trim();
}

function setupCharts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "'Sofia Sans','Segoe UI',Roboto,Arial,sans-serif";
  Chart.defaults.font.size = 14;
  Chart.defaults.color = COLORS.sage;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
}
