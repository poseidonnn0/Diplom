// Страница истории: график по зонам за период, итоги и выгрузка.
setupCharts();

const form = document.getElementById('period');
const msg = document.getElementById('msg');
const statsBody = document.getElementById('stats');
let chart = null;

function iso(date) {
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return shifted.toISOString().slice(0, 10);
}

function setRange(days) {
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - (days - 1));
  form.start.value = iso(start);
  form.end.value = iso(end);
}

function draw(data) {
  const config = {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: data.series.map((series) => ({
        label: series.name, data: series.values, borderColor: series.color, backgroundColor: series.color,
        borderWidth: 2.5, pointRadius: data.labels.length > 60 ? 0 : 3, pointHitRadius: 8, tension: 0.25, spanGaps: false,
      })),
    },
    options: {
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false }, border: { color: COLORS.grid }, ticks: { maxRotation: 0, autoSkipPadding: 28 } },
        y: { beginAtZero: true, grid: { color: COLORS.grid }, border: { display: false }, ticks: { precision: 0 } },
      },
      plugins: {
        legend: { labels: { color: COLORS.mist, usePointStyle: true, pointStyle: 'rectRounded' } },
        tooltip: { callbacks: { label: (item) => ` ${item.dataset.label} ${num(item.raw, 1)} чел.` } },
      },
    },
  };
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('history'), config);

  statsBody.replaceChildren(...data.stats.map((row) => el('tr', {},
    el('td', { text: row.zone }),
    el('td', { class: 'num', text: num(row.mean, 1) }),
    el('td', { class: 'num', text: row.peak === null ? '—' : String(row.peak) }),
    el('td', { text: row.peak_time || '—' }),
    el('td', { class: 'num', text: pct(row.busy_share, 1) }))));
}

async function load() {
  setMessage(msg, 'Загружаю');
  try {
    const data = await api(`/api/history?start=${form.start.value}&end=${form.end.value}`);
    draw(data);
    const empty = data.series.every((series) => series.values.every((value) => value === null));
    setMessage(msg, empty ? 'За этот период данных нет.' : '');
  } catch (error) {
    setMessage(msg, error.message, 'err');
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); load(); });
form.querySelectorAll('[data-days]').forEach((button) => button.addEventListener('click', () => {
  setRange(Number(button.dataset.days));
  load();
}));
document.querySelectorAll('[data-format]').forEach((button) => button.addEventListener('click', () => {
  const step = document.getElementById('step').value;
  const query = `start=${form.start.value}&end=${form.end.value}&step=${step}&format=${button.dataset.format}`;
  window.location.href = `/api/export?${query}`;
}));

setRange(7);
load();
