// Страница моделей: таблица метрик, ошибка по горизонтам, запуск переобучения.
setupCharts();

const runsBody = document.getElementById('runs');
const thead = document.getElementById('thead');
const trainButton = document.getElementById('train');
const trainMsg = document.getElementById('train-msg');
const PALETTE = ['#A9C4B6', '#F4F7F1', '#7CC6A4', '#FFB224', '#F28F6B'];
let chart = null;
let timer = null;

function render(data) {
  const pill = document.getElementById('data');
  pill.querySelector('span').textContent = data.data.rows
    ? `История ${data.data.start} – ${data.data.end}, ${data.data.rows.toLocaleString('ru-RU')} записей`
    : 'Истории пока нет';
  pill.className = `pill ${data.data.rows ? 'on' : ''}`;
  document.getElementById('no-runs').hidden = data.runs.length > 0;
  document.getElementById('trained').textContent = data.trained
    ? `Обучение от ${data.trained}. Шаг прогноза ${data.freq_minutes} минут, горизонт ${data.horizon_steps * data.freq_minutes / 60} часов.`
    : '';
  if (!data.runs.length) { thead.replaceChildren(); runsBody.replaceChildren(); return; }

  const buckets = Object.keys(data.runs[0].metrics.by_horizon || {});
  thead.replaceChildren(el('tr', {},
    el('th', { text: 'Модель' }), el('th', { class: 'num', text: 'MAE' }), el('th', { class: 'num', text: 'RMSE' }),
    el('th', { class: 'num', text: 'WAPE' }),
    ...buckets.map((label) => el('th', { class: 'num', text: `MAE, ${label}` })),
    el('th', { class: 'num', text: `Покрытие, цель ${Math.round(data.coverage_target * 100)} %` }),
    el('th', { class: 'num', text: 'Выигрыш' })));
  const sorted = [...data.runs].sort((a, b) => a.metrics.mae - b.metrics.mae);
  runsBody.replaceChildren(...sorted.map((run) => {
    const m = run.metrics;
    const name = el('td', { text: run.title });
    if (run.active) name.append(el('span', { class: 'tag', text: 'используется' }));
    return el('tr', { class: run.active ? 'best' : '' }, name,
      el('td', { class: 'num', text: num(m.mae, 3) }), el('td', { class: 'num', text: num(m.rmse, 3) }),
      el('td', { class: 'num', text: pct(m.wape, 1) }),
      ...buckets.map((label) => el('td', { class: 'num', text: num(m.by_horizon[label], 3) })),
      el('td', { class: 'num', text: pct(m.coverage, 1) }),
      el('td', { class: 'num', text: m.skill === undefined ? '—' : pct(m.skill, 1) }));
  }));

  const config = {
    type: 'bar',
    data: {
      labels: buckets,
      datasets: data.runs.map((run, index) => ({
        label: run.title, data: buckets.map((label) => run.metrics.by_horizon[label]),
        backgroundColor: PALETTE[index % PALETTE.length], borderRadius: 4,
      })),
    },
    options: {
      scales: {
        x: { grid: { display: false }, border: { color: COLORS.grid } },
        y: { beginAtZero: true, grid: { color: COLORS.grid }, border: { display: false },
             title: { display: true, text: 'MAE, человек' } },
      },
      plugins: {
        legend: { labels: { color: COLORS.mist, usePointStyle: true, pointStyle: 'rectRounded' } },
        tooltip: { callbacks: { label: (item) => ` ${item.dataset.label} ${num(item.raw, 3)}` } },
      },
    },
  };
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('horizons'), config);
}

function showTraining(training) {
  if (!trainMsg) return;
  const running = training.state === 'running';
  trainButton.disabled = running;
  if (training.state === 'idle') setMessage(trainMsg, '');
  else if (running) setMessage(trainMsg, `Идёт с ${training.started}. ${training.message || ''}`);
  else setMessage(trainMsg, `${training.message || ''} Завершено в ${training.finished}.`, training.state === 'error' ? 'err' : 'ok');
  if (running && !timer) timer = setInterval(load, 4000);
  if (!running && timer) { clearInterval(timer); timer = null; }
}

async function load() {
  try {
    const data = await api('/api/models');
    render(data);
    showTraining(data.training);
  } catch (error) {
    document.getElementById('trained').textContent = `Не удалось загрузить. ${error.message}`;
  }
}

if (trainButton) {
  trainButton.addEventListener('click', async () => {
    try {
      const training = await api('/api/admin/train', {
        method: 'POST', body: { include_neural: document.getElementById('neural').checked },
      });
      showTraining(training);
    } catch (error) {
      setMessage(trainMsg, error.message, 'err');
    }
  });
}

load();
