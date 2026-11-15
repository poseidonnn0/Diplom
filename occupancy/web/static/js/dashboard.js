// Главная страница: карточки зон, лента «факт и прогноз», типовая неделя.
setupCharts();

const zonesBox = document.getElementById('zones');
const summary = document.getElementById('summary');
const sourcePill = document.getElementById('source');
const modelNote = document.getElementById('model');
const noForecast = document.getElementById('no-forecast');
const weekBox = document.getElementById('week');
const SOURCE_NAMES = { demo: 'Имитация', video: 'Видео', off: 'Сбор выключен' };
const DAYS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

let selected = Number(localStorage.getItem('zone')) || null;
let chart = null;
const cards = new Map();

function hintFor(zone) {
  const tip = zone.insight;
  if (!tip) return '';
  if (zone.status !== 'free' && tip.free_time) return `Свободнее станет около ${tip.free_time}`;
  if (tip.busy_time) return `Заполнится около ${tip.busy_time}`;
  if (tip.peak_value >= 1) return `Пик около ${tip.peak_time}, до ${Math.round(tip.peak_value)} чел.`;
  return 'В ближайшие часы свободно';
}

function renderCards(zones) {
  const seen = new Set();
  for (const zone of zones) {
    seen.add(zone.id);
    let card = cards.get(zone.id);
    if (!card) {
      card = el('button', { class: 'zone', type: 'button', onclick: () => select(zone.id) },
        el('div', { class: 'name' }, el('i'), el('span')),
        el('div', { class: 'count' }, el('b'), el('span')),
        el('div', { class: 'meter' }, el('i')),
        el('div', { class: 'state' }),
        el('div', { class: 'hint' }));
      cards.set(zone.id, card);
      zonesBox.append(card);
    }
    card.dataset.status = zone.status;
    card.setAttribute('aria-pressed', String(zone.id === selected));
    card.querySelector('.name i').style.background = zone.color;
    card.querySelector('.name span').textContent = zone.name;
    card.querySelector('.count b').textContent = zone.count === null ? '—' : zone.count;
    card.querySelector('.count span').textContent = `из ${zone.capacity} мест`;
    card.querySelector('.meter i').style.width = `${Math.min(100, Math.round((zone.load || 0) * 100))}%`;
    card.querySelector('.state').textContent = STATUS[zone.status];
    card.querySelector('.hint').textContent = hintFor(zone);
  }
  for (const [id, card] of cards) if (!seen.has(id)) { card.remove(); cards.delete(id); }
}

async function loadOverview() {
  try {
    const data = await api('/api/overview');
    if (!data.zones.length) {
      summary.textContent = 'Зон пока нет. Добавьте их на странице «Управление».';
      return;
    }
    if (!data.zones.some((zone) => zone.id === selected)) selected = data.zones[0].id;
    renderCards(data.zones);
    summary.textContent = data.total === null
      ? `Свежих подсчётов нет. Обновлено ${data.now}`
      : `В помещении ${data.total} из ${data.capacity} мест занято. Обновлено ${data.now}`;
    const mode = data.source.mode || 'off';
    sourcePill.className = `pill ${data.source.state === 'running' ? 'on' : data.source.state === 'error' ? 'err' : ''}`;
    sourcePill.querySelector('span').textContent =
      data.source.state === 'error' ? `${SOURCE_NAMES[mode]}, ошибка` : SOURCE_NAMES[mode];
    sourcePill.title = data.source.error || '';
    modelNote.textContent = data.model ? `Модель «${data.model.title}», обучена ${data.model.trained}` : '';
    return data;
  } catch (error) {
    summary.textContent = `Не удалось загрузить данные. ${error.message}`;
  }
}

const markers = {
  id: 'markers',
  afterDatasetsDraw(c) {
    const { ctx, chartArea: area, scales: { x, y } } = c;
    const options = c.options.plugins.markers || {};
    ctx.save();
    ctx.font = "600 13px 'Sofia Sans', sans-serif";
    if (options.capacity && options.capacity <= y.max) {
      const py = y.getPixelForValue(options.capacity);
      ctx.strokeStyle = COLORS.sage; ctx.setLineDash([6, 6]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(area.left, py); ctx.lineTo(area.right, py); ctx.stroke();
    }
    if (Number.isInteger(options.now)) {
      const half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
      const px = x.getPixelForValue(options.now) - half;
      ctx.strokeStyle = COLORS.mist; ctx.setLineDash([3, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, area.top + 16); ctx.lineTo(px, area.bottom); ctx.stroke();
      ctx.fillStyle = COLORS.mist; ctx.textAlign = 'center';
      ctx.fillText('сейчас', px, area.top + 10);
    }
    ctx.restore();
  },
};

function drawTimeline(data) {
  noForecast.hidden = data.has_forecast;
  const top = Math.max(data.zone.capacity, ...data.observed.filter((v) => v !== null),
    ...data.hi.filter((v) => v !== null), 1);
  const config = {
    data: {
      labels: data.labels,
      datasets: [
        { type: 'line', label: 'Прогноз', data: data.forecast, borderColor: COLORS.amber, borderWidth: 3,
          pointRadius: 0, pointHitRadius: 8, tension: 0.3, order: 1 },
        { type: 'bar', label: 'Наблюдения', data: data.observed, backgroundColor: COLORS.mist, borderRadius: 3,
          barPercentage: 0.82, categoryPercentage: 1, order: 2 },
        { type: 'line', label: 'hi', data: data.hi, borderWidth: 0, pointRadius: 0, fill: '+1',
          backgroundColor: COLORS.band, tension: 0.3, order: 3 },
        { type: 'line', label: 'lo', data: data.lo, borderWidth: 0, pointRadius: 0, fill: false, tension: 0.3, order: 3 },
      ],
    },
    options: {
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: {
          grid: { display: false }, border: { color: COLORS.grid },
          ticks: {
            autoSkip: false, maxRotation: 0,
            callback(value) {
              const label = data.labels[value];
              if (!label.endsWith(':00')) return null;
              const hour = Number(label.slice(0, 2));
              if (hour === 0) return data.days[value];
              const every = this.chart.width < 640 ? 6 : 3;  // на узком экране подписи реже
              return hour % every === 0 ? label : null;
            },
          },
        },
        y: { beginAtZero: true, suggestedMax: Math.ceil(top * 1.08), grid: { color: COLORS.grid },
             border: { display: false }, ticks: { precision: 0 } },
      },
      plugins: {
        legend: { display: false },
        markers: { now: data.now_index, capacity: data.zone.capacity },
        tooltip: {
          filter: (item) => item.dataset.label === 'Прогноз' || item.dataset.label === 'Наблюдения',
          callbacks: {
            title: (items) => `${data.days[items[0].dataIndex]}, ${items[0].label}`,
            label(item) {
              if (item.raw === null) return null;
              if (item.dataset.label === 'Наблюдения') return ` Наблюдение ${num(item.raw, 1)} чел.`;
              const i = item.dataIndex;
              return ` Прогноз ${num(item.raw, 1)} чел., диапазон ${num(data.lo[i], 1)}–${num(data.hi[i], 1)}`;
            },
          },
        },
      },
    },
    plugins: [markers],
  };
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('timeline'), config);
  document.getElementById('chart-title').textContent = `Факт и прогноз, ${data.zone.name}`;
}

function drawWeek(data) {
  weekBox.replaceChildren(el('div'));
  for (let hour = 0; hour < 24; hour += 1) {
    weekBox.append(el('div', { class: 'h', text: hour % 3 === 0 ? String(hour) : '' }));
  }
  data.cells.forEach((row, day) => {
    weekBox.append(el('div', { class: 'd', text: DAYS[day] }));
    row.forEach((value, hour) => {
      const cell = el('div', { class: 'c' });
      if (value !== null) {
        const share = Math.min(1, Math.max(0, value));
        cell.style.background = `color-mix(in srgb, #0E4439 ${Math.round(share * 100)}%, #E2E9E0)`;
        cell.title = `${DAYS[day]}, ${hour}:00, в среднем ${Math.round(value * 100)} % мест`;
      }
      weekBox.append(cell);
    });
  });
  document.getElementById('week-title').textContent = `Типовая неделя, ${data.zone}`;
}

async function loadZone() {
  if (!selected) return;
  try {
    const [timeline, week] = await Promise.all([
      api(`/api/zones/${selected}/timeline`), api(`/api/zones/${selected}/week`),
    ]);
    drawTimeline(timeline);
    drawWeek(week);
  } catch (error) {
    noForecast.hidden = false;
    noForecast.textContent = `Не удалось загрузить график. ${error.message}`;
  }
}

function select(id) {
  selected = id;
  localStorage.setItem('zone', String(id));
  for (const [zoneId, card] of cards) card.setAttribute('aria-pressed', String(zoneId === id));
  loadZone();
}

(async () => {
  await loadOverview();
  await loadZone();
  setInterval(loadOverview, 5000);
  setInterval(loadZone, 60000);
})();
