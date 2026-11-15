// Страница управления: источник подсчётов, кадр, редактор зон.
const sourceForm = document.getElementById('source');
const sourceMsg = document.getElementById('source-msg');
const statePill = document.getElementById('state');
const stage = document.getElementById('stage');
const frame = document.getElementById('frame');
const placeholder = document.getElementById('placeholder');
const zonesBox = document.getElementById('zones');
const zonesMsg = document.getElementById('zones-msg');
const newZone = document.getElementById('new-zone');
const STATES = { running: 'Работает', starting: 'Запускается', stopped: 'Остановлен', error: 'Ошибка' };

let zones = [];
let selected = null;
let sourceLoaded = false;

function drawZones() {
  stage.querySelectorAll('.z').forEach((node) => node.remove());
  for (const zone of zones) {
    const box = el('div', { class: `z ${zone.id === selected ? 'sel' : ''}` }, el('span', { text: zone.name }));
    box.style.left = `${zone.x1 * 100}%`;
    box.style.top = `${zone.y1 * 100}%`;
    box.style.width = `${(zone.x2 - zone.x1) * 100}%`;
    box.style.height = `${(zone.y2 - zone.y1) * 100}%`;
    box.style.borderColor = zone.color;
    box.querySelector('span').style.background = zone.color;
    stage.append(box);
  }
}

function drawList() {
  zonesBox.replaceChildren(...zones.map((zone) => {
    const radio = el('input', { type: 'radio', name: 'zone', 'aria-label': `Выбрать зону ${zone.name}` });
    radio.checked = zone.id === selected;
    radio.addEventListener('change', () => { selected = zone.id; drawZones(); });
    const name = el('input', { type: 'text', value: zone.name, maxlength: 80, 'aria-label': 'Название' });
    const capacity = el('input', { type: 'number', value: zone.capacity, min: 1, 'aria-label': 'Вместимость' });
    const color = el('input', { type: 'color', value: zone.color, 'aria-label': 'Цвет' });
    name.addEventListener('input', () => { zone.name = name.value; });
    capacity.addEventListener('input', () => { zone.capacity = Number(capacity.value); });
    color.addEventListener('input', () => { zone.color = color.value; drawZones(); });
    return el('div', { class: 'zrow' }, radio, name, capacity, color,
      el('button', { class: 'btn', type: 'button', text: 'Сохранить', onclick: () => save(zone) }),
      el('button', { class: 'btn warn', type: 'button', text: 'Скрыть', onclick: () => hide(zone) }));
  }));
}

async function loadZones() {
  zones = await api('/api/zones');
  if (!zones.some((zone) => zone.id === selected)) selected = zones.length ? zones[0].id : null;
  drawList();
  drawZones();
}

async function save(zone) {
  try {
    await api(`/api/admin/zones/${zone.id}`, { method: 'PUT', body: zone });
    setMessage(zonesMsg, `Зона «${zone.name}» сохранена.`, 'ok');
    await loadZones();
  } catch (error) {
    setMessage(zonesMsg, error.message, 'err');
  }
}

async function hide(zone) {
  if (!window.confirm(`Скрыть зону «${zone.name}»? Накопленная история останется в базе.`)) return;
  try {
    await api(`/api/admin/zones/${zone.id}`, { method: 'DELETE' });
    setMessage(zonesMsg, 'Зона скрыта.', 'ok');
    await loadZones();
  } catch (error) {
    setMessage(zonesMsg, error.message, 'err');
  }
}

newZone.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    const created = await api('/api/admin/zones', {
      method: 'POST',
      body: {
        name: newZone.elements.namedItem('name').value.trim(),
        capacity: Number(newZone.elements.namedItem('capacity').value),
      },
    });
    selected = created.id;
    newZone.reset();
    setMessage(zonesMsg, 'Зона добавлена. Обведите её область на кадре и сохраните.', 'ok');
    await loadZones();
  } catch (error) {
    setMessage(zonesMsg, error.message, 'err');
  }
});

// Рисование области выбранной зоны мышью или пальцем.
let draft = null;
let origin = null;
function relative(event) {
  const rect = stage.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}
stage.addEventListener('pointerdown', (event) => {
  if (selected === null) return;
  origin = relative(event);
  draft = el('div', { class: 'draft' });
  stage.append(draft);
  stage.setPointerCapture(event.pointerId);
});
stage.addEventListener('pointermove', (event) => {
  if (!draft) return;
  const point = relative(event);
  draft.style.left = `${Math.min(origin.x, point.x) * 100}%`;
  draft.style.top = `${Math.min(origin.y, point.y) * 100}%`;
  draft.style.width = `${Math.abs(point.x - origin.x) * 100}%`;
  draft.style.height = `${Math.abs(point.y - origin.y) * 100}%`;
});
stage.addEventListener('pointerup', (event) => {
  if (!draft) return;
  const point = relative(event);
  draft.remove();
  draft = null;
  const zone = zones.find((item) => item.id === selected);
  if (!zone || Math.abs(point.x - origin.x) < 0.02 || Math.abs(point.y - origin.y) < 0.02) return;
  zone.x1 = Math.min(origin.x, point.x); zone.x2 = Math.max(origin.x, point.x);
  zone.y1 = Math.min(origin.y, point.y); zone.y2 = Math.max(origin.y, point.y);
  drawZones();
  setMessage(zonesMsg, `Область зоны «${zone.name}» изменена. Нажмите «Сохранить».`);
});

sourceForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await api('/api/admin/source', { method: 'PUT', body: { mode: sourceForm.mode.value, url: sourceForm.url.value } });
    setMessage(sourceMsg, 'Источник применён, сборщик перезапущен.', 'ok');
    refresh();
  } catch (error) {
    setMessage(sourceMsg, error.message, 'err');
  }
});

async function refresh() {
  try {
    const { source } = await api('/api/admin/status');
    statePill.className = `pill ${source.state === 'running' ? 'on' : source.state === 'error' ? 'err' : ''}`;
    statePill.querySelector('span').textContent = `Сборщик: ${STATES[source.state] || source.state}`;
    if (!sourceLoaded) {
      sourceForm.mode.value = source.mode;
      sourceForm.url.value = source.url || '';
      sourceLoaded = true;
    }
    if (source.error) setMessage(sourceMsg, source.error, 'err');
    if (source.has_video) {
      frame.onload = () => { frame.style.display = 'block'; placeholder.style.display = 'none'; };
      frame.src = `/api/admin/frame.jpg?t=${Date.now()}`;
    } else {
      frame.style.display = 'none';
      placeholder.style.display = 'flex';
    }
  } catch (error) {
    setMessage(sourceMsg, error.message, 'err');
  }
}

loadZones();
refresh();
setInterval(refresh, 3000);
