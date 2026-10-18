const pptxgen = require('pptxgenjs');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT = path.join(ROOT, 'output');
const ASSETS = path.join(ROOT, 'assets');
const REPO = path.resolve(ROOT, '..', '..');
fs.mkdirSync(OUTPUT, { recursive: true });

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Lado';
pptx.company = 'Lado';
pptx.subject = 'Проверка кассовых операций по видеозаписи';
pptx.title = 'Lado: Всё важное на виду';
pptx.lang = 'ru-RU';
pptx.theme = { headFontFace: 'Segoe UI', bodyFontFace: 'Segoe UI', lang: 'ru-RU' };
pptx.defineSlideMaster({ title: 'LADO', background: { color: 'FCFAFB' }, objects: [], margin: 0 });

const C = {
  paper: 'FCFAFB', white: 'FFFFFF', ink: '280914', stage: '1D0610', stage2: '380A20',
  muted: '6C5761', light: 'D8C7CF', rule: 'D9C8CF', magenta: 'A31551', coral: 'FC9385',
  orange: 'FF6A3D', red: 'E54D55', pink: 'CF1D6A', sky: '61CFFF', blue: '0877A7',
};

const logo = path.join(REPO, 'identity', 'logo', 'lado-logo.svg');
const logoStage = path.join(REPO, 'identity', 'logo', 'lado-logo-on-stage.svg');
const markStage = path.join(REPO, 'identity', 'logo', 'lado-mark-on-stage.svg');
const cover = path.join(ASSETS, 'imagegen', 'cash-counter-cover.png');
const macro = path.join(ASSETS, 'imagegen', 'cash-paper-macro.png');
const gradient = path.join(ASSETS, 'svg', 'closing-gradient.svg');
for (const file of [logo, logoStage, markStage, cover, macro, gradient]) {
  if (!fs.existsSync(file)) throw new Error(`Missing asset: ${file}`);
}

function txt(slide, value, x, y, w, h, o = {}) {
  slide.addText(value, {
    x, y, w, h, fontFace: 'Segoe UI', fontSize: o.size || 22,
    bold: o.bold || false, color: o.color || C.ink,
    align: o.align || 'left', valign: o.valign || 'top',
    margin: 0, fit: 'shrink', isTextBox: true,
    objectName: o.name, hyperlink: o.link,
  });
}

function rule(slide, x, y, w, h, color, width = 1, o = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color, width, transparency: o.transparency || 0, beginArrowType: o.begin, endArrowType: o.end },
    objectName: o.name,
  });
}

function field(slide, x, y, w, h, color, transparency = 0, name) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h, fill: { color, transparency }, line: { color, transparency: 100 }, objectName: name,
  });
}

function dot(slide, x, y, d, color, name) {
  slide.addShape(pptx.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color }, line: { color, transparency: 100 }, objectName: name });
}

function heading(slide, value, o = {}) {
  txt(slide, value, o.x ?? 0.72, o.y ?? 0.48, o.w ?? 11.6, o.h ?? 0.86, {
    size: o.size || 44, bold: true, color: o.color || C.ink, name: 'Slide title',
  });
}

function addNotes(slide, speech, sources) {
  slide.addNotes(`${speech}\n\nИсточники слайда: ${sources}`);
}

const speech = [
  'Lado принимает видеозапись кассовой операции и возвращает конкретный эпизод проверки: исходный таймкод, наблюдения, кадры-доказательства и правило. Специалист открывает запись сразу в нужном месте, видит основание и отдельно фиксирует решение.',
  'Начну с результата. В одной демонстрационной записи длительностью двадцать целых девять десятых секунды Lado выделил эпизод с началом на шестнадцатой секунде. В этот момент сотрудник находился у выхода, та же инкассаторская сумка оставалась на столе, а рядом не было видимого сотрудника. Сервис вернул один эпизод и один кадр-доказательство. Этот критерий описывает демонстрационный кейс, а не норму законодательства.',
  'Результат анализа разделён на две части. Система возвращает исходный таймкод, описание наблюдений, основание правила, ограничения и до трёх кадров с временными метками. Специалист отдельно принимает решение, добавляет заметку и при необходимости отмечает, что сцена неразличима. Поэтому автоматическое наблюдение не подменяет внутренний контроль.',
  'Lado связывает участников и кассовые предметы во времени, различает состояние наличных и действие, а затем сохраняет порядок событий. Qwen анализирует окна до тридцати двух кадров, а треки и исходные таймкоды связывают передачу, пересчёт и работу с той же пачкой. Правило получает эту последовательность вместе с ролями, обзором и временным окном.',
  'Восприятие строится в три слоя. RF-DETR Nano находит людей и общие объекты. Обученный RT-DETRv2-r18 распознаёт руки, банкноты и счётчик в составном предметном профиле. Qwen3-VL-8B-Instruct анализирует последовательность действий, после чего программная проверка сверяет идентификаторы объектов и интервалы. CGI на слайде является концептуальной иллюстрацией, а не банковским кадром.',
  'Каждый уровень вывода имеет свою границу. Наличные в руках или на столе являются наблюдаемым состоянием. Передача или пересчёт требуют подтверждённого действия во времени. Правило проверяет роли, порядок, обзор и интервал, а также хранит источник и ограничения применимости. После этого человек сопоставляет эпизод с процедурой и принимает итоговое решение.',
  'Рабочий сервис принимает запись через защищённый API и возвращает результат для проверки. Само видео, веса моделей и inference находятся в частном контуре проекта на Modal L4. Запись не отправляется во внешние AI API. Правила отделены от моделей, а решение специалиста сохраняется отдельно от автоматического наблюдения.',
  'Lado принимает видео через защищённый HTTP API, анализирует его в частном контуре и возвращает проверяемые эпизоды с таймкодом, правилом и кадрами-доказательствами. Рабочий сервис и публичный репозиторий доступны по ссылкам.',
];

// 1. Typographic cover with one original CGI subject.
{
  const s = pptx.addSlide('LADO');
  s.addImage({ path: cover, x: 0, y: 0, w: 13.333, h: 7.5, altText: 'Оригинальная CGI-иллюстрация счётчика и стопок чистой бумаги без банковских данных', objectName: 'Cover CGI' });
  s.addImage({ path: logoStage, x: 0.72, y: 0.54, w: 1.88, h: 0.61, altText: 'Lado', objectName: 'Lado logo' });
  txt(s, 'Lado находит\nэпизод проверки.', 0.72, 1.56, 5.3, 1.9, { size: 54, bold: true, color: C.white, name: 'Cover title' });
  txt(s, 'Видео → таймкод,\nдоказательства и правило', 0.76, 3.52, 5.15, 0.92, { size: 25, bold: true, color: C.white, name: 'Product line' });
  txt(s, 'Специалист сразу открывает нужный момент и принимает решение.', 0.76, 4.58, 5.05, 1.24, { size: 23, color: C.white, name: 'Product promise' });
  txt(s, 'Открыть сервис ↗', 0.76, 6.18, 2.2, 0.4, { size: 20, bold: true, color: C.white, link: { url: 'https://darinax2005x--lado-ladoservice-web.us-east.modal.run/app' }, name: 'Service link' });
  addNotes(s, speech[0], 'contracts/product.json; README.md; identity/logo/lado-logo-on-stage.svg. CGI provenance: assets/imagegen/redesign-prompts.md.');
}

// 2. Large demonstrated onset and three compact facts.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.paper };
  heading(s, 'От записи к эпизоду', { w: 11.6, size: 46 });
  txt(s, '16,0', 0.72, 1.55, 5.5, 1.48, { size: 94, bold: true, color: C.orange, name: 'Onset value' });
  txt(s, 'секунда начала эпизода', 0.78, 3.13, 4.4, 0.45, { size: 25, bold: true, name: 'Onset meaning' });
  txt(s, 'Один демонстрационный сценарий, запись 20,9 с', 0.78, 3.83, 4.95, 0.7, { size: 20, color: C.muted, name: 'Case scope' });

  const facts = [
    ['Сотрудник у выхода', 'движение подтверждено по последовательности'],
    ['Та же сумка на столе', 'предмет остался после ухода'],
    ['Место без сотрудника', 'условия совпали в одном окне'],
  ];
  [1.63, 3.05, 4.47].forEach((y, i) => {
    txt(s, facts[i][0], 7.05, y, 5.3, 0.54, { size: 27, bold: true, color: i === 2 ? C.blue : C.magenta, name: `Case fact ${i + 1}` });
    txt(s, facts[i][1], 7.05, y + 0.56, 5.1, 0.68, { size: 22, color: C.muted, name: `Case detail ${i + 1}` });
  });

  rule(s, 0.82, 6.45, 11.68, 0, C.rule, 4, { name: 'Source timeline' });
  rule(s, 0.82, 6.45, 8.94, 0, C.orange, 4, { name: 'Timeline to onset' });
  dot(s, 9.62, 6.31, 0.28, C.orange, 'Onset marker');
  txt(s, 'начало', 0.78, 6.7, 1.2, 0.38, { size: 20, color: C.muted });
  txt(s, '16,0', 9.27, 6.73, 0.95, 0.3, { size: 20, bold: true, color: C.orange, align: 'center' });
  txt(s, '20,9 с', 11.67, 6.73, 0.82, 0.3, { size: 18, color: C.muted, align: 'right' });
  addNotes(s, speech[1], 'eval/reports/pitch-onset-release.json; eval/reports/modal-question-verification.json; rules/catalog/V-BAG-UNATTENDED.yaml.');
}

// 3. Editorial result spread.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.paper };
  heading(s, 'Что получает специалист', { w: 11.6, size: 46 });
  rule(s, 7.58, 1.55, 0, 5.18, C.rule, 1.5, { name: 'Editorial divider' });

  txt(s, 'Система возвращает', 0.78, 1.62, 5.8, 0.45, { size: 26, bold: true, color: C.magenta });
  txt(s, 'Таймкод', 0.78, 2.35, 2.3, 0.58, { size: 30, bold: true });
  txt(s, 'точка входа в запись', 3.08, 2.32, 4.08, 0.8, { size: 22, color: C.muted });
  txt(s, 'Наблюдения', 0.78, 3.32, 2.3, 0.58, { size: 30, bold: true });
  txt(s, 'участники, предметы, действия', 3.08, 3.29, 4.08, 0.8, { size: 22, color: C.muted });
  txt(s, 'Основание', 0.78, 4.29, 2.3, 0.58, { size: 30, bold: true });
  txt(s, 'условия и ограничения правила', 3.08, 4.26, 4.08, 0.8, { size: 22, color: C.muted });
  txt(s, 'До 3 кадров', 0.78, 5.26, 2.75, 0.62, { size: 34, bold: true, color: C.orange });
  txt(s, 'кадры с временной меткой', 3.38, 5.25, 3.78, 0.8, { size: 22, color: C.muted });

  txt(s, 'Специалист решает', 8.18, 1.62, 4.2, 0.45, { size: 26, bold: true, color: C.magenta });
  txt(s, 'Подтвердить', 8.18, 2.47, 3.5, 0.6, { size: 31, bold: true });
  txt(s, 'Отклонить', 8.18, 3.42, 3.5, 0.6, { size: 31, bold: true });
  txt(s, 'Не видно', 8.18, 4.37, 3.5, 0.6, { size: 31, bold: true, color: C.blue });
  txt(s, 'Добавить заметку', 8.18, 5.48, 3.8, 0.54, { size: 24, bold: true, color: C.muted });
  txt(s, 'Решение хранится отдельно\nот наблюдений.', 8.18, 5.94, 4.15, 0.98, { size: 22, color: C.muted });
  addNotes(s, speech[2], 'README.md; contracts/openapi.json; contracts/schemas/Finding.json; contracts/schemas/Decision.json.');
}

// 4. Full-width temporal graph.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.stage };
  heading(s, 'Связи во времени', { color: C.white, w: 11.6, h: 0.88, size: 46 });
  txt(s, 'Qwen анализирует окна до 32 кадров.\nТреки и исходные таймкоды связывают события.', 0.76, 1.25, 10.9, 0.86, { size: 23, color: C.light });

  const axisX = 3.05, axisW = 9.2;
  const rows = [
    { y: 2.35, label: 'Участники', segments: [[0.1, 0.38, C.sky], [0.52, 0.38, C.coral]], names: ['клиент', 'кассир'] },
    { y: 3.33, label: 'Наличные', segments: [[0.16, 0.68, C.orange]], names: ['та же пачка'] },
    { y: 4.31, label: 'Действия', segments: [[0.2, 0.18, C.coral], [0.49, 0.22, C.sky], [0.74, 0.13, C.orange]], names: ['передача', 'пересчёт', 'упаковка'] },
    { y: 5.29, label: 'Правило', segments: [[0.16, 0.71, C.magenta]], names: ['роли  /  порядок  /  видимость  /  интервал'] },
  ];
  rows.forEach((row, ri) => {
    txt(s, row.label, 0.78, row.y - 0.16, 1.95, 0.4, { size: 23, bold: true, color: C.white });
    rule(s, axisX, row.y, axisW, 0, '6A4758', 2, { name: `${row.label} baseline` });
    row.segments.forEach((seg, si) => {
      const x = axisX + axisW * seg[0]; const w = axisW * seg[1];
      rule(s, x, row.y, w, 0, seg[2], ri === 3 ? 12 : 8, { name: `${row.label} interval ${si + 1}` });
      const label = ri === 3 ? 'роли / порядок / обзор / время' : row.names[si];
      txt(s, label, x, row.y - 0.52, Math.max(w, 2.0), 0.52, { size: 20, bold: true, color: ri === 3 ? C.coral : C.white, align: ri === 3 ? 'center' : 'left' });
    });
  });
  rule(s, axisX, 6.34, axisW, 0, C.light, 1.5, { name: 'Time axis' });
  [0, 0.25, 0.5, 0.75, 1].forEach((v, i) => {
    const x = axisX + axisW * v; rule(s, x, 6.24, 0, 0.2, C.light, 1);
    txt(s, i === 0 ? 'время' : `t${i}`, x - 0.45, 6.56, 0.9, 0.34, { size: 18, color: C.light, align: 'center' });
  });
  addNotes(s, speech[3], 'README.md, sections “Действия во времени” and “Факты и правила”; backend/navidu/perception/grounding.py; rules/catalog/V-COUNT-ABSENT.yaml.');
}

// 5. One macro visual with three model labels.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.paper };
  heading(s, 'Что система видит в кадре', { w: 11.6, size: 44 });
  s.addImage({ path: macro, x: 0.56, y: 1.48, w: 7.28, h: 5.46, altText: 'Оригинальная CGI-макросцена чистой бумаги, пустой карточки и края счётчика без банковских данных', objectName: 'Cash paper macro CGI' });

  txt(s, 'RF-DETR Nano', 8.42, 1.54, 4.2, 0.56, { size: 27, bold: true, color: C.magenta });
  txt(s, 'люди и общие объекты', 8.42, 2.16, 4.2, 0.62, { size: 22, color: C.muted });
  txt(s, 'RT-DETRv2-r18', 8.42, 3.16, 4.2, 0.56, { size: 27, bold: true, color: C.magenta });
  txt(s, 'руки, банкноты, счётчик', 8.42, 3.78, 4.2, 0.62, { size: 22, color: C.muted });
  txt(s, 'Qwen3-VL-8B', 8.42, 4.78, 4.35, 0.56, { size: 27, bold: true, color: C.magenta });
  txt(s, 'последовательность\nи действия\nкод сверяет объекты\nи время', 8.42, 5.34, 4.2, 1.42, { size: 22, color: C.muted });
  addNotes(s, speech[4], 'README.md; backend/navidu/perception/object_detector.py; backend/navidu/perception/qwen.py; backend/navidu/perception/grounding.py. CGI provenance: assets/imagegen/redesign-prompts.md.');
}

// 6. Fact to rule typography table.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.paper };
  heading(s, 'От факта к решению', { w: 11.6, size: 46 });
  const xs = [0.78, 4.45, 8.72];
  const ws = [3.15, 3.78, 3.82];
  ['В кадре', 'Проверяет правило', 'Что остаётся человеку'].forEach((v, i) => txt(s, v, xs[i], 1.4, ws[i], 0.62, { size: 23, bold: true, color: C.magenta }));
  rule(s, 0.78, 1.97, 11.78, 0, C.ink, 2);

  const table = [
    ['Наличные в руках\nили на столе', 'Состояние предмета,\nучастник и зона', 'Сопоставить наблюдение\nс процедурой'],
    ['Передача\nили пересчёт', 'Роли, порядок действий\nи полнота обзора', 'Подтвердить\nлибо отклонить'],
    ['Та же пачка\nмежду кадрами', 'Общая идентичность\nи временной интервал', 'Отметить «не видно»,\nесли связи недостаточно'],
  ];
  [2.18, 3.66, 5.14].forEach((y, r) => {
    table[r].forEach((value, c) => txt(s, value, xs[c], y, ws[c], 1.16, { size: 23, bold: c === 0, color: c === 0 ? C.ink : C.muted }));
    rule(s, 0.78, y + 1.24, 11.78, 0, C.rule, 1);
  });
  txt(s, 'Правило хранит источник, ограничения и кадры-доказательства.', 0.78, 6.64, 10.8, 0.62, { size: 22, bold: true, color: C.blue });
  addNotes(s, speech[5], 'README.md; rules/catalog/V-COUNT-ABSENT.yaml; rules/catalog/V-CASH-UNSECURED.yaml; rules/sources/not-visual.yaml; contracts/schemas/Decision.json.');
}

// 7. Two-sided private inference boundary.
{
  const s = pptx.addSlide('LADO'); s.background = { color: C.stage };
  field(s, 0, 0, 6.66, 7.5, C.stage2, 0, 'Client side');
  heading(s, 'Частный контур анализа', { color: C.white, w: 11.6, size: 46 });
  rule(s, 6.66, 1.55, 0, 4.95, C.coral, 2, { name: 'Private boundary' });

  txt(s, 'Защищённый API', 0.82, 1.88, 4.8, 0.52, { size: 31, bold: true, color: C.white });
  txt(s, 'Запись поступает\nс API ключом', 0.82, 2.58, 4.8, 0.78, { size: 24, color: C.light });
  txt(s, 'Результат возвращается\nдля проверки и решения', 0.82, 4.34, 4.9, 0.88, { size: 24, color: C.light });

  txt(s, 'Private Modal L4', 7.24, 1.88, 4.8, 0.52, { size: 31, bold: true, color: C.white });
  txt(s, 'Видео, веса моделей\nи inference', 7.24, 2.65, 4.7, 0.88, { size: 24, color: C.light });
  txt(s, 'Правила отделены от моделей.\nРешение человека — от наблюдений.', 7.24, 4.22, 4.75, 1.08, { size: 22, color: C.light });

  rule(s, 4.42, 3.42, 4.42, 0, C.orange, 3, { end: 'triangle', name: 'Request line' });
  txt(s, 'запрос', 5.0, 3.02, 1.2, 0.3, { size: 19, bold: true, color: C.coral, align: 'center' });
  rule(s, 4.42, 5.72, 4.42, 0, C.sky, 3, { begin: 'triangle', name: 'Result line' });
  txt(s, 'результат', 7.06, 5.34, 1.5, 0.3, { size: 19, bold: true, color: C.sky, align: 'center' });
  txt(s, 'Видео не уходит во внешние AI API.', 0.82, 6.67, 11.7, 0.42, { size: 25, bold: true, color: C.white, align: 'center' });
  addNotes(s, speech[6], 'README.md; deploy/modal_app.py; contracts/openapi.json; eval/reports/pitch-onset-release.json.');
}

// 8. Typographic close on the brand gradient.
{
  const s = pptx.addSlide('LADO');
  s.addImage({ path: gradient, x: 0, y: 0, w: 13.333, h: 7.5, altText: 'Оранжево-красно-розовый градиент Lado', objectName: 'Closing gradient' });
  s.addImage({ path: logoStage, x: 0.72, y: 0.54, w: 1.92, h: 0.63, altText: 'Lado', objectName: 'Closing logo' });
  s.addImage({ path: markStage, x: 11.42, y: 0.62, w: 1.02, h: 1.02, altText: 'Знак Lado', objectName: 'Closing mark' });
  txt(s, 'Каждая находка\nс доказательством', 0.72, 1.62, 8.2, 1.8, { size: 55, bold: true, color: C.white, name: 'Closing title' });
  txt(s, 'Таймкод, правило и кадры\nв одном рабочем пространстве.', 0.76, 3.68, 7.4, 1.14, { size: 26, color: C.white, name: 'Closing thesis' });
  txt(s, 'Открыть Lado ↗', 0.76, 5.66, 2.45, 0.48, { size: 22, bold: true, color: C.white, link: { url: 'https://darinax2005x--lado-ladoservice-web.us-east.modal.run/app' }, name: 'Closing service link' });
  txt(s, 'Репозиторий ↗', 3.72, 5.66, 2.1, 0.48, { size: 22, bold: true, color: C.white, link: { url: 'https://github.com/DarinaX2005X/dreamteam-datathon' }, name: 'Closing repository link' });
  addNotes(s, speech[7], 'README.md; working service URL; public repository URL supplied by project.');
}

async function main() {
  const file = path.join(OUTPUT, 'lado-pitch.pptx');
  await pptx.writeFile({ fileName: file, compression: true });
  console.log(`Wrote ${file}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
