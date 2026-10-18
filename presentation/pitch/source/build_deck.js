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
pptx.title = 'Lado — Всё важное на виду';
pptx.lang = 'ru-RU';
pptx.theme = {
  headFontFace: 'Segoe UI',
  bodyFontFace: 'Segoe UI',
  lang: 'ru-RU',
};
pptx.defineSlideMaster({
  title: 'LADO_BLANK',
  background: { color: 'FCFAFB' },
  objects: [],
  margin: 0,
});

const C = {
  paper: 'FCFAFB',
  white: 'FFFFFF',
  ink: '2A0A18',
  stage: '1E0711',
  stage2: '2A0B19',
  stage3: '3A1225',
  muted: '6B5761',
  mutedLight: 'CDB6C1',
  border: 'E1D5DB',
  magenta: 'A31551',
  darkMagenta: '4A0221',
  coral: 'FC9385',
  orange: 'FF6A3D',
  red: 'E8604F',
  sky: '61CFFF',
  skyDark: '0B6F99',
  success: '17704A',
};

const FONT = 'Segoe UI';
const SW = 13.333;
const SH = 7.5;
const logoLight = path.join(REPO, 'identity', 'logo', 'lado-logo.svg');
const logoStage = path.join(REPO, 'identity', 'logo', 'lado-logo-on-stage.svg');
const markLight = path.join(REPO, 'identity', 'logo', 'lado-mark.svg');
const hero = path.join(ASSETS, 'imagegen', 'hero-film-lens.png');
const ribbon = path.join(ASSETS, 'svg', 'mechanism-ribbon.svg');
const lens = path.join(ASSETS, 'svg', 'observation-lens.svg');
const weave = path.join(ASSETS, 'svg', 'trust-weave.svg');
const warmWash = path.join(ASSETS, 'svg', 'warm-wash.svg');
const closingGradient = path.join(ASSETS, 'svg', 'closing-gradient.svg');

for (const asset of [logoLight, logoStage, markLight, hero, ribbon, lens, weave, warmWash, closingGradient]) {
  if (!fs.existsSync(asset)) throw new Error(`Missing asset: ${asset}`);
}

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT,
    color: options.color || C.ink,
    fontSize: options.fontSize || 20,
    bold: options.bold || false,
    breakLine: options.breakLine,
    align: options.align || 'left',
    valign: options.valign || 'top',
    margin: options.margin ?? 0,
    inset: options.inset ?? 0,
    paraSpaceAfterPt: options.paraSpaceAfterPt || 0,
    lineSpacingMultiple: options.lineSpacingMultiple,
    fit: 'shrink',
    isTextBox: true,
    objectName: options.objectName,
    hyperlink: options.hyperlink,
    transparency: options.transparency,
  });
}

function addRichText(slide, runs, x, y, w, h, options = {}) {
  slide.addText(runs, {
    x, y, w, h,
    fontFace: FONT,
    color: options.color || C.ink,
    fontSize: options.fontSize || 20,
    bold: options.bold || false,
    align: options.align || 'left',
    valign: options.valign || 'top',
    margin: options.margin ?? 0,
    inset: options.inset ?? 0,
    breakLine: false,
    fit: 'shrink',
    isTextBox: true,
    objectName: options.objectName,
  });
}

function title(slide, text, opts = {}) {
  addText(slide, text, opts.x ?? 0.72, opts.y ?? 0.48, opts.w ?? 8.7, opts.h ?? 0.9, {
    fontSize: opts.fontSize ?? 34,
    bold: true,
    color: opts.color || C.ink,
    objectName: opts.objectName || 'Slide title',
  });
}

function line(slide, x, y, w, h, color, width = 2, opts = {}) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: {
      color,
      width,
      transparency: opts.transparency || 0,
      dashType: opts.dashType || 'solid',
      beginArrowType: opts.beginArrowType,
      endArrowType: opts.endArrowType,
    },
    objectName: opts.objectName,
  });
}

function circle(slide, x, y, d, fill, lineColor = fill, lineWidth = 0, opts = {}) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: d, h: d,
    fill: { color: fill, transparency: opts.transparency || 0 },
    line: { color: lineColor, width: lineWidth, transparency: opts.lineTransparency || 0 },
    objectName: opts.objectName,
  });
}

function rect(slide, x, y, w, h, fill, opts = {}) {
  slide.addShape(opts.rounded ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: opts.radius,
    fill: { color: fill, transparency: opts.transparency || 0 },
    line: opts.lineColor ? { color: opts.lineColor, width: opts.lineWidth || 1, transparency: opts.lineTransparency || 0 } : { color: fill, transparency: 100 },
    radius: opts.radius,
    objectName: opts.objectName,
  });
}

function addBulletLine(slide, markerColor, heading, body, x, y, w, opts = {}) {
  circle(slide, x, y + 0.105, 0.11, markerColor, markerColor, 0, { objectName: `${heading} marker` });
  addRichText(slide, [
    { text: heading, options: { bold: true, color: opts.color || C.ink } },
    { text: body ? `\n${body}` : '', options: { breakLine: false, color: opts.bodyColor || C.muted, fontSize: opts.bodySize || 17 } },
  ], x + 0.24, y, w - 0.24, opts.h || 0.72, {
    fontSize: opts.headingSize || 19,
    color: opts.color || C.ink,
    objectName: `${heading} text`,
  });
}

function addSourceNotes(slide, speech, sources) {
  slide.addNotes(`${speech}\n\nИсточники слайда: ${sources}`);
}

const speech = [
  'Камера уже видит кассовую операцию. Но видеозапись сама по себе не отвечает на главный вопрос внутреннего контроля: какой именно момент нужно проверить и на каком основании. Lado превращает запись в проверяемый эпизод — с таймкодом, наблюдаемыми действиями, условиями правила и решением человека.',
  'Здесь важна граница вывода. Деньги в кадре — это наблюдаемый предмет. Передача или пересчёт — действие, которое нужно подтвердить во времени. И даже подтверждённое действие ещё не является нарушением. Для вывода нужны участники, порядок событий, видимость сцены и условие конкретного правила. Lado сохраняет эту цепочку, чтобы проверяющий видел, из чего сделан вывод.',
  'Поэтому продукт построен как одна доказательная цепочка. Сначала сохраняются исходные таймкоды и наблюдаемые события. Затем движок правил проверяет роли, объекты, последовательность и достаточность обзора. Если условие выполнено, Lado формирует эпизод с моментом начала и кадрами-доказательствами. После этого человек подтверждает находку, отклоняет её или отмечает, что событие неразличимо.',
  'Проверяющий работает не с бесконечным видеопотоком, а с конкретным эпизодом. Он видит момент начала эпизода, описание того, что наблюдалось, объяснение, почему может применяться правило, и до трёх кадров-доказательств. Рядом остаются источник правила и ограничения вывода. Решение и заметка сохраняются отдельно от автоматических наблюдений. Результат можно открыть в интерфейсе, получить через HTTP API или вывести в лист проверки.',
  'Чтобы получить такое объяснение, система смотрит на запись в нескольких масштабах. Специализированные детекторы находят руки, банкноты, удостоверения и счётчики; отдельный слой ведёт людей и предметы во времени. Qwen3-VL-8B анализирует последовательность кадров и отношения между действиями. Затем программная проверка сверяет интервалы, идентификаторы объектов и временные связи. Все модели работают в частном GPU-контуре проекта в Modal, без отправки видео во внешние AI API.',
  'Вот конкретный пример работы всего контура. В записи длительностью около двадцати одной секунды сервис определил начало эпизода на шестнадцатой секунде: сотрудник дошёл до выхода, та же инкассаторская сумка осталась на столе, а рядом не было видимого сотрудника. Сервис вернул один эпизод и кадр-доказательство. Проверяющий получает именно момент совпадения условий сценария, а не весь длинный интервал доказательств.',
  'Для внедрения важны две независимые границы контроля. Первая — данные и модели: запись, веса и результаты остаются в частном контуре, а внешний API защищён ключом. Вторая — правила: они хранятся отдельно от модели, содержат источники и ограничения применимости. Автоматические наблюдения и решение проверяющего сохраняются как разные слои, поэтому результат можно пересмотреть и объяснить.',
  'Lado уже доступен как работающий веб-сервис и HTTP API. Пилот можно начать с ограниченного потока записей и конкретных критериев внутреннего контроля: подключить загрузку, согласовать правила, проверить эпизоды и сохранить решения. Так путь от видеозаписи до проверяемого решения становится короче, а ответственность остаётся у специалиста.',
];

// Slide 1 — opening
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.paper };
  slide.addImage({ path: hero, x: 0, y: 0, w: SW, h: SH, altText: 'Синтетическая концептуальная иллюстрация: киноплёнка и оптическое кольцо в тёплой палитре', objectName: 'Hero concept art' });
  slide.addImage({ path: logoLight, x: 0.72, y: 0.58, w: 1.86, h: 0.605, altText: 'Lado', objectName: 'Lado logo' });
  addText(slide, 'Всё важное —\nна виду.', 0.72, 1.74, 5.7, 1.75, {
    fontSize: 48,
    bold: true,
    color: C.ink,
    objectName: 'Hero title',
  });
  addText(slide, 'Проверка кассовых операций по видеозаписи', 0.76, 3.64, 5.25, 0.46, {
    fontSize: 20,
    bold: true,
    color: C.magenta,
    objectName: 'Product description',
  });
  addText(slide, 'Lado превращает запись в проверяемый эпизод: таймкод, наблюдения, основание и решение человека.', 0.76, 4.24, 5.25, 1.22, {
    fontSize: 21,
    color: C.ink,
    objectName: 'Hero explanation',
  });
  line(slide, 0.76, 5.83, 1.06, 0, C.magenta, 3, { objectName: 'Hero link underline' });
  addText(slide, 'Работающий сервис ↗', 0.76, 5.58, 2.35, 0.46, {
    fontSize: 17,
    bold: true,
    color: C.magenta,
    hyperlink: { url: 'https://darinax2005x--lado-ladoservice-web.us-east.modal.run/app' },
    objectName: 'Service hyperlink',
  });
  addSourceNotes(slide, speech[0], 'contracts/product.json; README.md; identity/logo/lado-logo.svg. Hero art: assets/imagegen/hero-film-lens.png, synthetic concept illustration.');
}

// Slide 2 — why context matters
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.paper };
  title(slide, 'Видео фиксирует событие.\nРешение требует контекста.', { w: 5.35, h: 1.55, fontSize: 36 });
  addText(slide, 'Lado разделяет три уровня вывода, чтобы автоматическая находка оставалась проверяемой.', 0.75, 2.2, 4.95, 0.95, {
    fontSize: 21,
    color: C.muted,
    objectName: 'Context explanation',
  });
  addBulletLine(slide, C.orange, 'Предмет', 'Банкнота, рука, удостоверение или счётчик видны в кадре.', 0.76, 3.35, 4.85, { h: 0.75 });
  addBulletLine(slide, C.coral, 'Действие', 'Передача, пересчёт или оставленный предмет подтверждаются во времени.', 0.76, 4.38, 4.85, { h: 0.86 });
  addBulletLine(slide, C.skyDark, 'Основание', 'Условие правила проверяет роли, порядок событий и достаточность обзора.', 0.76, 5.52, 4.85, { h: 0.86 });

  // right-side editorial mechanism: scattered observations converge into a review frame
  rect(slide, 6.35, 0.0, 6.98, 7.5, C.stage2, { objectName: 'Context stage' });
  const dots = [
    [6.86, 1.25, 0.17, C.orange], [7.65, 1.82, 0.11, C.coral], [8.38, 1.14, 0.14, C.sky],
    [7.24, 3.04, 0.12, C.coral], [8.55, 2.78, 0.17, C.orange], [9.08, 1.98, 0.10, C.sky],
    [7.72, 4.2, 0.14, C.sky], [8.78, 4.62, 0.12, C.coral], [9.33, 3.78, 0.16, C.orange],
  ];
  dots.forEach((d, i) => circle(slide, d[0], d[1], d[2], d[3], d[3], 0, { objectName: `Observation ${i + 1}` }));
  line(slide, 7.03, 1.42, 3.64, 2.15, C.coral, 2, { transparency: 35, objectName: 'Observation path 1' });
  line(slide, 7.38, 3.18, 3.29, 0.39, C.coral, 2, { transparency: 35, objectName: 'Observation path 2' });
  line(slide, 7.84, 4.34, 2.83, -0.77, C.sky, 2, { transparency: 35, objectName: 'Observation path 3' });
  rect(slide, 10.12, 2.5, 2.45, 2.45, C.stage3, { rounded: true, lineColor: C.coral, lineWidth: 3, objectName: 'Decision frame' });
  circle(slide, 10.72, 3.04, 1.25, C.darkMagenta, C.coral, 3, { objectName: 'Decision lens' });
  line(slide, 11.58, 4.18, 0.42, 0.42, C.coral, 7, { objectName: 'Lens handle' });
  addText(slide, 'наблюдение\n≠\nнарушение', 10.25, 5.33, 2.2, 1.05, {
    fontSize: 22,
    bold: true,
    color: C.white,
    align: 'center',
    valign: 'mid',
    objectName: 'Key distinction',
  });
  addSourceNotes(slide, speech[1], 'README.md, sections “Возможности” and “Как устроен анализ”.');
}

// Slide 3 — mechanism
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.stage };
  title(slide, 'От записи до решения —\nодна доказательная цепочка', { color: C.white, w: 7.5, h: 1.15, fontSize: 34 });
  addText(slide, 'Каждый этап сохраняет связь с исходным временем и наблюдаемыми фактами.', 8.18, 0.63, 4.42, 0.62, {
    fontSize: 18,
    color: C.mutedLight,
    align: 'right',
    objectName: 'Mechanism support',
  });
  slide.addImage({ path: ribbon, x: 0.6, y: 1.47, w: 12.15, h: 2.87, altText: 'Пять связанных этапов доказательной цепочки', objectName: 'Mechanism ribbon' });
  const stages = [
    { x: 0.45, w: 2.05, title: 'Видеозапись', body: 'исходные таймкоды' },
    { x: 2.8, w: 2.35, title: 'Наблюдения', body: 'участники · предметы · действия' },
    { x: 5.37, w: 2.22, title: 'Правила', body: 'условия · порядок · видимость' },
    { x: 7.98, w: 2.4, title: 'Эпизод', body: 'момент начала · доказательства' },
    { x: 10.7, w: 2.18, title: 'Решение', body: 'подтвердить · отклонить · не видно', accent: C.sky },
  ];
  stages.forEach((s, i) => {
    addText(slide, s.title, s.x, 4.42, s.w, 0.44, { fontSize: 22, bold: true, color: s.accent || C.white, align: 'center', objectName: `Stage ${i + 1} title` });
    addText(slide, s.body, s.x, 4.94, s.w, 0.82, { fontSize: 15, color: C.mutedLight, align: 'center', objectName: `Stage ${i + 1} detail` });
  });
  line(slide, 0.76, 6.28, 11.86, 0, C.stage3, 2, { objectName: 'Mechanism closing rule' });
  addText(slide, 'Автоматическая находка указывает, что проверить. Финальное решение остаётся за человеком.', 1.42, 6.48, 10.5, 0.56, {
    fontSize: 20,
    bold: true,
    color: C.coral,
    align: 'center',
    objectName: 'Human decision statement',
  });
  addSourceNotes(slide, speech[2], 'README.md pipeline description; backend/navidu/pipeline/run.py; backend/navidu/contracts/.');
}

// Slide 4 — reviewer output
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.paper };
  title(slide, 'Проверяющий видит не поток,\nа конкретный эпизод', { w: 7.15, h: 1.2, fontSize: 35 });
  addText(slide, '00:16', 0.72, 1.78, 3.35, 1.1, { fontSize: 66, bold: true, color: C.magenta, objectName: 'Illustrative timecode' });
  addText(slide, 'момент начала события', 0.82, 2.9, 3.05, 0.46, { fontSize: 18, bold: true, color: C.muted, objectName: 'Timecode label' });

  // Native aperture/timeline, deliberately schematic rather than UI.
  circle(slide, 1.13, 3.63, 2.3, C.paper, C.coral, 4, { objectName: 'Episode aperture outer' });
  circle(slide, 1.47, 3.97, 1.62, C.paper, C.magenta, 4, { objectName: 'Episode aperture middle' });
  circle(slide, 1.91, 4.41, 0.74, C.darkMagenta, C.darkMagenta, 0, { objectName: 'Episode aperture core' });
  line(slide, 2.28, 4.78, 3.22, 0, C.magenta, 3, { objectName: 'Episode timeline' });
  line(slide, 3.45, 4.33, 0, 0.9, C.coral, 5, { objectName: 'Episode onset marker' });
  circle(slide, 3.32, 4.64, 0.26, C.coral, C.coral, 0, { objectName: 'Episode onset dot' });
  addText(slide, 'начало', 3.08, 5.34, 0.86, 0.34, { fontSize: 15, bold: true, color: C.magenta, align: 'center', objectName: 'Episode onset label' });
  addText(slide, 'окно события', 4.2, 4.22, 1.45, 0.36, { fontSize: 15, color: C.muted, align: 'center', objectName: 'Episode window label' });

  line(slide, 6.22, 1.58, 0, 4.92, C.border, 2, { objectName: 'Episode divider' });
  addBulletLine(slide, C.orange, 'Время', 'Таймкод и границы эпизода.', 6.68, 1.66, 5.72, { h: 0.67 });
  addBulletLine(slide, C.coral, 'Наблюдаемое', 'Что именно произошло и какие объекты связаны с событием.', 6.68, 2.56, 5.72, { h: 0.82 });
  addBulletLine(slide, C.magenta, 'Основание', 'Почему может применяться правило, его источник и ограничения.', 6.68, 3.62, 5.72, { h: 0.82 });
  addBulletLine(slide, C.skyDark, 'Доказательства', 'От одного до трёх кадров с исходными временными метками.', 6.68, 4.68, 5.72, { h: 0.82 });
  addBulletLine(slide, C.success, 'Решение человека', 'Подтвердить, отклонить или отметить: «не видно».', 6.68, 5.74, 5.72, { h: 0.82 });
  addText(slide, 'Интерфейс · лист проверки · HTTP API', 0.76, 6.74, 4.95, 0.4, { fontSize: 17, bold: true, color: C.magenta, objectName: 'Output surfaces' });
  addSourceNotes(slide, speech[3], 'README.md; contracts/openapi.json, Finding, EvidenceFrame and Decision schemas. The timeline is an editable schematic, not a UI screenshot.');
}

// Slide 5 — technical mechanism
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.stage };
  title(slide, 'Система смотрит\nв нескольких масштабах', { color: C.white, w: 5.1, h: 1.25, fontSize: 36 });
  addText(slide, 'Один слой уточняет другой, прежде чем наблюдение попадёт в правило.', 0.76, 1.86, 4.45, 0.8, {
    fontSize: 20,
    color: C.mutedLight,
    objectName: 'Technical explanation',
  });
  slide.addImage({ path: lens, x: 5.07, y: 0.55, w: 7.35, h: 5.99, altText: 'Три уровня анализа сходятся в одной проверяемой точке', objectName: 'Observation lens' });
  addBulletLine(slide, C.orange, 'Специализированные детекторы', 'Руки, банкноты, удостоверения и счётчики.', 0.76, 3.03, 4.5, { color: C.white, bodyColor: C.mutedLight, h: 0.92, headingSize: 19 });
  addBulletLine(slide, C.coral, 'Qwen3-VL-8B', 'Последовательность кадров, действия и связи между ними.', 0.76, 4.18, 4.5, { color: C.white, bodyColor: C.mutedLight, h: 0.9, headingSize: 19 });
  addBulletLine(slide, C.sky, 'Трекинг и временная проверка', 'Кто, что и когда; интервалы и ссылки на объекты.', 0.76, 5.31, 4.5, { color: C.white, bodyColor: C.mutedLight, h: 0.9, headingSize: 19 });
  rect(slide, 0.72, 6.55, 11.9, 0.53, C.stage3, { rounded: true, lineColor: C.stage3, objectName: 'Private inference statement background' });
  addText(slide, 'Частный GPU-контур Modal · NVIDIA L4 24 GB · без внешних AI API для анализа видео', 1.05, 6.67, 11.25, 0.3, {
    fontSize: 16,
    bold: true,
    color: C.coral,
    align: 'center',
    objectName: 'Private inference statement',
  });
  addSourceNotes(slide, speech[4], 'README.md model and privacy sections; backend/navidu/perception/object_detector.py; backend/navidu/perception/qwen.py; eval/reports/modal-question-verification.json.');
}

// Slide 6 — narrow demonstrated example
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.paper };
  slide.addImage({ path: warmWash, x: 0, y: 0, w: SW, h: SH, altText: 'Светлый фон с оранжево-красно-розовым градиентом справа', objectName: 'Warm case background' });
  title(slide, 'Найден момент, с которого\nначинается проверка', { w: 7.0, h: 1.2, fontSize: 35 });
  addText(slide, '16,0 с', 0.72, 1.94, 3.35, 1.12, { fontSize: 66, bold: true, color: C.magenta, objectName: 'Demonstrated onset' });
  addText(slide, 'таймкод начала эпизода', 0.8, 3.05, 3.0, 0.42, { fontSize: 17, bold: true, color: C.muted, objectName: 'Onset label' });
  rect(slide, 4.08, 1.7, 8.45, 1.62, C.white, { rounded: true, lineColor: C.border, lineWidth: 1, objectName: 'Example narrative field' });
  addText(slide, 'Сотрудник дошёл до выхода; та же инкассаторская сумка осталась на рабочем столе без видимого сотрудника рядом.', 4.42, 1.98, 7.78, 1.04, {
    fontSize: 21,
    bold: true,
    color: C.ink,
    objectName: 'Observed bag scenario',
  });
  addText(slide, 'Работающий HTTP-пример: запись 20,9 с · один эпизод · один кадр-доказательство · схема ответа и JPEG валидны.', 4.16, 3.6, 8.25, 0.82, {
    fontSize: 18,
    color: C.muted,
    objectName: 'HTTP example result',
  });

  const tx = 0.84, ty = 5.3, tw = 11.65;
  line(slide, tx, ty, tw, 0, C.border, 8, { objectName: 'Bag example timeline base' });
  const onsetX = tx + tw * (16 / 20.866667);
  const endX = tx + tw * (19 / 20.866667);
  line(slide, onsetX, ty, endX - onsetX, 0, C.coral, 12, { objectName: 'Detected event window' });
  line(slide, onsetX, ty - 0.62, 0, 1.18, C.magenta, 4, { objectName: 'Detected onset line' });
  circle(slide, onsetX - 0.11, ty - 0.11, 0.22, C.magenta, C.magenta, 0, { objectName: 'Detected onset dot' });
  addText(slide, '16,0', onsetX - 0.44, ty - 1.05, 0.88, 0.34, { fontSize: 17, bold: true, color: C.magenta, align: 'center', objectName: 'Onset tick label' });
  addText(slide, '19,0', endX - 0.42, ty + 0.22, 0.84, 0.3, { fontSize: 15, color: C.muted, align: 'center', objectName: 'Window end label' });
  addText(slide, '00:00', tx - 0.08, ty + 0.22, 0.7, 0.3, { fontSize: 15, color: C.muted, objectName: 'Timeline start label' });
  addText(slide, '20,9 с', tx + tw - 0.78, ty + 0.22, 0.78, 0.3, { fontSize: 15, color: C.muted, align: 'right', objectName: 'Timeline duration label' });
  addText(slide, 'до ухода сумка видна', 5.25, 4.72, 2.2, 0.35, { fontSize: 16, color: C.muted, align: 'center', objectName: 'Prior bag visibility label' });
  addText(slide, 'событие подтверждено\nсовпадающими кадрами', onsetX - 1.05, ty + 0.6, 2.35, 0.62, { fontSize: 16, bold: true, color: C.magenta, align: 'center', objectName: 'Confirmed window label' });
  rect(slide, 0.76, 6.67, 11.82, 0.48, C.magenta, { transparency: 94, objectName: 'Example result background' });
  addText(slide, 'Результат начинается в 16,0 с — с момента совпадения условий сценария.', 1.02, 6.77, 11.32, 0.26, {
    fontSize: 16,
    bold: true,
    color: C.magenta,
    align: 'center',
    objectName: 'Example result statement',
  });
  addSourceNotes(slide, speech[5], 'eval/reports/modal-question-verification.json, targeted_pipeline_check and deployment.bag_positive; rules/catalog/V-BAG-UNATTENDED.yaml.');
}

// Slide 7 — controlled deployment and governance
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.paper };
  title(slide, 'Контроль остаётся внутри контура\nи внутри регламента', { w: 9.1, h: 1.2, fontSize: 35 });
  slide.addImage({ path: weave, x: 3.25, y: 1.63, w: 6.75, h: 4.12, altText: 'Переплетение контура данных и контура правил вокруг решения', objectName: 'Trust weave' });

  addText(slide, 'Данные и модели', 0.76, 2.0, 2.8, 0.42, { fontSize: 21, bold: true, color: C.skyDark, objectName: 'Data boundary heading' });
  addText(slide, 'Видео, веса и результаты остаются в частном контуре проекта в Modal. Анализ не требует внешних AI API.', 0.76, 2.62, 2.74, 1.7, { fontSize: 18, color: C.ink, objectName: 'Data boundary copy' });
  addText(slide, 'HTTP API защищён ключом', 0.76, 4.62, 2.7, 0.45, { fontSize: 17, bold: true, color: C.skyDark, objectName: 'API key fact' });

  addText(slide, 'Правила и решения', 10.02, 2.0, 2.55, 0.42, { fontSize: 21, bold: true, color: C.magenta, objectName: 'Rules boundary heading' });
  addText(slide, 'Условия, источники и ограничения хранятся отдельно от модели. Решение проверяющего сохраняется отдельной ревизией.', 10.02, 2.62, 2.55, 1.75, { fontSize: 18, color: C.ink, objectName: 'Rules boundary copy' });
  addText(slide, 'Вывод можно пересмотреть и объяснить', 10.02, 4.62, 2.55, 0.74, { fontSize: 17, bold: true, color: C.magenta, objectName: 'Reviewability fact' });

  line(slide, 0.76, 6.1, 11.8, 0, C.border, 2, { objectName: 'Trust closing rule' });
  addText(slide, 'Автоматические наблюдения не подменяют решение специалиста.', 1.06, 6.34, 11.2, 0.55, {
    fontSize: 22,
    bold: true,
    color: C.ink,
    align: 'center',
    objectName: 'Governance statement',
  });
  addSourceNotes(slide, speech[6], 'README.md privacy, rules and architecture sections; contracts/openapi.json security and Decision schemas; eval/reports/modal-question-verification.json deployment.');
}

// Slide 8 — close and pilot
{
  const slide = pptx.addSlide('LADO_BLANK');
  slide.background = { color: C.stage };
  slide.addImage({ path: hero, x: 0, y: 0, w: SW, h: SH, altText: 'Синтетическая концептуальная иллюстрация: киноплёнка и оптическое кольцо', objectName: 'Closing concept art' });
  slide.addImage({ path: closingGradient, x: 0, y: 0, w: 6.7, h: 7.5, altText: 'Оранжево-красно-розовое градиентное поле', objectName: 'Closing gradient field' });
  slide.addImage({ path: logoStage, x: 0.72, y: 0.55, w: 1.98, h: 0.64, altText: 'Lado', objectName: 'Lado stage logo' });
  addText(slide, 'Проверим Lado\nна ваших сценариях', 0.72, 1.53, 5.75, 1.45, {
    fontSize: 39,
    bold: true,
    color: C.white,
    objectName: 'Closing title',
  });
  addText(slide, 'Подключить запись или сервис через HTTP API', 1.13, 3.36, 4.96, 0.5, { fontSize: 19, bold: true, color: C.white, objectName: 'Pilot step 1' });
  line(slide, 0.77, 3.58, 0.2, 0, C.orange, 5, { objectName: 'Pilot marker 1' });
  addText(slide, 'Согласовать критерии и источники под процедуру контроля', 1.13, 4.09, 4.96, 0.66, { fontSize: 19, bold: true, color: C.white, objectName: 'Pilot step 2' });
  line(slide, 0.77, 4.34, 0.2, 0, C.coral, 5, { objectName: 'Pilot marker 2' });
  addText(slide, 'Проверять эпизоды и сохранять решения', 1.13, 4.92, 4.96, 0.52, { fontSize: 19, bold: true, color: C.white, objectName: 'Pilot step 3' });
  line(slide, 0.77, 5.14, 0.2, 0, C.sky, 5, { objectName: 'Pilot marker 3' });

  addText(slide, 'Открыть Lado ↗', 0.76, 6.14, 2.25, 0.48, {
    fontSize: 18,
    bold: true,
    color: C.coral,
    hyperlink: { url: 'https://darinax2005x--lado-ladoservice-web.us-east.modal.run/app' },
    objectName: 'Closing app link',
  });
  addText(slide, 'Репозиторий ↗', 3.24, 6.14, 2.05, 0.48, {
    fontSize: 18,
    bold: true,
    color: C.sky,
    hyperlink: { url: 'https://github.com/DarinaX2005X/dreamteam-datathon' },
    objectName: 'Closing repository link',
  });
  addText(slide, 'Репозиторий закрытый; доступ по приглашению.', 0.76, 6.77, 4.85, 0.3, { fontSize: 13, color: C.mutedLight, objectName: 'Repository access note' });
  addSourceNotes(slide, speech[7], 'README.md live app and API sections; eval/reports/modal-question-verification.json deployment; requested repository URL.');
}

pptx.writeFile({ fileName: path.join(OUTPUT, 'lado-pitch.pptx'), compression: true });
