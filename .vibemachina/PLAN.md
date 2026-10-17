# План продукта

Статус: реализация переданного плана прямо поручена пользователем 2026-10-17. Сохранены 7 этапов и содержательный объём, кроме исключённых презентации/речи/рекламного видео. Название Lado; внутренние namespace и пути navidu сохранены.

## Результат и границы

Lado состоит из пяти частей: локальный движок анализа записи кассы (факты → правила из норм РК → эпизоды с таймкодом, кадрами и обоснованием), API жюри по схеме кейса, рабочее место проверяющего, короткий лендинг и README и воспроизводимая подготовка запуска. Полный состав и смысл описаны в [PROJECT.md](PROJECT.md), оформление в [DESIGN.md](DESIGN.md), требования в [REQUIREMENTS.md](REQUIREMENTS.md).

Первый сквозной путь «реальная запись → JSON API с реальными эпизодами на публичном адресе» является внутренней вехой (G2), а не урезанным объёмом. После него доводятся все задачи этапов 04–07.

Контрольные точки во времени (дата дедлайна неизвестна, допущение A3; доли считаются от времени между стартом реализации и дедлайном 12:00):

| Веха | Не позже | Что должно быть |
| --- | --- | --- |
| G1 | 15% | Этап 01 принят: схема разметки, каталог норм со сверенными пунктами, контракты в коде, выбор моделей и замер времени на целевом железе |
| G2 | 45% | Реальная запись проходит POST → GET на публичном адресе и возвращает эпизоды из настоящего конвейера; интерфейс показывает ход анализа и результат |
| G3 | 65% | Заморозка новых возможностей: всё из этапов 02–04 принято, дальше только интеграция, устойчивость и исправления |
| G4 | 85% | Ревью и исправления закрыты (этап 06), публичный сервис стабилен |
| G5 | 100% − запас | README и материалы сдачи готовы, ключ и адреса переданы, сервис не меняется в последние часы, кроме исправлений блокирующих сбоев |

## Общие контракты

Авторитетная реализация всех контрактов: `backend/navidu/contracts/` (pydantic v2), из неё генерируются `contracts/openapi.json`, `contracts/schemas/*.json` и `contracts/ts/types.gen.ts` (задача 01.3). Ниже определения, которые нельзя трактовать по-разному. Внутренние функции модулей расписывать заранее не нужно.

### C-UNITS. Время, координаты, идентификаторы

- Время: `float` секунд от начала видео, 3 знака после запятой. Источник: `pts * time_base − start_time` декодированного кадра. Переменная частота кадров поддерживается, номера кадров в контрактах не используются.
- Таймкод эпизода `timecode_sec` равен `timestamp_sec` первого (основного) кадра-доказательства.
- Координаты: `[x, y]` нормированы в `[0,1]` относительно кадра после применения поворота из метаданных; рамка `bbox = [x0, y0, x1, y1]`.
- `check_id` (он же `job_id` в API v1): 12 символов `[0-9a-f]`, случайный (`secrets.token_hex(6)`).
- Прочие ID: трек людей `p<число>`, трек объектов `o<число>`, зона `z-<тип>-<число>`, факт `ob-<число>`, кадр `fr-<sec*1000 целое>`, эпизод `fd-<sha1(rule_id|round(window.start,1)|round(window.end,1))[:10]>`. Эпизод сохраняет ID между ревизиями, если правило и окно не изменились.
- Язык строк для человека: русский.

### C-ERR. Ошибки

Все ошибки API имеют один вид:

```json
{ "error": { "code": "FILE_TOO_LARGE", "message": "Файл больше 300 МБ.", "request_id": "r-5c1d0e7a" } }
```

| HTTP | code | Когда |
| --- | --- | --- |
| 400 | `MISSING_FILE` | нет поля `file` или пустой файл |
| 401 | `UNAUTHORIZED` | нет или неверный `X-API-Key` (только `/api/v1`) |
| 404 | `NOT_FOUND` | неизвестный `job_id`/`check_id`/эпизод |
| 409 | `STALE_REVISION` | `base_revision` устарел (решения, пересчёт) |
| 409 | `READ_ONLY_EXAMPLE` | попытка сохранить решение в примере |
| 409 | `NOT_FAILED` | повтор для задания, которое не в статусе `failed` |
| 410 | `VIDEO_EXPIRED` | исходная запись удалена по сроку хранения |
| 413 | `FILE_TOO_LARGE` | больше `NAVIDU_MAX_UPLOAD_MB` (по умолчанию 300) |
| 415 | `UNSUPPORTED_MEDIA` | файл не декодируется как видео (проверка ffprobe, не по расширению) |
| 422 | `INVALID_INPUT` | ошибки полей; добавляется `field_issues: [{path, message}]` |
| 422 | `VIDEO_TOO_LONG` | длительность больше `NAVIDU_MAX_DURATION_SEC` (600) |
| 429 | `QUEUE_FULL` | очередь переполнена; заголовок `Retry-After` |
| 503 | `NOT_READY` | модели ещё не загружены |
| 500 | `INTERNAL` | прочее; детали только в логе по `request_id` |

### C-API-V1. API жюри (производитель 03.1, потребитель жюри и `scripts/smoke_api.py`)

Ключ: заголовок `X-API-Key`. Ключи задаются в `NAVIDU_API_KEYS` (через запятую); сравнение за постоянное время. Без верного ключа задание не создаётся.

`POST /api/v1/analyze`, `multipart/form-data`, поле `file` (.mov; mp4/m4v тоже принимаются, если декодируются).

```http
POST /api/v1/analyze
X-API-Key: <ключ>
Content-Type: multipart/form-data; boundary=…
```

Ответ `200 OK` (выбран 200, а не 202, чтобы любой простой клиент считал его успехом):

```json
{ "job_id": "7c1e2a9b40d3", "status": "processing" }
```

Повторная загрузка того же файла (тот же `sha256` и та же `pipeline_version`) создаёт новый `job_id`, но может переиспользовать готовые артефакты анализа. Тогда задание переходит в `done` почти сразу.

`GET /api/v1/jobs/{job_id}`:

- в процессе (внутренние `queued` и `processing`): `200 {"job_id":"7c1e2a9b40d3","status":"processing"}`;
- готово: `200` и результат строго по схеме кейса;
- сбой: `200 {"job_id":"…","status":"failed","error":{"code":"VIDEO_DECODE_FAILED","message":"Не удалось прочитать кадры после 01:12."}}`.

```json
{
  "job_id": "7c1e2a9b40d3",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Пересчёт денег без клиента у кассы",
      "timecode_sec": 44.2,
      "description": "Кассир держит пачку денег 6 с, клиента у окна нет. Наличные принимаются и пересчитываются в присутствии клиента: п. 20 Правил осуществления кассовых операций (пост. Правления НБРК от 29.11.2019 № 231).",
      "evidence_frames": [
        { "timestamp_sec": 44.2, "image_base64": "/9j/4AAQ…" },
        { "timestamp_sec": 42.0, "image_base64": "/9j/4AAQ…" }
      ]
    }
  ]
}
```

Правила поля: `verdict` принимает значения `violation` | `no_violation`. `violations` упорядочены по `timecode_sec`, при `no_violation` список пуст. `type` берётся из `type_label` каталога правил (стабильные строки). `evidence_frames`: от 1 до 3 кадров, первый основной, JPEG качества 85, длинная сторона не больше 1280 px, base64 без префикса `data:`, с тонкой разметкой при `NAVIDU_EVIDENCE_OVERLAY=1` (по умолчанию). Других ключей в успешном ответе нет. API отдаёт только автоматическую ревизию 1, решения проверяющего на него не влияют.

### C-UI-API. API интерфейса (производитель 03.1, потребители 04.2–04.4, 05.1)

Без ключа, тот же источник, что и фронтенд. Защита от перегрузки: не больше `NAVIDU_UI_MAX_ACTIVE_PER_IP` (2) активных заданий с одного адреса, задания API v1 всегда идут раньше заданий интерфейса.

| Метод и путь | Запрос | Ответ | Ошибки |
| --- | --- | --- | --- |
| `POST /api/ui/checks` | multipart `file` | `201 {"check": CheckSummary}` | 400, 413, 415, 422, 429 |
| `GET /api/ui/checks/{id}` | — | `200 {"check": CheckDetail}` | 404 |
| `GET /api/ui/checks/{id}/video` | заголовок `Range` | `206/200 video/*` (исходный файл, пока не удалён) | 404, `410 VIDEO_EXPIRED` |
| `GET /api/ui/checks/{id}/frames/{frame_id}.jpg?overlay=0\|1` | — | `image/jpeg` | 404 |
| `PUT /api/ui/checks/{id}/findings/{finding_id}/decision` | `{"decision":"confirmed"\|"rejected"\|"not_visible"\|null,"note":string≤2000,"base_revision":int}` | `200 {"decision": Decision, "revision": int}` | 404, 409 `STALE_REVISION`, 409 `READ_ONLY_EXAMPLE`, 422 |
| `POST /api/ui/checks/{id}/reevaluate` | `{"base_revision":int,"role_overrides":[{"track_id":"p3","role":"collector"}],"zone_overrides":[Zone],"dry_run":bool}` | `200 {"finding_set": FindingSet, "diff": {"added":[id],"removed":[id],"changed":[id],"verdict_changed":bool}}` | 404, 409, 422 |
| `POST /api/ui/checks/{id}/retry` | — | `200 {"check": CheckSummary}` (статус снова `queued`, та же запись) | 404, `409 NOT_FAILED`, `410 VIDEO_EXPIRED` |
| `GET /api/ui/rules` | — | `200 {"rules":[RuleView], "not_checkable":[SourceClauseView]}` | — |
| `GET /api/ui/example` | — | `200 {"check_id": "…"}` или `404` (примера нет) | — |
| `GET /api/ui/health` | — | `200 {"ready":bool,"queue":{"api":int,"ui":int},"pipeline_version":"…"}` | — |

`CheckSummary`: `{id, file_name, status: queued|processing|done|failed, created_at (ISO UTC), duration_sec|null, verdict|null, findings_count|null, is_example}`.

`CheckDetail` расширяет `CheckSummary`:

```json
{
  "id": "7c1e2a9b40d3", "file_name": "kassa_0412.mov", "status": "processing",
  "created_at": "2026-10-20T06:12:03Z", "is_example": false,
  "video": { "duration_sec": 125.04, "fps_nominal": 29.97, "width": 1920, "height": 1080, "rotation": 0 },
  "progress": { "phase": "perception", "phases": ["decode","perception","facts","rules","evidence"],
                "processed_until_sec": 48.0, "queue_position": null },
  "layout": { "revision": 1, "zones": [ { "zone_id": "z-client_window-1", "kind": "client_window",
               "polygon": [[0.31,0.10],[0.69,0.10],[0.69,0.39],[0.31,0.39]], "source": "auto" } ] },
  "tracks": [ { "track_id": "p1", "kind": "person", "role": { "value": "cashier", "source": "auto" },
                "first_sec": 0.0, "last_sec": 125.0 } ],
  "observations": [ { "obs_id": "ob-17", "kind": "cash_in_hands", "start_sec": 41.5, "end_sec": 47.5,
                      "subjects": ["p1","o4"], "zones": ["z-cash_desk-1"] } ],
  "coverage": [ { "start_sec": 72.0, "end_sec": 80.0, "scope": "desk", "reason": "occluded" } ],
  "finding_set": null,
  "error": null
}
```

После `done` поле `finding_set` содержит текущую ревизию (`FindingSet`, см. C-DATA), `progress.phase` равен `done`. При `failed` поле `error` содержит `{code, message, failed_phase, at_sec|null}`, `observations` содержат всё, что успело сохраниться. Клиент опрашивает `GET /api/ui/checks/{id}` раз в 1.5 с, пока статус `queued|processing`. Опрос останавливается на скрытой вкладке и возобновляется при возврате. Потоковый транспорт не нужен: так проще за туннелем.

Решения: `PUT …/decision` с `decision: null` снимает решение. Решения привязаны к `finding_id` и сохраняются, пока эпизод существует в текущей ревизии. Если эпизод исчез после пересчёта, решение остаётся в истории ревизии, но не показывается. Пересчёт с `dry_run: true` (обязателен для примера) ничего не сохраняет.

### C-DATA. Предметные объекты (pydantic в `backend/navidu/contracts/domain.py`)

```text
Role = "cashier" | "client" | "collector" | "staff" | "outsider" | "unknown"
ZoneKind = "cashier_post" | "client_window" | "cash_desk" | "storage" | "service_door" | "transfer_tray" | "count_room"
ObservationKind = "person_in_zone" | "cash_in_hands" | "cash_on_desk" | "cash_handover" | "counting_motion"
                | "phone_visible" | "bag_present" | "bag_opened" | "storage_open" | "visibility_low"
Track        {track_id, kind: person|object, object_class|null, first_sec, last_sec,
              role: {value: Role, source: auto|reviewer}|null, samples: [{t, bbox, score}] (храним прореженно)}
Zone         {zone_id, kind: ZoneKind, polygon: [[x,y]...], source: auto|reviewer}
SceneLayout  {revision:int, zones: [Zone], camera_note: str|null}
Observation  {obs_id, kind, start_sec, end_sec, subjects:[track_id], zones:[zone_id], score: 0..1, frame_ids:[frame_id]}
Coverage     {start_sec, end_sec, scope: cashier|client|desk|storage|whole, reason: out_of_frame|occluded|low_light|motion_blur}
EvidenceFrame{frame_id, timestamp_sec, overlay_boxes: [{bbox, label, tone: observation|finding}]}
Finding      {finding_id, rule_id, type_label, timecode_sec, window: {start_sec, end_sec},
              basis: [obs_id], what_seen: str, why_may_violate: str, norm_refs: [clause_id],
              description: str, evidence: [EvidenceFrame] (1..3), strength: strong|moderate,
              decision: {value: confirmed|rejected|not_visible, note: str, decided_at}|null}
RuleStatus   {rule_id, status: violation|no_violation|not_checkable|partly_checkable, not_checkable_windows: [{start_sec,end_sec,reason}]}
FindingSet   {revision:int, basis: auto|reviewer, created_at, layout_revision:int, role_overrides:[{track_id, role}],
              verdict: violation|no_violation, findings:[Finding], rules:[RuleStatus]}
AnalysisRun  {check_id, pipeline_version, analysis_fps, model_versions:{name: sha256}, phase_timings_sec:{phase: float}}
```

Вердикт ревизии: `violation`, если есть хотя бы один эпизод с `decision.value != "rejected"`. Для API v1 берётся ревизия 1, а в ней решений нет. `description` = `what_seen` + « » + `why_may_violate` + « : » + краткая ссылка на норму. Описание строится шаблоном из фактов, свободная генерация текста запрещена.

### C-PIPE. Конвейер (производитель 02.4 собирает 02.1–02.3; потребитель 03.1)

```python
# backend/navidu/pipeline/run.py
def analyze(video_path: Path, *, work_dir: Path, config: PipelineConfig,
            emit: Callable[[PipelineEvent], None], should_stop: Callable[[], bool]) -> AnalysisArtifacts
def reevaluate(artifacts: AnalysisArtifacts, *, role_overrides: list[RoleOverride],
               zone_overrides: list[Zone], base: FindingSet) -> FindingSet   # чистая функция, < 2 с
```

- `PipelineEvent = {phase, processed_until_sec|None, observations_delta: list[Observation], message|None}`. Сервис сохраняет события в SQLite, и интерфейс читает их через `CheckDetail`.
- Фазы и их смысл: `decode` (проверка и чтение кадров с частотой `analysis_fps`), `perception` (люди, позы, объекты, треки; эмитит `processed_until_sec` каждые ~5 с видео), `facts` (схема сцены, роли, наблюдения, покрытие), `rules` (вычисление каталога), `evidence` (извлечение кадров в полном разрешении и JPEG).
- Артефакты в `data/checks/{id}/`: `video.<ext>`, `meta.json`, `tracks.jsonl`, `layout.r{n}.json`, `observations.json`, `coverage.json`, `findings.r{n}.json`, `frames/{frame_id}.jpg`, `frames/{frame_id}.overlay.jpg`, `run.json`.
- `should_stop` проверяется между фазами и каждые 50 кадров. При остановке сервисом (перезапуск) задание возвращается в очередь и начинается заново. Частичные артефакты перезаписываются, а не дополняются.
- Бюджет по умолчанию: `analysis_fps` подбирается в 01.4 так, чтобы видео 145 с обрабатывалось не дольше 120 с на целевом железе. Цель для 60 с: не дольше 60 с на GPU.

### C-RULES. Каталог правил (`rules/catalog/*.yaml`, производитель 02.3; потребители 02.4, 03.1 `/api/ui/rules`, 04.4, 04.5, 07.1)

```yaml
rule_id: V-COUNT-ABSENT
type_label: "Пересчёт денег без клиента у кассы"
norm_refs: [S231-p20, S231-p31]
requires_visible: [cashier, desk]            # без них окно получает not_checkable
condition:                                     # интерпретирует backend/navidu/rules/engine.py
  all:
    - observation: cash_in_hands               # или counting_motion
      subject_role: cashier
      min_duration_sec: 2.0                    # калибруется в 02.4, файл rules/calibration/thresholds.yaml
    - absent: person_in_zone
      role: client
      zone: client_window
      overlap_ratio_min: 0.8
timecode_policy: max_evidence                 # max_evidence | onset | obligation_due
what_seen: "Кассир держит деньги {duration_s} с, клиента у окна нет{since_clause}."
why_may_violate: "Наличные принимаются и пересчитываются в присутствии клиента."
not_claimed: "Сумму и соответствие документу по видео не проверяем."
```

Условия описываются только через типы фактов из C-DATA и временные отношения (`all`, `any`, `absent`, `before`, `after`, `within_sec`, `overlap_ratio_min`, `min_duration_sec`). Пороги выносятся в `rules/calibration/thresholds.yaml` (владелец 02.4), чтобы калибровка не правила каталог.

### C-SRC. Источники норм (`rules/sources/*.yaml`, производитель 01.2)

```yaml
clause_id: S231-p20
document:
  title: "Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня, …"
  act: "Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231"
  registry_code: V1900019680
  url: "https://adilet.zan.kz/rus/docs/V1900019680"
  edition_checked: "2026-10-17"        # дата редакции/проверки, заполняет 01.2
locator: "глава 2, параграф 2, пункт 20"
quote: "Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента."
verified_verbatim: false               # true только после дословной сверки с adilet.zan.kz
visual_test: "присутствие клиента у окна во время приёма и пересчёта"
not_visual: "сумма, оформление документа"
```

Интерфейс и API показывают цитату только при `verified_verbatim: true`, иначе только документ и пункт. Если пункт не найден в актуальной редакции, правило выключается, а не остаётся с неточной ссылкой.

### C-STORE. Хранение, очередь, срок жизни

- SQLite `data/navidu.sqlite3` (WAL). Таблицы: `checks` (id, file_name, sha256, status, priority api|ui, client_ip_hash, created_at, started_at, finished_at, lease_until, attempts, error_json, is_example, pipeline_version), `events` (check_id, seq, phase, processed_until_sec, payload_json), `finding_sets` (check_id, revision, json, created_at), `decisions` (check_id, finding_id, revision, value, note, decided_at). Миграции в `backend/navidu/service/migrations/`.
- Очередь: обработчик берёт задание `UPDATE … SET lease_until=now+90s WHERE status='queued' ORDER BY priority, created_at LIMIT 1` и продлевает аренду каждые 30 с. После перезапуска задания с истёкшей арендой возвращаются в `queued`, `attempts += 1`; после 3 попыток задание получает `failed` с кодом `RETRIES_EXHAUSTED`.
- Параллелизм: `NAVIDU_WORKERS` (1 на GPU по умолчанию; значение подбирается в 05.2 по замеру).
- Срок жизни: исходное видео удаляется через `NAVIDU_RETAIN_VIDEO_HOURS` (24), кадры-доказательства и JSON хранятся `NAVIDU_RETAIN_RESULTS_DAYS` (7). Пример не удаляется. Каталог `data/` исключён из git и из всех обучающих скриптов: `training/` читает только `NAVIDU_DATASET_DIR`.

### C-CONFIG. Переменные окружения (`.env.example`, владелец 03.2)

`NAVIDU_API_KEYS`, `NAVIDU_DATA_DIR`, `NAVIDU_MODELS_DIR`, `NAVIDU_DATASET_DIR` (только обучение), `NAVIDU_DEVICE` (`cuda`|`cpu`|`auto`), `NAVIDU_WORKERS`, `NAVIDU_MAX_UPLOAD_MB`, `NAVIDU_MAX_DURATION_SEC`, `NAVIDU_UI_MAX_ACTIVE_PER_IP`, `NAVIDU_RETAIN_VIDEO_HOURS`, `NAVIDU_RETAIN_RESULTS_DAYS`, `NAVIDU_EVIDENCE_OVERLAY`, `NAVIDU_EXAMPLE_CHECK_ID`. Секреты в репозиторий не попадают.

### C-LANDING. Пакет лендинга (производитель 04.5, точка подключения 05.1)

- Путь `landing/`, npm-workspace `@navidu/landing`, React 18 + TypeScript, без собственного роутера и без зависимостей от `web/src/*`.
- Экспорт: `export function Landing(props: LandingProps): JSX.Element` из `landing/src/index.ts`.

```ts
export interface LandingProps {
  dashboardHref: string;               // "/app"
  onNavigate?: (href: string, e: React.MouseEvent) => void; // для клиентского перехода без перезагрузки
  rules: LandingRule[];                // строки блока «Что проверяет», из /api/ui/rules или статического экспорта каталога при сборке
}
export interface LandingRule { ruleId: string; typeLabel: string; visibleSign: string; normShort: string; }
```

- Стили: всё под корневым `<div data-surface="landing" class="nv-landing">`, CSS Modules или классы с префиксом `nv-landing-`. Глобальные стили, reset и `@font-face` пакет не задаёт (их даёт `web/src/styles/`, задача 04.1). Значения берутся только из `--nv-*` (`identity/tokens/tokens.css`).
- Ассеты: SVG из `identity/art/` и `identity/logo/` импортируются через относительный путь из пакета (`../../identity/...`) или копируются в `landing/src/assets/` с пометкой источника.
- Для автономной разработки внешний исполнитель запускает `landing/dev/` (Vite playground с моковыми `rules` из `landing/dev/rules.sample.json`, помеченными как пример). Он не владеет `/app`, backend, `identity/` и `.vibemachina/`.
- Подключение: `web/src/app/routes.tsx` рендерит `<Landing dashboardHref="/app" rules={rules} onNavigate={navigate}/>` на `/`.

### C-EVAL. Оценка (производитель 02.4, потребитель 05.2, 07.1)

- Разметка нормализуется в `eval/labels.csv`: `video_id, has_violation (0/1), label_sec (float|пусто), type_code (заполняет команда в 01.1 вручную, может быть пусто)`. Исходный формат файла организаторов устанавливает 01.1, конвертер лежит в `eval/convert_labels.py`.
- Метрики на видео (как в критериях): точность вердикта, полнота по видео с нарушением, доля ложных вердиктов на чистых видео, медиана и 90-й перцентиль `|timecode − label_sec|`, совпадение типа там, где `type_code` известен. Разбиение по видео: leave-one-video-out (или 5 фолдов по видео); один ролик не может оказаться и в обучении, и в проверке.
- Отчёт `eval/reports/<дата>-<pipeline_version>.md` и `.json`. Тестовые видео жюри в оценку не попадают никогда.

### C-MODEL. Границы обучения и применения моделей

- **Обучение** (только `training/`, PyTorch): данные берутся из `NAVIDU_DATASET_DIR` (датасет организаторов), собственных записей команды и аннотаций `research/dataset/annotations/`. Разбиение всегда по роликам. Записи из `data/checks/` (включая видео жюри) не читаются ни одним обучающим скриптом; это проверяется тестом, который ищет в `training/` и `eval/` обращения к `NAVIDU_DATA_DIR`.
- **Что обучается**: (1) детектор объектов (деньги, сумки, телефоны, хранилище, стойка, лоток) дообучением открытой модели; (2) пороги фактов и правил и смещения таймкода (калибровка, не нейросеть). Люди и позы используются как есть, без дообучения, если 01.4 не покажет обратного.
- **Применение** (локальный runtime выбранной модели: ONNX Runtime или PyTorch; ONNX-only не является требованием кейса): вход — кадры RGB после поворота, прореженные по времени до `analysis_fps`; масштабирование до входа модели с сохранением пропорций. Выход каждой модели сразу переводится в `Track`/`Observation` (C-DATA) и в нормированные координаты.
- **Версии**: `models/manifest.json` хранит имя, источник, sha256, лицензию и класс для каждого файла весов; `pipeline_version = <semver пакета>+<короткий хеш манифеста и rules/calibration>`. Версия пишется в `run.json`, в `CheckDetail` (через `health`) и в лист проверки. Переиспользование результатов по sha256 видео допустимо только при совпадении `pipeline_version`.
- **Смысл результата**: эпизод означает «на записи видны признаки, соответствующие условию правила», а не «нарушение доказано». Оценки моделей (`score`) внутренние и не показываются как «уверенность в процентах».

## Этапы и зависимости

| Этап | Результат | Файл с задачами | Зависимости и параллельная работа |
| --- | --- | --- | --- |
| 01 | Основа: схема данных и типы нарушений, сверенные нормы, контракты в коде, выбор моделей с замером | [stages/01-foundation.md](stages/01-foundation.md) | Старт сразу и параллельно: 01.1, 01.2, 01.3, 01.4. Одновременно можно начать 04.5 (лендинг) |
| 02 | Движок: от кадров к эпизодам с кадрами, описанием и нормой, калибровка по разметке | [stages/02-engine.md](stages/02-engine.md) | 02.1 после 01.1+01.3+01.4; 02.2 после 01.1+01.3+01.4; 02.3 после 01.2+01.3. Три задачи идут параллельно. 02.4 собирает их |
| 03 | Сервис: очередь, хранение, API v1 и API интерфейса, образ, офлайн-запуск, публичный адрес | [stages/03-service.md](stages/03-service.md) | 03.1 после 01.3, параллельно с 02.x и 04.x (до готовности движка использует контрактный фиктивный конвейер); 03.2 после 03.1+01.4 |
| 04 | Интерфейс: рабочее место проверяющего, лист проверки, нормы, лендинг | [stages/04-interface.md](stages/04-interface.md) | 04.1 после 01.3; 04.2, 04.3, 04.4 параллельно после 04.1 по фикстурам контрактов; 04.5 без зависимостей |
| 05 | Сквозная интеграция, пример, устойчивость к ракурсу, нагрузке и сбоям | [stages/05-integration.md](stages/05-integration.md) | 05.1 после 02.4, 03.2, 04.2, 04.3, 04.4, 04.5; 05.2 после 05.1 |
| 06 | Код-ревью по исходному брифу и контрактам, исправления | [stages/06-review.md](stages/06-review.md) | 06.1 после 05.2 (слепой проход), 06.2 после 06.1, 06.3 после 06.2 |
| 07 | README, материалы сдачи, финальный выпуск, передача ключа, защита | [stages/07-delivery.md](stages/07-delivery.md) | 07.1 после 06.3; 07.2 после 07.1 |

Первая волна (сразу после утверждения): 01.1, 01.2, 01.3, 01.4, 04.5. Вторая волна (после 01.3): 03.1, 04.1. После 01.1/01.2/01.4: 02.1, 02.2, 02.3. После 04.1: 04.2, 04.3, 04.4. Точки соединения: 02.4 (движок целиком), 05.1 (продукт целиком).

Интерфейс и сервис не ждут движок: 04.x работают по фикстурам `contracts/fixtures/*.json` (из 01.3), 03.1 до готовности 02.4 подключает `FakePipeline` с реальными событиями из фикстур. Фикстуры не считаются доказательством готовности возможности; финальная приёмка идёт на реальном конвейере в 05.1.

## Общие проверяемые сценарии

| ID | Реальный вход и условия | Действие → наблюдаемый результат | Отказ/восстановление | Задачи и контракты |
| --- | --- | --- | --- | --- |
| S01 | Обучающий ролик с меткой «нарушение» и таймкодом T | POST с ключом → `job_id`; GET до `done` ≤ 300 с; `verdict=violation`; есть эпизод с `|timecode_sec − T|` в пределах допуска, который устанавливает 02.4 (цель ≤ 2 с, не измерено); 1–3 декодируемых JPEG; описание содержит норму | — | 02.4, 03.1, 05.1; C-API-V1 |
| S02 | Обучающий ролик без нарушения | `verdict=no_violation`, `violations=[]` | — | 02.4, 05.1 |
| S03 | Запрос без ключа, с неверным ключом; файл .txt под именем .mov | 401 без создания задания; 415 `UNSUPPORTED_MEDIA` | клиент получает понятный код | 03.1; C-ERR |
| S04 | Новый посетитель, ролик ~60 с | `/` → «Проверить запись» → загрузка → видимые фазы и растущая лента → результат; цель: результат не позже ~2 мин на целевом GPU (замерить в 05.2) | неверный файл: сообщение у зоны загрузки | 04.2, 04.3, 04.5, 05.1 |
| S05 | Результат с эпизодом V-OUTSIDER по человеку `p4` | Проверяющий меняет роль `p4` на «Инкассатор» → эпизод исчезает, вердикт пересчитан, решения по другим эпизодам сохранены, ревизия +1; API v1 по-прежнему отдаёт ревизию 1 | устаревшая `base_revision` → 409, введённое не теряется | 02.3, 03.1, 04.3; C-UI-API |
| S06 | Ролик, где стол перекрыт на участке | Правила, требующие стол, получают `not_checkable` на участке; эпизодов из отсутствия данных нет; интерфейс показывает штриховку и текст | — | 02.2, 02.3, 04.3 |
| S07 | 5 записей по ~145 с отправлены одновременно через API | Каждая `done` ≤ 300 с от своего POST на целевом железе; параллельная загрузка из интерфейса не задерживает API | очередь переполнена → 429 с `Retry-After` | 03.1, 03.2, 05.2 |
| S08 | Остановка обработчика посреди анализа | После перезапуска задание продолжается заново и завершается, дублей нет | 3 неудачи → `failed` `RETRIES_EXHAUSTED` | 03.1, 05.2 |
| S09 | Контейнер запущен с `--network none` после загрузки весов | Запись обрабатывается полностью | — | 03.2 |
| S10 | Своя запись команды под другим углом (не участвовала в настройке) | Схема сцены строится автоматически; вердикт и тип совпадают с ручной разметкой команды; расхождения фиксируются в отчёте | — | 02.2, 05.2; C-EVAL |
| S11 | Любой эпизод | Каждая `norm_refs` указывает на существующий `clause_id`; цитата показана только при `verified_verbatim=true` | несверенный пункт → только ссылка; ненайденный → правило выключено | 01.2, 02.3, 04.4 |
| S12 | Пример на `/app` | Открыть пример → результат собственной записи команды; решения и пересчёт работают как предпросмотр и не сохраняются | без записи кнопка «Открыть пример» не выводится | 05.1 |

### Один убедительный пример продукта

Человек открывает `/` и за несколько секунд видит, что это: заголовок «Смотрите главное, а не всю запись», сцена кассы сверху и карточка эпизода с пунктом нормы. Он жмёт «Проверить запись», перетаскивает свою запись .mov, и через секунды появляется лента: кассир на месте, клиент у окна. Когда приходит результат, плеер уже стоит на кадре 00:44,2, где кассир держит пачку денег, а окно клиента пусто. Справа видно наблюдение, требование п. 20 и кнопки решения. Проверяющий жмёт «Не видно на записи» на сомнительном эпизоде или переводит человека в «Инкассатор», и вердикт честно пересчитывается. Затем он печатает лист проверки. Существенный эффект: несколько отмеченных секунд вместо полного просмотра, и каждое замечание выдерживает перепроверку.

### Критерии жюри (раздел 9 кейса) → задачи и доказательства

| № | Критерий (вес) | Где достигается | Чем подтверждаем |
| --- | --- | --- | --- |
| 1 | Качество обнаружения (25) | 02.1–02.4 (полнота, подавление ложных), 05.2 | Отчёт C-EVAL: точность вердикта, полнота, ложные на чистых видео; S01, S02 |
| 2 | Качество локализации (15) | политика таймкода и выбор кадра в 02.3, калибровка смещения в 02.4 | медиана и p90 `|Δt|`; основной кадр с разметкой; S01 |
| 3 | Качество классификации (15) | каталог типов 01.1/01.2, правила 02.3 | совпадение типа на видео с ручным `type_code` |
| 4 | Обоснованность правил (15) | 01.2 (дословная сверка), 02.3 (шаблоны из фактов), 04.4 | S11; описание без домыслов; справочник «не проверяется по видео» |
| 5 | Устойчивость (10) | автосхема сцены 02.2, аугментации 02.4, свои ракурсы 05.2 | S10, отчёт по ракурсам |
| 6 | Готовность продукта (10) | 04.1–04.5, 03.1 (очередь, ошибки), 05.1 | S03, S04, S07, S08; проверка пользователя в TESTING |
| 7 | Техническое решение (10) | архитектура PROJECT, 03.2 (офлайн, образ), 01.4 (лицензии) | S09; таблица лицензий; README |

Критерии задают минимум: интерфейс и обоснование доводятся до состояния, в котором проверяющему удобно работать, а не только «засчитывается».

## Поставки и проверка

- Каждая задача содержит подшаги с готовностью и границей commit в своём файле этапа. Связанные подшаги одного исполнителя образуют одну поставку; промежуточный отчёт не останавливает исполнителя.
- Git, checkpoint и общие файлы (`.vibemachina/`, `identity/`, корневые конфиги вне назначенных путей) ведёт MAIN. Проверочная команда checkpoint настраивается в `vibemachina.checkpoint.json` после появления реальных проверок (01.3, 04.1): `ruff`, `pytest backend/tests`, `npm run typecheck`, `npm run build`, контрактный тест `pytest backend/tests/contracts`. Ни одной «проверки», которая ничего не проверяет.
- Визуальную проверку выполняет пользователь (TESTING.md). Агенты проверяют код, домен, HTTP и API. Браузерная автоматизация запрещена правилами VibeMachina.
- Свободные ресурсы GPU нужны на 01.4, 02.1, 02.4, 05.2; остальные задачи работают на CPU.

## Утверждение

2026-10-17 пользователь прямо поручил реализовать переданное решение: «Этим запросом я поручаю реализацию переданного решения с поправками ниже». Поправки: Lado латиницей, новый простой лендинг, сохранённая айдентика, полный локальный продукт и подготовка L40S; без презентации/речи/рекламного видео.

## Уточнения исполнения 2026-10-17

- Пользователь утвердил реализацию переданного решения с новой латинской надписью Lado и упрощённым лендингом. Речь, слайды и рекламное видео исключены.
- C-UI-API: добавлены status=cancelled и POST /api/ui/checks/{id}/cancel. Retry разрешён для failed/cancelled. V1 отображает отмену как failed/CANCELLED; успешная схема кейса неизменна. Health дополняется model_status (loading/ready/missing/error), limitations, device и example_check_id.
- C-LANDING не содержит presentationHref; продуктовые строки читает из contracts/product.json.
- 04.1 может начать по уже согласованным контрактам PLAN параллельно с их генерацией 01.3; проверки типов и финальная интеграция требуют фактических contracts/ts/types.gen.ts. Один UI-исполнитель последовательно владеет 04.1–04.4, без искусственного дробления тесных зависимостей.
- C-EVAL группирует разбиение по SHA256 исходного видео: в open.zip найдены побайтовые дубликаты с разными таймкодами меток. Не выдавать модельное описание сцены за ручную ground truth. Банковские видео/кадры не передаются внешнему AI даже при исследовании.
- Публичный сервер ещё не приобретён: сейчас готовятся образ, конфигурация Brev, переносимые веса и проверки. Фактический URL/скорость L40S/публичный прогон требуют оплаченного инстанса; не заявляются готовыми заранее.
