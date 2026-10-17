# Состояние проекта
Phase: build
Approval: approved
Active stage: 01
Active task: 01.4

## Сейчас

Реализуется порученный 2026-10-17 продукт Lado. Пользователь явно утвердил основу переданного плана с новым названием латиницей, коротким лендингом и сохранённой айдентикой. Внутренний Python namespace navidu сохранён. Презентация и рекламное видео исключены.

Сервис, движок правил и рабочие экраны собраны. Настоящий offline RF-DETR UINT8 + ByteTrack проходит полный pipeline; pinned InternVL3.5-1B HF веса скачаны и проверены SHA256. Первый CUDA-прогон 8 кадров дал пустой результат за 81.23 с; профиль с 24 тайлами остановлен после четырёх минут у предела 4 GB VRAM. Контроль цвета и порядка изображений прошёл, но это не измерение нарушений. Сейчас проверяются короткие вопросы о действиях на development-видео; параллельно скачивается официальный InternVL3.5-2B как кандидат для измеренного сравнения. Качество ещё не принято.

F001: пользователь нашёл ложный чистый результат diagnostic-режима на Х-40 и сбой на Х-33. Обычная загрузка временно закрыта, диагностический worker остановлен. Вводится отдельная готовность полноценного анализа и запрет публикации diagnostic/unvalidated как обычного результата. Старые диагностические артефакты и решения сохраняются; повтор опубликованной диагностики получает новый check ID. Причина Х-33 техническая: старый загруженный код с новым YAML каталога; исправляется preflight/fingerprint и отказ от повторов конфигурационной ошибки. Х-33 относится к holdout: содержимое и semantic результаты для настройки не использованы.

## Следующие шаги

- Первый checkpoint d5ff273 сохранён на codex/lado-local-video: lint, 71 Python tests, 14 web tests, TS/build, 34 цитаты прошли. Push не выполнялся; приватные видео/веса/артефакты исключены.
- Довести реальный temporal inference по положительным и чистым development-видео; затем оценить 32 development и 8 holdout (fold0), без утечки дубликатов и ручного выдумывания типов.
- Принять сквозной API/UI путь, подготовленный пример и устойчивость, затем отдельное ревью, исправления, проверенный README и чистый запуск.
- Проверка внешнего вида принадлежит пользователю. Публичный Brev L40S ещё не приобретён; подготовка не подменяет фактические GPU/public проверки.

## Проверено

- open.zip: 42 MOV / 40 уникальных SHA256; 22 positive / 20 negative строк. Метки дубликатов Х-4/Х-47 57/68 секунд конфликтуют; разбиение по SHA. Реальные PTS обязательны: video VFR.
- 34 дословные цитаты норм РК сверены с официальным ЭКБ, 11 аспектов непроверяемы по видео. Каталог не требует выдуманного отдельного помещения; V-OUTSIDER отключён.
- Контракты Pydantic/OpenAPI/JSON Schema/TS и синтетические fixtures согласованы. API revision 1 неизменен после решений проверяющего; cache сохраняет происхождение и не копирует чужие заметки/роли.
- Service: 41 focused tests до дополнительных regression fixes, реальный worker kill/recovery, ffprobe, auth/limits/Range/revisions/retry/cancel. First real HTTP integration exposed video.json reserved-name bug; fix передан для rerun.
- Engine: 14 domain tests. X-1.mov 25.933 с: full offline detector pipeline 1 fps, 8 tracks, perception 13.399 с; все неподдерживаемые semantic rules not_checkable. Это не измерение качества нарушений.
- Frontend: 14 tests, typecheck, format:check, production build. Визуальная приёмка не выполнялась агентами.
- Ноутбук: i7-1260P, 31.73 GiB RAM, RTX2050 4 GiB. Рабочий runtime probe Python3.10; основной service env Python3.12. Минимум Python3.10 подтверждён.
- Docker29.4/Compose5.1 доступны; конфигурация проверена. L40S 48 GB/72 GB RAM/12 CPU/625 GB SSD не запущен и не измерен.

## Сервисы

Управляющий дашборд: http://127.0.0.1:4177/ — один штатный процесс этой сессии, runtime в .vibemachina/runtime/. Браузер автоматически не открывается.

Local: http://127.0.0.1:8020/

Продукт реально отвечает HTTP200 на / и health. После F001 старый owned stack остановлен; сейчас API-only на 8020, launcher PID235880 (перед изменением сверять принадлежность). Health ready=false; модельные прогоны выполняются отдельно. Frontend production build с основными исправлениями F001 готов, верхний каталог норм и возвращённый переход «Правила» дорабатываются по уточнению пользователя. После проверки semantic path полный worker будет запущен на том же порту. Нельзя считать прежние диагностические no_violation проверкой кассовых действий.

## Работники

MAIN владеет .vibemachina/, Git и checkpoint scripts. Новые исполнители GPT-5.6 Sol по запросу пользователя; ранее запущенные продолжают свою работу.

- contracts_service: contracts/, backend/navidu/contracts/, service/, tests/contracts/, tests/service/, pyproject.toml, Makefile. Исправлены cache source rebinding, effective loaded version, staging/dry-run и immutable media revisions; 48 tests на3.10 с actualFFmpeg/PyAVположительнымintegration,47+1runtime skipна3.12. Завершается structuredlogging.
- model_probe: perception runtime, tests/perception/, research/spike/ и общий models/manifest.json, scripts/fetch_models.py. Настоящий semantic adapter запущен наCUDA, текущий риск quality/performance. Fingerprint включает код/prompt/weights/config.
- model_download: только models/internvl3_5-1b/ и research/spike/downloads/, public weights. Завершено 14 файлов,2.138GB; SHA проверен. Optional deletion2.13GBtemporaryparts blocked by toolpolicy, части оставлены.
- engine: ingest/scene/facts/rules/evidence/pipeline, tests/engine, rules/catalog/calibration. Исправлены purestaging reevaluate, samplingcadence, robustdwellpost, coverage иparticipant/cashbinding. Добавляется observeddeparture person_left_zone + scopedrelation_id по явномуlocalmodelrelation.
- deploy: deploy/, Dockerfile/Compose/env, scripts/run_local/setup/preflight/smoke. Реальный diagnosticHTTPsmoke прошёл. Исправляет .env loading, Dockerassets/GPUcommands и проверяет самостоятельное окружение без глобальных Python-зависимостей.
- frontend: web/, root package/lock. F001:22tests/typecheck/format/buildpassed; этапы/elapsed, таблица наблюдений, высота видео/реальные bbox, retry нового ID и readiness исправлены. По уточнению из чата верхняя таблица норм меняется на раскрываемый список. Visual QA user-owned.
- landing_identity: landing/ и lado logo assets. Код передан; original art/mark сохранены.
- dataset: research/dataset/, eval/. Evalrunner/metrics/resume/holdoutfreeze подготовлены,20tests. Ждёт stableactualsemanticruntime, не конкурирует заGPU; ручная COCO-разметка не выдумана.
- norms: rules/sources/, research/norms/. Поставка принята.

## Передача

Рабочая папка C:/Users/user/Desktop/dreamteam, origin https://github.com/DarinaX2005X/dreamteam-datathon.git. Push и оплачиваемая инфраструктура не выполнялись. Первый checkpoint d5ff273 создан; Git только у MAIN. Сохраняются семь этапов, цель — весь работающий продукт; успешный детектор не завершает поручение. Веса и банковские данные остаются локально. Следующее действие — real temporal path плюс meaningful measured evaluation, затем integration/review/delivery.
