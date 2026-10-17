# Состояние проекта
Phase: build
Approval: approved
Active stage: 01
Active task: 01.4

## Сейчас

Реализуется порученный 2026-10-17 продукт Lado. Пользователь явно утвердил основу переданного плана с новым названием латиницей, коротким лендингом и сохранённой айдентикой. Внутренний Python namespace navidu сохранён. Презентация и рекламное видео исключены.

Сервис, движок правил и рабочие экраны собраны. Настоящий offline RF-DETR UINT8 + ByteTrack проходит полный pipeline; семантические события пока не проверены — скачиваются pinned InternVL3.5-1B HF веса. Diagnostic detector-only не считается готовым анализом нарушений; по умолчанию отсутствие semantic runtime блокирует readiness.

## Следующие шаги

- Закрепить первый coherent checkpoint после source handover; приватные видео, веса и результаты inference исключены из Git.
- Завершить получение весов, реальный temporal inference и оценку по 40 уникальным видео; исправить подтверждённые ошибки качества.
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

Продуктовые API/worker сейчас запускаются deploy исполнителем для реального HTTP smoke; постоянный адрес будет записан после проверки.

## Работники

MAIN владеет .vibemachina/, Git и checkpoint scripts. Новые исполнители GPT-5.6 Sol по запросу пользователя; ранее запущенные продолжают свою работу.

- contracts_service: contracts/, backend/navidu/contracts/, service/, tests/contracts/, tests/service/, pyproject.toml, Makefile. Stable source handover; далее HTTP integration.
- model_probe: perception runtime, tests/perception/, research/spike/ и общий models/manifest.json, scripts/fetch_models.py. Stable readiness boundary; далее semantic adapter.
- model_download: только models/internvl3_5-1b/ и research/spike/downloads/, public weights. Скачивание продолжается.
- engine: ingest/scene/facts/rules/evidence/pipeline, tests/engine, rules/catalog/calibration. Stable handover, ждёт actual semantic integration.
- deploy: deploy/, Dockerfile/Compose/env, scripts/run_local/setup/preflight/smoke. Stable source handover; повторяет реальный smoke.
- frontend: web/, root package/lock. Код передан, 10 подтверждённых замечаний исправлены.
- landing_identity: landing/ и lado logo assets. Код передан; original art/mark сохранены.
- dataset: research/dataset/, eval/. Инвентаризация передана; ручная COCO-разметка не выдумана.
- norms: rules/sources/, research/norms/. Поставка принята.

## Передача

Рабочая папка C:/Users/user/Desktop/dreamteam, origin https://github.com/DarinaX2005X/dreamteam-datathon.git. Push и оплачиваемая инфраструктура не выполнялись. Первый checkpoint готовится; Git только у MAIN. Сохраняются семь этапов, цель — весь работающий продукт; успешный детектор не завершает поручение. Веса и банковские данные остаются локально. Следующее действие — real temporal path плюс meaningful measured evaluation, затем integration/review/delivery.
