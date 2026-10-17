# Состояние проекта
Phase: build
Approval: approved
Active stage: 01
Active task: 01.4

## Сейчас

Реализуется порученный 2026-10-17 продукт Lado. Пользователь явно утвердил основу переданного плана с новым названием латиницей, коротким лендингом и сохранённой айдентикой. Внутренний Python namespace navidu сохранён. Презентация и рекламное видео исключены.

Сервис, движок правил и рабочие экраны собраны. Полная development-пара Х-40/Х-21 на 1B закончена: TP0/FP0/TN1/FN1, оба файла технически успешны, максимум198.232с, подготовка35.144с. Профиль не принят; report в eval/reports/development-pair.*. Числовая диагностика нашла потерю второго человека при подтверждении трека; исправление проверено синтетической последовательностью. Содержательное качество этим не доказано.

Пользователь оплатил останавливаемый Brev `l40s-48gb.1x`, инстанс `fierce-aquamarine-beaver`, открыл HTTP8080 и разрешил полную настройку. Доступ через WSL Brev CLI работает; прежняя ошибка вызвана неверным разбором CLI-аргументов, повторный вход не нужен. Фактически проверены L40S46068MiB, driver565.57.01, 8vCPU,144GiBRAM и persistent ext4 `/data`250.9GiB. Проект `/data/dreamteam`; чистый source3e4a122 перенесён. Fetch4B+UINT8 и Docker build/API-up запущены в фоне; датасет263833600B передаётся. Обучение ещё не начиналось. Все тяжёлые прогоны только на L40S. Исходники и лёгкие проверки остаются в одном локальном Git.

Текущая просьба: запустить фоновое обучение и параллельно рабочее приложение на исходной модели, передать ссылку и команды прогресса/остатка, после чего приостановить активную работу в чате. Сервер при этом продолжает работать. Общий расход VRAM нужно измерить перед одновременным inference/training. Данные и результаты сохраняются на `/data`; сервер не останавливается без прямой команды. Перед будущей ночной остановкой — локальный экспорт результатов; диск оплачивается и на паузе, restart зависит от GPU capacity.

Принято настоящее обучение по существующим binary/time меткам: native PEFT LoRA4B, proposal-задача candidate/frame,24train(12/12),8inner-validation(4/4),8holdout(5/3) только после freeze. Положительные окна содержат размеченное время; отрицательные берутся только из чистых видео. Входы — кадры и относительное время; нет имён/SHA/фолдов. Подготовлены shared helpers и standalone runner:144 окна/эпоху,18 optimizer steps/эпоху, максимум3эпохи/54шага;66 label-blind validation windows за проход. Atomic progress.json и numeric progress.jsonl содержат actual steps/loss/VRAM/измеренный stageETA; checkpoint перед validation. Независимое ревью подтвердило границы train/validation; исправлены ранний failed-status и нормализация неполной группы градиентов,17focusedtests прошли. Обученный proposal направляет более плотную проверку, но не выдумывает события/роли/нормы и не обходит coverage. Native4B16файлов9.481GB pinned; исходные веса приложения и LoRA outputs раздельны. V-COUNT-NO-CONTROLLER отключено до измеренного доказательства контекста/обзора сцены.

F001: обычная загрузка временно закрыта, диагностический worker остановлен. Исправлены отдельная готовность полноценного анализа и запрет публикации diagnostic/unvalidated как обычного результата, preflight/fingerprint и отказ от повторов конфигурационной ошибки. Старые диагностические артефакты и решения сохранены; повтор опубликованной диагностики получает новый check ID. Х-33 относится к holdout: содержимое и semantic результаты для настройки не использованы. Правки интерфейса, лендинга и верхнего каталога норм собраны; 24 web tests, typecheck/build прошли, визуальная приёмка ждёт пользователя.

## Следующие шаги

- Checkpoints d5ff273, f49b044 и3e4a122 сохранены и отправлены в origin/codex/lado-local-video по прямой просьбе; удалённый SHA3e4a122eebde59846e1455d1d5044afe5b310ff4 подтверждён. Последний:206Python tests,1skip,11subtests,24webtests,lint,TS/build,34цитаты. Приватные видео/веса/артефакты исключены. Новый training launch package готовится к checkpoint.
- Довести remote baseline на4B, проверяемый API и независимый тренировочный процесс. Качество неподтверждённого профиля не объявлять принятым. После пользовательской паузы остаются actual model comparison,32development/8holdout, integration/review/README.
- R003 закрыт независимым ревьюером:8deploytests и матрица CPU/4/8/48GiB согласуют fetch/preflight/runtime; optional4B override ещё нужно включить в основной setup перед его выбором production default.
- Принять сквозной API/UI путь, устойчивость, закончить независимое ревью, исправления и проверенный README. Builder собственного примера реализован; собственной записи пока нет, кнопка примера скрыта.
- Проверка внешнего вида принадлежит пользователю. Сервер выбран и подключение разрешено; фактические GPU/public проверки впереди.

## Проверено

- open.zip: 42 MOV / 40 уникальных SHA256; 22 positive / 20 negative строк. Метки дубликатов Х-4/Х-47 57/68 секунд конфликтуют; разбиение по SHA. Реальные PTS обязательны: video VFR.
- 34 дословные цитаты норм РК сверены с официальным ЭКБ, 11 аспектов непроверяемы по видео. Каталог не требует выдуманного отдельного помещения; V-OUTSIDER отключён.
- Контракты Pydantic/OpenAPI/JSON Schema/TS и синтетические fixtures согласованы. API revision 1 неизменен после решений проверяющего; cache сохраняет происхождение и не копирует чужие заметки/роли.
- Service/engine: actual FFmpeg/PyAV integration, lease recovery, auth/Range/revisions/cache, staged reevaluation, expiry, publication/readiness gates прошли. Последняя общая проверка включает 179 тестов backend/deploy/eval/spike; качество модели отдельно.
- Frontend: 24 tests, typecheck, format:check, production build. Визуальная приёмка не выполнялась агентами.
- Ноутбук: i7-1260P, 31.73 GiB RAM, RTX2050 4 GiB. Рабочий runtime probe Python3.10; основной service env Python3.12. Минимум Python3.10 подтверждён.
- Docker29.4/Compose5.1 доступны; отдельный Python runtime без глобальных зависимостей установлен. Проверка реального CPU-образа и нового удалённого L40S выполняется deploy.

## Сервисы

Управляющий дашборд: http://127.0.0.1:4177/ — один штатный процесс этой сессии, runtime в .vibemachina/runtime/. Браузер автоматически не открывается.

Local: http://127.0.0.1:8020/

Продукт реально отвечает HTTP200 на / и health. После F001 работает API-only на8020; последний отчёт deploy:launcherPID198204 (перед изменением сверять принадлежность). Health ready=false, UI с исправлениями F001/норм доступен. Для первой удалённой проверки запрошен SSH forward127.0.0.1:8021→remote8080; пока не подтверждён. HTTP8080 открыт пользователем в Brev, публичный рабочий адрес ещё не получен.

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

Рабочая папка C:/Users/user/Desktop/dreamteam, origin https://github.com/DarinaX2005X/dreamteam-datathon.git. Checkpoint3e4a122 уже в origin/codex/lado-local-video; Git только у MAIN. Инстанс приобретён, настройка разрешена; банковские данные допустимы только в local/self-hosted inference, не во внешнем AI API и не в ответах модели-агента. Сохраняются семь этапов и цель всего продукта. Ближайшая граница — живые серверные приложение/обучение с прогрессом, затем запрошенная пауза чата; полноценное окончание продукта этим не объявляется.
