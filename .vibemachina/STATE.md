# Состояние проекта
Phase: build
Approval: approved
Active stage: 01
Active task: 01.4

## Сейчас

Работа возобновлена пользователем 2026-10-17. Первый приоритет выполнен: приложение с исходной InternVL3.5-4B bf16 работает на разрешённом Brev `fierce-aquamarine-beaver`, `0.0.0.0:8080`. Пользователь подтвердил, что всё работает через свой NVIDIA-домен. Браузерные и визуальные проверки выполняет только пользователь.

По замечанию пользователя убраны редактор зон и вопросы о ролях. Причина перехвата воспроизведения — pointer-events на SVG-рамках поверх video; рамки теперь только отображаются. Web исправлен и развёрнут, asset index-BpncwtxV.js проверен HTTP. Следующее замечание — отражения людей в двери считались отдельными участниками. Reflection-патч1140de1 развёрнут и готов; пользователь сам проверяет его на сайте и отказался от отдельного воспроизведения замечания агентом. Качество исправления пока не подтверждено.

API и worker запущены. `/`, `/app`, `/api/ui/health` возвращают HTTP 200; `ready=true`, `analysis_ready=true`, `analysis_mode=experimental`, модель загружена на L40S. Все 20 файлов текущего модельного профиля прошли проверку размера/SHA. Idle GPU-memory после загрузки 9508 MiB. Проверен полный HTTP-путь одного development-ролика: 3032008 B, 14.734 с, upload 201, job done примерно через 15 с (опрос каждые 5 с), error_code=null. Это подтверждает исполнение, но не качество и не скорость длинных записей.

Экспериментальный режим включается только явным `NAVIDU_EXPERIMENTAL_ANALYSIS=1`. Реальные semantic windows и обычные проверки полноты обязательны; detector-only и all-not-checkable не дают ложный чистый результат. Jury API остаётся закрыт до принятого профиля качества. `models/validation.json` не подделывался. Исходная модель и её качество честно обозначены в UI.

Кейс перечитан: 25 баллов обнаружение, 15 локализация, 15 типы, 15 нормы, 10 устойчивость/иной ракурс, 10 продукт, 10 техника; максимум 300 с на видео, закрытого списка типов нет. Пользователь разрешил полную нагрузку сервера при одинаковой почасовой цене; общий бюджет обучения 2–3 часа. Оба пилота обучения завершены и отклонены. LoRA за 347.04 с: TP4/TN1/FP3/FN0 → TP4/TN0/FP4/FN0, MAE12.75→17.5с. Frozen-vision temporal MIL за 97.486 с: все четыре отрицательных validation-видео получили FP на каждой из пяти эпох, selected_epoch=null. Это proposal-метрики, не нормативные вердикты. Ни один обученный вариант не подключён.

Сейчас устраняются ошибки сквозного анализа на тех же 8 inner-validation видео. Профиль v3/w16 завершил 6/8: TP0/TN3/FP0/FN3 и 2 технические ошибки. Prompt v4 при лимите512 токенов завершил1/8: обрезался JSON. Лимит1024 и типизированная MODEL_OUTPUT_INVALID устранили сбои: текущий8/8 завершён, TP0/TN4/FP0/FN4, опросные времена10.1–75.2с. 23окна выдали деньги/сортировку, но ни одной передачи/пересчёта/связи партии. Числовой отчёт eval/reports/l40s-pipeline-4b-w16-t1024.json. Исправлена также client_window: зона устойчивого пребывания вместо всего пути клиента.

Идёт контролируемый max_patches4 A/B на том же коде3e84e50/w16/t1024. На первом positive smoke90.235с, 3реальных patches/кадр, max12150.5MiB; raw42events, accepted7. MAIN подготовил числовые причины отбраковки без изменения acceptance(8fc7772). После full8p4 server_launch развернёт diagnostics+исправление qualitybinding324c354 и повторит тот же smoke. Проверяется путаница индексов кадров/участников; до диагностики новая модель не включается. UI-качество и отражения проверяет пользователь.

## Обучение и данные

- Разбиение по уникальному SHA: 24 train, 8 inner validation, 8 holdout. Holdout не используется для выбора метода или параметров.
- Trainer: native PEFT LoRA4B, 16 кадров/2 fps, rank8 q/v, frozen vision/base, bf16, response-only loss, gradient checkpointing. 144 окна/18 optimizer steps в эпохе; максимум 3 эпохи; 66 label-blind validation windows за проход.
- Positive windows содержат размеченный момент; остальные окна положительного видео не превращаются в отрицательные. Одинаковое число окон на источник. Имена, SHA, folds и целевые времена не входят в модельный prompt.
- После research/review добавлены: 165 мин бюджет с резервом записи checkpoint; generate-only validation; baseline-validation.json; paired selection без потери любого ранее правильного ответа. Если решения те же, нужно среднее улучшение таймкода ≥0.5 с на тех же положительных источниках. Невалидные ответы запрещают выбор. Нет улучшения в первой эпохе → complete с selected_epoch=null; следующая эпоха допускается только по измеренному времени.
- Atomic progress.json/numeric progress.jsonl: стадия, шаги, loss, VRAM, stage ETA и остаток бюджета. Checkpoint пишется до validation. SIGTERM сохраняет адаптер на границе optimizer step; точного mid-epoch resume нет. Внешний supervisor должен ограничить запуск 180 мин, сохраняя приложение.
- Полный архив40 (263833600 B) доставлен; trainer проверил хеши используемых 24train+8validation. Малый6development (25098240 B, SHA ba5862dcf34d668d963a20488a582989d46a708bb1daa593b9230fac23ddb8bc) проверен и извлечён обновлённым helper. Transfer manifests разделены по SHA, subset→full безопасен.
- Книга `data/annotations/lado-annotation.xlsx` готова и передана ссылкой: 32 unique development/33 имени, 0 holdout, пустые человеческие J:R и отдельные эпизоды. Не перезаписывать существующую книгу. Ручные типы пока не потребляются trainer и не задерживают разрешённый binary/time pilot.
- `research/training/datasets-and-budget.md`: 8 кандидатов с первичными лицензиями/размерами. CAP близок по ракурсу, HoloAssist/CaptainCook4D — по ошибкам последовательности; немедленная большая загрузка не оправдана. Внешние действия не являются кассовыми юридическими метками. Frozen-feature temporal MIL — резерв после измеренного пилота.
- `research/norms/expanded-taxonomy.md`: текущие6 не исчерпывают область; ещё5 условных типов, наиболее сильный — выдача без пересчёта при клиенте по п31(4). Новые типы ещё не реализованы; словарь книги v1 не заменён.
- Второй trainer: frozen InternVL4B features2fps, global+2x2 pooling, фиксированная проекция256, train-only normalization, небольшой temporal MIL head с бинарной и point-time supervision. Фактический запуск завершён за97.486с:1785кадров32development,5эпох, остановка по patience без принятого улучшения. Чекпойнты сохранены приватно на сервере, числовой отчёт eval/reports/l40s-temporal-pilot.json. Torch CPU unit-tests5/5, metadata/data11/11, Ruff passed.
- Абляция patched8→16кадров на одном development-видео:14.067→11.020с,0→0findings,2→3events,0→0boundrelations; оба итоговых ответа FN. Promptv4 и числовые inference_metrics.json развёрнуты. Ни размер окна, ни новые поля пока не доказали улучшения итогового качества.

## Владельцы и границы

MAIN один редактирует `.vibemachina/` и выполняет Git writes. Новые работники GPT-5.6 Sol по прямой просьбе пользователя. Старые незавершённые файлы сохранены.

- server_launch — единственный SSH/GPU/deploy-владелец; оба обучения завершены, UI/reflection/geometry/JSON исправления развёрнуты. Измеряет8HTTP-вердиктов w16/t1024, затем max_patches4 при соблюдении времени/VRAM. Holdout не используется.
- baseline_preview — UI/reflection/geometry и аудит правил завершены. Готовит опциональный helper структурированного декодирования с LM Format Enforcer: только новый structured_output.py, его tests и optional dependency в pyproject. MAIN интегрирует после диагностики; принятие событий не ослабляется.
- dataset_training_research — завершил новые training/temporal_head.py, train_temporal.py, tests/test_temporal_head.py и review-исправления; файлы frozen. Runtime-интеграция только после измеренного улучшения.
- violation_scope — завершил нормативное исследование.

Все банковские видео, кадры и их описания остаются локально или на разрешённом self-hosted сервере. Внешним AI-сервисам их не отправлять. В ответы агента передаются только числовая диагностика, статусы, коды ошибок. Все тяжёлые прогоны и обучение — на L40S. Данные, модельные веса и секреты не коммитить. Презентации, питч и рекламное видео исключены.

## Проверки и ограничения

- Общий checkpoint-check: 255 Python tests passed, 16 environment skips; 24 web tests, Ruff, TypeScript, сборки landing/web прошли. Проверены 34 цитаты и 11 невизуальных условий. UI-исправление отдельно прошло24tests/TS/build. Reflection:186 tests passed15skips, затем исправлены2 устаревших fake-модели в scheduler tests; повтор reflection+proposal31passed, Ruffpassed.
- Reflection использует только явные локальные пары отражение→видимый источник, ≥3 последовательных кадра, known IDs/пространственное разделение, без blind merging/FaceID. Учитывается только доказанный интервал; неоднозначные пары исключены. Полностью покрытые действия привязываются к физическому источнику с прежними временем/доказательствами, ложная передача самому себе отбрасывается. Partial overlap не получает выдуманные новые границы; факт отбрасывается с числовой диагностикой. Качество на отражениях ещё не измерено.
- Исторический 1B development pair: TP0/FP0/TN1/FN1, максимум198.232с. Это не оценка текущей4B. Принятого профиля качества пока нет.
- V-COUNT-NO-CONTROLLER отключено до доказательства внутреннего контекста и обзора; V-OUTSIDER не автоматический нормативный положительный класс. Условия сумки/личного предмета/непрерывности нельзя угадывать.
- Статический audit:5enabledrules, текущим runtime достижимы COUNT-ABSENT/LEFT-POST/PERSONAL. CASH-UNSECURED/BAG заблокированы отсутствующим producer continuous_desk/collector_visibility. Для их включения нужны реальные наблюдения обзора; флаг нельзя просто переключить.
- InternVL3.5-8B добавлена как неактивный candidate с18пинами, Apache2,17.073GB. Метаданные совместимы с native4.57.6, runtimeнепроверен. Веса не скачаны, текущая4B остаётся активной; сначала проверяется формат её событий.
- F001 остаётся in-progress в TESTING.md: UI-исправления собраны и ждут пользователя; качество модели не закрыто. Последнее чтение TESTING не выявило новых записей.
- Для финального завершения всего продукта остаются полноценная model evaluation, проверка trained proposal в конвейере, freeze/holdout, Jury API readiness, скорость на длинных видео, автономный прогон, интеграционное review и актуальный README. HTTP-smoke не закрывает эти пункты.

## Сервисы и сохранение

Разрешённый сервер: NVIDIA L40S46068MiB,8CPU,144GiBRAM, ext4 /data250.9GiB. Проект `/data/dreamteam`, образ `lado-local:0.1.0`; API hostnetwork8080, worker GPU reservation1. Исходная база SOURCE_COMMIT=e274672e62b9ce67ac8867b794552ec44a0da9af плюс overlay backend/navidu и contracts. Модель4B лежит отдельно от будущих adapters. Не останавливать сервер и не удалять дубликаты/части весов без проверки.

Доступ: Windows PowerShell → WSL Ubuntu `/home/user/.local/bin/brev exec fierce-aquamarine-beaver` с одной строкой команды либо `@/mnt/c/.../script.sh`. Не передавать PowerShell-interpolated `$()`; использовать literal script. Rsync через `/home/user/.brev/ssh_config` с `--partial --append-verify` работает. Не конкурировать за SSH и не перезапускать авторизацию без оснований.

Управляющий dashboard: http://127.0.0.1:4177/; local product8020 остаётся API-only без готового semantic профиля. Рабочая папка C:/Users/user/Desktop/dreamteam. Origin https://github.com/DarinaX2005X/dreamteam-datathon.git, branch codex/lado-local-video. Последний отправленный checkpoint3e84e50. Данные/веса исключаются. Приватные runs: training/runs/pilot-4c5aa506-r2 и training/runs/temporal-pilot-1; progress и веса сохранены, оба selected_epoch=null.
