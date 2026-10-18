# Lado

**Всё важное — на виду.** Анализ кассовых операций по видеозаписи.

Lado находит эпизоды, которые требуют внимания внутреннего контроля, показывает
их на временной шкале и связывает наблюдаемые действия с условиями правил.
Проверяющий видит исходные кадры, подтверждает или отклоняет находку и сохраняет
своё решение.

[Открыть приложение](https://darinax2005x--lado-ladoservice-web.us-east.modal.run/app)

## Возможности

- Загрузка видео, очередь обработки и прогресс анализа.
- Отслеживание участников и объектов на записи.
- Эпизоды с таймкодами, объяснением и кадрами-доказательствами.
- Справочник правил с источниками и условиями применимости.
- Решения проверяющего, заметки и печатный лист проверки.
- HTTP API для загрузки записи и получения результата.

Автоматическая находка — признак возможного нарушения. Окончательное решение
принимает проверяющий. Нечёткие, перекрытые или отсутствующие в записи действия
могут остаться непроверяемыми.

## Как пользоваться

1. Откройте приложение и загрузите видеозапись.
2. Дождитесь обработки и выберите эпизод на временной шкале.
3. Сопоставьте видео, рамки объектов, кадры-доказательства и условие правила.
4. Подтвердите или отклоните эпизод; при необходимости добавьте заметку.
5. Откройте лист проверки и сохраните его в PDF через печать браузера.

Справочник находится по адресу `/app/norms`, лист отдельной проверки —
`/app/checks/<id>/sheet`.

## Устройство проекта

Конвейер декодирует видео с исходными временными метками, обнаруживает и
отслеживает объекты, анализирует последовательность действий, применяет правила
и формирует JPEG-доказательства. Временная модель — Qwen3-VL; правила и
нормативные источники хранятся отдельно от модели. API и интерфейс используют
общие контракты данных.

Для обучения детектора размечен датасет с руками, банкнотами,
удостоверениями и счётчиками банкнот.

Backend написан на Python: FastAPI, Pydantic, PyTorch, Transformers, ONNX Runtime,
PyAV и SQLite. Интерфейс — React, TypeScript и Vite.

| Каталог | Назначение |
| --- | --- |
| `backend/navidu/` | Анализ видео, правила, доказательства и HTTP-сервис. |
| `web/`, `landing/` | Интерфейс проверки и главная страница. |
| `contracts/` | Общие схемы данных и TypeScript-типы. |
| `rules/` | Исполняемые правила, источники и параметры. |
| `models/` | Манифест моделей; веса загружаются отдельно. |
| `training/`, `eval/` | Подготовка данных, обучение и оценка моделей. |
| `scripts/`, `deploy/` | Установка, запуск и размещение сервиса. |

Публичное имя — **Lado**. Внутренние имена `navidu` и `NAVIDU_*` сохранены
для совместимости Python-пакета и конфигурации.

## Запуск

Для GPU-сервера нужны Linux, Python 3.10+, FFmpeg и ffprobe, Node.js с npm,
драйвер NVIDIA и совместимая CUDA-сборка PyTorch. Конфигурация Modal использует
L4 с 24 ГБ видеопамяти. Точные версии зависимостей закреплены в
[backend/pyproject.toml](backend/pyproject.toml) и `package-lock.json`.

Команды выполняются из корня репозитория:

```bash
bash scripts/setup_local.sh --device=cuda --venv=backend/.venv --skip-models
backend/.venv/bin/python scripts/fetch_models.py --model rfdetr-nano-onnx --model qwen3-vl-4b
backend/.venv/bin/python scripts/fetch_models.py --model rfdetr-nano-onnx --model qwen3-vl-4b --verify-only
```

Создайте `.env` с собственным API-ключом и параметрами запуска:

```dotenv
NAVIDU_ENV=production
NAVIDU_API_KEYS=<ваш API-ключ>
NAVIDU_HOST=127.0.0.1
NAVIDU_PORT=8000
NAVIDU_DEVICE=auto
NAVIDU_DETECTOR_MODEL=rfdetr-nano-onnx
NAVIDU_SEMANTIC_MODEL=qwen3-vl-4b
NAVIDU_SEMANTIC_DTYPE=bf16
NAVIDU_REQUIRE_SEMANTIC=true
NAVIDU_EXPERIMENTAL_ANALYSIS=1
NAVIDU_ALLOW_EXPERIMENTAL_API=1
NAVIDU_QWEN_INPUT_MODE=video
NAVIDU_SEMANTIC_WINDOW_FRAMES=64
NAVIDU_SEMANTIC_FPS=1
NAVIDU_SEMANTIC_MAX_NEW_TOKENS=2048
NAVIDU_SEMANTIC_PROMPT_VARIANT=v8-focused
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

Экспериментальный режим включает текущую модель для проверки её результатов.
В API он обозначается заголовком `X-Analysis-Mode: experimental`. Для режима
с принятым профилем качества оба параметра `EXPERIMENTAL` отключаются.

```bash
backend/.venv/bin/python scripts/run_local.py
```

Приложение будет доступно на `http://127.0.0.1:8000/app`, состояние обработчика —
на `/api/ui/health`. Дополнительные настройки собраны в [.env.example](.env.example).
Развёртывание в Modal описано в [deploy/MODAL.md](deploy/MODAL.md), контейнерный
запуск — в [deploy/README.md](deploy/README.md).

## API

Все запросы `/api/v1` требуют заголовок `X-API-Key`.

```bash
curl --fail "$LADO_BASE_URL/api/v1/analyze" \
  -H "X-API-Key: $LADO_API_KEY" \
  -F "file=@/path/to/video.mov"
```

Ответ на принятую загрузку:

```json
{"job_id":"012345abcdef","status":"processing"}
```

Получение результата:

```bash
curl --fail "$LADO_BASE_URL/api/v1/jobs/012345abcdef" \
  -H "X-API-Key: $LADO_API_KEY"
```

Завершённый ответ содержит `job_id`, `status`, `verdict` и `violations`.
Каждый эпизод включает `type`, `timecode_sec`, `description` и от одного до трёх
`evidence_frames` с полями `timestamp_sec` и `image_base64` — JPEG в base64.
Ошибка обработки возвращает `status: "failed"` и `error`, без вердикта.

Схемы запросов и ответов находятся в [contracts/](contracts/).

## Правила и данные

Каталог проверок расположен в [rules/catalog/](rules/catalog/), нормативные
источники — в [rules/sources/](rules/sources/). Правило применяется только при
наличии необходимых наблюдений: участников, действий, временных связей и обзора
сцены. Отсутствие найденных эпизодов не подтверждает соблюдение всех требований.

Видео, модельные веса, ключи и рабочая база не входят в репозиторий.
Обработка использует модели в окружении сервиса. Исходные записи и результаты
хранятся на постоянном диске; сроки задаются параметрами `NAVIDU_RETAIN_VIDEO_HOURS`
и `NAVIDU_RETAIN_RESULTS_DAYS`. API-ключи защищают API v1; для закрытого размещения
интерфейса нужен внешний контроль доступа.

## Разработка

```bash
python -m ruff check --config backend/pyproject.toml backend/navidu backend/tests
python -m pytest backend/tests training/tests deploy/tests -q
npm test
npm run typecheck
npm run build
```

Python-модели в `backend/navidu/contracts/` служат источником контрактов.
JSON Schema и TypeScript-типы обновляются командой
`python -m navidu.contracts.generate`. Инструкции по обучению и оценке находятся
в [training/README.md](training/README.md) и [eval/README.md](eval/README.md).
