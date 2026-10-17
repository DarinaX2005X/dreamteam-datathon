# Лендинг Lado

Изолированный React 18 + TypeScript пакет для маршрута `/`. Название, слоган, описание и текст основной кнопки импортируются из `../contracts/product.json`; словесный знак — из `../identity/logo/lado-logo.svg`. Пакет не содержит роутер и не зависит от `web/src`.

```tsx
import { Landing, type LandingRule } from "@lado/landing";
import "@lado/landing/styles.css";

<Landing dashboardHref="/app" rules={rules} onNavigate={handleNavigate} />
```

`onNavigate` получает целевой `href` и исходное событие ссылки. Если обработчик не передан или не вызывает `preventDefault()`, работает обычный переход браузера. `rules` соответствует контракту `LandingRule { ruleId, typeLabel, visibleSign, normShort }`.

```powershell
npm --workspace landing run dev
npm --workspace landing run typecheck
npm --workspace landing run build
```

Песочница использует `dev/rules.sample.json`; все строки там явно помечены как примеры и не являются результатами анализа. Глобальный reset и шрифт предоставляет приложение. Песочница отдельно импортирует исходные токены `identity/tokens/tokens.css`.

Визуальное поведение проверяет пользователь. В пакет не входят браузерные или скриншотные тесты.
