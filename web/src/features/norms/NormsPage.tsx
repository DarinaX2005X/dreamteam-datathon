import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import type {
  RulesResponse,
  RuleView,
  SourceClauseView,
} from "../../../../contracts/ts/types.gen";
import { api } from "../../api/client";
import { Empty, InlineNotice } from "../../ui";

function sourceLabel(source: SourceClauseView) {
  return `${source.document.act}, ${source.locator}`;
}

export function ruleMatchesQuery(rule: RuleView, query: string) {
  const normalized = query.trim().toLocaleLowerCase("ru");
  if (!normalized) return true;
  const sourceText = (rule.sources ?? [])
    .flatMap((source) => [
      source.document.title,
      source.document.act,
      source.document.registry_code,
      source.locator,
    ])
    .join(" ");
  return [
    rule.type_label,
    rule.visible_sign,
    rule.why_may_violate,
    rule.not_claimed,
    sourceText,
  ]
    .join(" ")
    .toLocaleLowerCase("ru")
    .includes(normalized);
}

export function ruleIdFromHash(hash: string) {
  try {
    return decodeURIComponent(hash.replace(/^#/, ""));
  } catch {
    return hash.replace(/^#/, "");
  }
}

export function NormsPage() {
  const location = useLocation();
  const [data, setData] = useState<RulesResponse>();
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [openRules, setOpenRules] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    api
      .rules()
      .then(setData)
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  const anchor = useMemo(() => ruleIdFromHash(location.hash), [location.hash]);

  useEffect(() => {
    if (!data || !anchor || !data.rules.some((rule) => rule.rule_id === anchor))
      return;
    setOpenRules((current) => new Set(current).add(anchor));
    requestAnimationFrame(() => {
      const summary = document
        .getElementById(anchor)
        ?.querySelector<HTMLElement>("summary");
      summary?.scrollIntoView({ block: "center" });
      summary?.focus({ preventScroll: true });
    });
  }, [anchor, data]);

  const rules = useMemo(
    () => data?.rules.filter((rule) => ruleMatchesQuery(rule, query)) ?? [],
    [data, query],
  );

  const setRuleOpen = (ruleId: string, open: boolean) => {
    setOpenRules((current) => {
      const next = new Set(current);
      if (open) next.add(ruleId);
      else next.delete(ruleId);
      return next;
    });
  };

  return (
    <main className="page norms-page">
      <header className="norms-heading">
        <h1>Что проверяется</h1>
        <p>Признаки в записи и нормы, с которыми они сверяются.</p>
      </header>

      <label className="search-field norms-search">
        <span>Найти правило или норму</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Например, пересчёт денег или пункт 5.11"
        />
      </label>

      {error && <InlineNotice tone="danger">{error}</InlineNotice>}
      {!data && !error && <p>Получаем список правил…</p>}
      {data && rules.length === 0 && (
        <Empty title="Ничего не найдено">
          Измените запрос или очистите строку поиска.
        </Empty>
      )}

      <section className="norm-rules" aria-label="Правила проверки">
        {rules.map((rule) => {
          const sources = rule.sources ?? [];
          return (
            <details
              className="norm-rule"
              id={rule.rule_id}
              key={rule.rule_id}
              open={openRules.has(rule.rule_id)}
              onToggle={(event) =>
                setRuleOpen(rule.rule_id, event.currentTarget.open)
              }
            >
              <summary>
                <span className="norm-rule__title">
                  <strong>{rule.type_label}</strong>
                  <span>
                    {sources.length
                      ? sources.map(sourceLabel).join("; ")
                      : "Источник не указан"}
                  </span>
                </span>
                <span
                  className={`rule-availability ${rule.enabled ? "is-enabled" : "is-disabled"}`}
                >
                  {rule.enabled ? "Поддерживается" : "Пока недоступно"}
                </span>
              </summary>

              <div className="norm-rule__body">
                <section>
                  <h2>Что видно в записи</h2>
                  <p>{rule.visible_sign}</p>
                  <p className="norm-rule__context">{rule.why_may_violate}</p>
                </section>

                <section>
                  <h2>Норма</h2>
                  {sources.length ? (
                    sources.map((source) => (
                      <div className="norm-source" key={source.clause_id}>
                        <a
                          href={source.document.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {sourceLabel(source)}
                        </a>
                        {source.verified_verbatim && source.quote && (
                          <blockquote>«{source.quote}»</blockquote>
                        )}
                      </div>
                    ))
                  ) : (
                    <p>Источник не указан.</p>
                  )}
                </section>

                <section className="norm-boundary">
                  <h2>Граница вывода</h2>
                  <p>{rule.not_claimed}</p>
                </section>
                <small className="rule-id">Код правила: {rule.rule_id}</small>
              </div>
            </details>
          );
        })}
      </section>

      {data && data.not_checkable.length > 0 && (
        <details className="not-checkable">
          <summary>Что нельзя проверить по видео</summary>
          <div className="not-checkable-list">
            {data.not_checkable.map((item) => (
              <article key={item.clause_id}>
                <h3>{item.locator}</h3>
                <p>{item.not_visual}</p>
                <a href={item.document.url} target="_blank" rel="noreferrer">
                  Открыть документ
                </a>
              </article>
            ))}
          </div>
        </details>
      )}
    </main>
  );
}
