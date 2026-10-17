import { useEffect, useMemo, useState } from "react";
import type { RulesResponse } from "../../../../contracts/ts/types.gen";
import { api } from "../../api/client";
import { Empty, InlineNotice } from "../../ui";
import product from "../../../../contracts/product.json";
export function NormsPage() {
  const [data, setData] = useState<RulesResponse>();
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => {
    api
      .rules()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  const rules = useMemo(
    () =>
      data?.rules.filter((r) =>
        (r.type_label + r.visible_sign + r.not_claimed + r.rule_id)
          .toLowerCase()
          .includes(query.toLowerCase()),
      ) ?? [],
    [data, query],
  );
  return (
    <div className="page norms-page">
      <header className="page-heading">
        <p className="eyebrow">Справочник</p>
        <h1>Что проверяется</h1>
        <p>
          Каждый вывод связан с наблюдаемым признаком и пунктом нормативного
          акта. Если нужное не видно, {product.name} не делает вывод.
        </p>
      </header>
      <label className="search-field">
        <span>Поиск по правилам и признакам</span>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Например, пересчёт денег"
        />
      </label>
      {error && <InlineNotice tone="danger">{error}</InlineNotice>}
      {!data && !error && <p>Получаем каталог норм…</p>}
      {data && rules.length === 0 && (
        <Empty title="Ничего не найдено">
          Измените запрос или очистите строку поиска.
        </Empty>
      )}
      <div className="rules-table" role="table" aria-label="Проверяемые нормы">
        <div className="rules-table__head" role="row">
          <span>Тип и видимый признак</span>
          <span>Документ и пункт</span>
          <span>Граница вывода</span>
        </div>
        {rules.map((rule) => (
          <article role="row" id={rule.rule_id} key={rule.rule_id}>
            <div role="cell">
              <span className="rule-code">{rule.rule_id}</span>
              <h2>{rule.type_label}</h2>
              <p>{rule.visible_sign}</p>
            </div>
            <div role="cell">
              {(rule.sources ?? []).map((source) => (
                <section key={source.clause_id}>
                  <a
                    href={source.document.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {source.document.act}, {source.locator}
                  </a>
                  {source.verified_verbatim && source.quote && (
                    <blockquote>«{source.quote}»</blockquote>
                  )}
                </section>
              ))}
            </div>
            <div role="cell">
              <p>{rule.not_claimed}</p>
            </div>
          </article>
        ))}
      </div>
      {data && data.not_checkable.length > 0 && (
        <section className="not-checkable">
          <p className="eyebrow">За пределами записи</p>
          <h2>Не проверяется по видео</h2>
          <div>
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
        </section>
      )}
    </div>
  );
}
