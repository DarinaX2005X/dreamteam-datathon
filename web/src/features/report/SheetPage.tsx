import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { RulesResponse } from "../../../../contracts/ts/types.gen";
import product from "../../../../contracts/product.json";
import logoUrl from "../../../../identity/logo/lado-logo-mono.svg";
import { ApiError, api, type UiCheck } from "../../api/client";
import { Button, InlineNotice } from "../../ui";
import { errors, formatDuration, formatTimecode } from "../../i18n/ru";

const decisions: Record<string, string> = {
  confirmed: "Подтверждено",
  rejected: "Не нарушение",
  not_visible: "Не видно на записи",
};

export function SheetPage() {
  const { id = "" } = useParams();
  const [check, setCheck] = useState<UiCheck>();
  const [rules, setRules] = useState<RulesResponse>();
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.check(id, controller.signal),
      api.rules(controller.signal),
    ])
      .then(([nextCheck, nextRules]) => {
        setCheck(nextCheck);
        setRules(nextRules);
      })
      .catch((requestError) => {
        if ((requestError as Error).name !== "AbortError") {
          const apiError = requestError as ApiError;
          setError(errors[apiError.code] ?? apiError.message);
        }
      });
    return () => controller.abort();
  }, [id]);

  if (error)
    return (
      <div className="page">
        <InlineNotice tone="danger">{error}</InlineNotice>
      </div>
    );
  if (!check || !rules)
    return <div className="page">Готовим лист проверки…</div>;
  if (check.status !== "done" || !check.finding_set) {
    return (
      <div className="page">
        <InlineNotice
          tone={
            check.status === "failed" || check.status === "cancelled"
              ? "danger"
              : "warning"
          }
        >
          {check.status === "failed"
            ? "Анализ завершился с ошибкой. Лист проверки ещё не создан."
            : check.status === "cancelled"
              ? "Анализ отменён. Лист проверки не создан."
              : "Анализ ещё не завершён. Вернитесь к проверке, чтобы следить за ходом."}
        </InlineNotice>
        <Link to={`/app/checks/${id}`}>Вернуться к проверке</Link>
      </div>
    );
  }

  const findingSet = check.finding_set;
  const verdict = findingSet.findings.some(
    (item) => item.decision?.value !== "rejected",
  );
  const sources = (rules.rules ?? []).flatMap((rule) => rule.sources ?? []);
  return (
    <div className="sheet-shell">
      <nav className="sheet-actions">
        <Link to={`/app/checks/${id}`}>← К разбору</Link>
        <Button onClick={() => window.print()}>Печать или PDF</Button>
      </nav>
      <main className="sheet">
        <header className="sheet-header">
          <img src={logoUrl} alt={product.name} />
          <div>
            <span>Лист проверки · ревизия {findingSet.revision}</span>
            <strong>{check.file_name}</strong>
          </div>
          <dl>
            <div>
              <dt>Дата анализа</dt>
              <dd>
                {new Date(
                  check.analyzed_at ?? check.created_at,
                ).toLocaleDateString("ru")}
              </dd>
            </div>
            <div>
              <dt>Длительность</dt>
              <dd>{formatDuration(check.duration_sec)}</dd>
            </div>
            <div>
              <dt>Источник</dt>
              <dd>
                {check.analysis_origin === "cache"
                  ? "Повторно использован локальный результат"
                  : "Локальный анализ"}
              </dd>
            </div>
          </dl>
        </header>
        <section
          className={`sheet-verdict ${verdict ? "is-alert" : "is-clear"}`}
        >
          <p>Вердикт текущей ревизии с учётом решений</p>
          <h1>
            {verdict
              ? "Обнаружены признаки возможных нарушений"
              : "Нарушение не обнаружено"}
          </h1>
          <small>
            Основание ревизии:{" "}
            {findingSet.basis === "reviewer"
              ? "уточнения проверяющего"
              : "автоматический анализ"}
            .
            {check.automatic_verdict &&
              ` Автоматический вердикт: ${check.automatic_verdict === "violation" ? "обнаружены признаки" : "нарушение не обнаружено"}.`}
          </small>
        </section>
        {findingSet.findings.map((finding, index) => {
          const refs = sources.filter((source) =>
            finding.norm_refs.includes(source.clause_id),
          );
          return (
            <article className="sheet-finding" key={finding.finding_id}>
              <header>
                <span>{index + 1}</span>
                <div>
                  <time>{formatTimecode(finding.timecode_sec)}</time>
                  <h2>{finding.type_label}</h2>
                </div>
                <strong>
                  {finding.decision?.value
                    ? decisions[finding.decision.value]
                    : finding.decision?.note
                      ? "Есть заметка, решение не принято"
                      : "Решение не принято"}
                </strong>
              </header>
              <img
                src={api.frame(id, finding.evidence[0].frame_id)}
                alt={`Основной кадр ${formatTimecode(finding.timecode_sec)}`}
              />
              <div className="sheet-finding__text">
                <section>
                  <h3>Что видно</h3>
                  <p>{finding.what_seen}</p>
                </section>
                <section>
                  <h3>Почему требует внимания</h3>
                  <p>{finding.why_may_violate}</p>
                </section>
                {refs.map((source) => (
                  <section key={source.clause_id}>
                    <h3>Норма</h3>
                    <p>
                      {source.document.act}, {source.locator}
                    </p>
                    {source.verified_verbatim && source.quote && (
                      <blockquote>«{source.quote}»</blockquote>
                    )}
                  </section>
                ))}
                {finding.decision?.note && (
                  <section>
                    <h3>Заметка проверяющего</h3>
                    <p>{finding.decision.note}</p>
                  </section>
                )}
              </div>
            </article>
          );
        })}
        <section className="sheet-secondary">
          <h2>Правила без замечаний</h2>
          <p>
            {findingSet.rules
              .filter((item) => item.status === "no_violation")
              .map((item) => item.rule_id)
              .join(", ") || "Нет"}
          </p>
          <h2>Непроверяемые участки</h2>
          {check.coverage.length ? (
            check.coverage.map((item, index) => (
              <p key={index}>
                {formatTimecode(item.start_sec)}–{formatTimecode(item.end_sec)}{" "}
                · {item.scope} · {item.reason}
              </p>
            ))
          ) : (
            <p>Не обнаружены.</p>
          )}
        </section>
        <footer>
          Результат является подсказкой для проверки человеком и не является
          юридическим заключением.
        </footer>
      </main>
    </div>
  );
}
