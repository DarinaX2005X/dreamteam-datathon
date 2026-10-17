import type React from "react";
import product from "../../contracts/product.json";
import logoUrl from "../../identity/logo/lado-logo.svg";
import markOnStageUrl from "../../identity/logo/lado-mark-on-stage.svg";
import deskSceneUrl from "../../identity/art/desk-scene.svg";

export interface LandingRule {
  ruleId: string;
  typeLabel: string;
  visibleSign: string;
  normShort: string;
}

export interface LandingProps {
  dashboardHref: string;
  onNavigate?: (href: string, event: React.MouseEvent<HTMLAnchorElement>) => void;
  rules: LandingRule[];
}

function ArrowIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h13M14 7l5 5-5 5" /></svg>;
}

function StartLink({ dashboardHref, onNavigate, className = "" }: Pick<LandingProps, "dashboardHref" | "onNavigate"> & { className?: string }) {
  const target = dashboardHref;
  return (
    <a className={`nv-landing-button ${className}`} href={target} onClick={(event) => onNavigate?.(target, event)}>
      {product.primaryAction}<ArrowIcon />
    </a>
  );
}

function AnalysisScene() {
  return (
    <figure className="nv-landing-scene" data-surface="stage" aria-labelledby="nv-scene-caption">
      <div className="nv-landing-scene-top">
        <span className="nv-landing-scene-source">Запись кассы · иллюстрация</span>
        <span className="nv-landing-scene-time">00:44,2</span>
      </div>
      <div className="nv-landing-camera" aria-hidden="true">
        <img src={deskSceneUrl} alt="" />
      </div>
      <div className="nv-landing-finding" data-surface="landing">
        <div className="nv-landing-finding-head">
          <span className="nv-landing-marker" aria-hidden="true" />
          <strong>Эпизод для проверки</strong>
          <span>00:44,2</span>
        </div>
        <p>Клиент передал деньги. Затем кассир отошёл от рабочего места.</p>
        <div className="nv-landing-finding-rule">Кадр · точное время · пункт нормы</div>
      </div>
      <div className="nv-landing-track" aria-hidden="true">
        <span>клиент у окна</span><i /><b /><em />
      </div>
      <figcaption id="nv-scene-caption">Поясняющая сцена. Реальный разбор строится по вашей записи.</figcaption>
    </figure>
  );
}

export function Landing({ dashboardHref, onNavigate, rules }: LandingProps) {
  return (
    <div className="nv-landing" data-surface="landing">
      <header className="nv-landing-header">
        <a className="nv-landing-logo-link" href="#top" aria-label={`${product.name} — наверх`}>
          <img src={logoUrl} alt={product.name} />
        </a>
        <StartLink dashboardHref={dashboardHref} onNavigate={onNavigate} className="nv-landing-button-small" />
      </header>

      <main id="top">
        <section className="nv-landing-hero" aria-labelledby="nv-landing-title">
          <div className="nv-landing-hero-copy">
            <h1 id="nv-landing-title">{product.tagline}</h1>
            <p className="nv-landing-lead">
              {product.name} просматривает запись кассовой операции и собирает проверяемый разбор:
              точный момент, кадр и связанную норму.
            </p>
            <StartLink dashboardHref={dashboardHref} onNavigate={onNavigate} />
            <p className="nv-landing-local">Анализ выполняется локально. Решение по каждому эпизоду принимает человек.</p>
          </div>
          <AnalysisScene />
        </section>

        <section className="nv-landing-evidence" aria-labelledby="nv-evidence-title">
          <div className="nv-landing-evidence-intro">
            <img src={markOnStageUrl} alt="" aria-hidden="true" />
            <h2 id="nv-evidence-title">У каждого эпизода есть основание</h2>
            <p>Откройте нужную секунду, сверьте кадр с нормой и зафиксируйте решение. Если роль человека определена неверно, её можно исправить и сразу пересчитать вывод.</p>
          </div>
          <div className="nv-landing-evidence-card">
            <div className="nv-landing-evidence-line"><span>Момент</span><strong>00:44,2</strong></div>
            <div className="nv-landing-evidence-line"><span>Что видно</span><strong>Последовательность действий в кадре</strong></div>
            {rules[0] ? (
              <div className="nv-landing-evidence-line">
                <span>Основание</span><strong>{rules[0].typeLabel}</strong><small>{rules[0].normShort}</small>
              </div>
            ) : null}
            <div className="nv-landing-evidence-decision">
              <span>Решение проверяющего</span>
              <div><b>Подтвердить</b><b>Не нарушение</b><b>Не видно</b></div>
            </div>
            <p className="nv-landing-evidence-note">По изображению нельзя подтвердить сумму, подписи и данные учётной системы. Такие факты Lado не додумывает.</p>
          </div>
        </section>

        <section className="nv-landing-final" aria-labelledby="nv-final-title">
          <div>
            <h2 id="nv-final-title">Разберите следующую запись.</h2>
            <p>Начните с файла кассовой операции и проверьте найденные моменты.</p>
          </div>
          <StartLink dashboardHref={dashboardHref} onNavigate={onNavigate} className="nv-landing-button-light" />
        </section>
      </main>
      <footer className="nv-landing-footer"><img src={logoUrl} alt={product.name} /><span>{product.description}</span></footer>
    </div>
  );
}
