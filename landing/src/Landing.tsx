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
  normsHref?: string;
  onNavigate?: (href: string, event: React.MouseEvent<HTMLAnchorElement>) => void;
  rules: LandingRule[];
}

function ArrowIcon() {
  return <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 12h13M14 7l5 5-5 5" /></svg>;
}

function GithubIcon() {
  // Official GitHub mark geometry from https://github.com/logos.
  return <svg className="nv-landing-github-icon" aria-hidden="true" viewBox="0 0 24 24"><path fill="currentColor" d="M12 .7A11.5 11.5 0 0 0 8.36 23c.58.1.79-.25.79-.56v-2.2c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18A10.98 10.98 0 0 1 12 6.04c.98 0 1.94.13 2.86.39 2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.71 5.38-5.29 5.67.42.36.78 1.07.78 2.16v3.22c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" /></svg>;
}

function ProjectLinks() {
  return (
    <div className="nv-landing-project-links">
      <a className="nv-landing-project-button nv-landing-project-button-github" href="https://github.com/DarinaX2005X/dreamteam-datathon" target="_blank" rel="noreferrer"><GithubIcon />GitHub</a>
      <a className="nv-landing-project-button" href="/presentation/lado-pitch.pdf" target="_blank" rel="noreferrer">Презентация · PDF</a>
      <a className="nv-landing-project-button" href="/presentation/lado-pitch.pptx" download>Скачать PPTX</a>
    </div>
  );
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
      <div className="nv-landing-track" aria-hidden="true">
        <span>клиент у окна</span><i /><b /><em />
      </div>
      <figcaption id="nv-scene-caption">Иллюстрация</figcaption>
    </figure>
  );
}

export function Landing({ dashboardHref, normsHref = "/app/norms", onNavigate }: LandingProps) {
  const accent = "на виду";
  const accentStart = product.tagline.toLocaleLowerCase("ru").indexOf(accent);
  const titleBefore = accentStart >= 0 ? product.tagline.slice(0, accentStart) : product.tagline;
  const titleAccent = accentStart >= 0 ? product.tagline.slice(accentStart, accentStart + accent.length) : "";
  const titleAfter = accentStart >= 0 ? product.tagline.slice(accentStart + accent.length) : "";
  return (
    <div className="nv-landing" data-surface="landing">
      <header className="nv-landing-header">
        <a className="nv-landing-logo-link" href="#top" aria-label={`${product.name} — наверх`}>
          <img src={logoUrl} alt={product.name} />
        </a>
        <nav className="nv-landing-header-actions" aria-label="Основная навигация">
          <a className="nv-landing-rules-link" href={normsHref} onClick={(event) => onNavigate?.(normsHref, event)}>Правила</a>
          <StartLink dashboardHref={dashboardHref} onNavigate={onNavigate} className="nv-landing-button-small" />
        </nav>
      </header>

      <main id="top">
        <section className="nv-landing-hero" aria-labelledby="nv-landing-title">
          <div className="nv-landing-hero-copy">
            <h1 id="nv-landing-title">{titleBefore}{titleAccent ? <span>{titleAccent}</span> : null}{titleAfter}</h1>
            <p className="nv-landing-lead">
              {product.name} просматривает запись кассовой операции и собирает проверяемый разбор:
              точный момент, кадр и связанную норму.
            </p>
            <StartLink dashboardHref={dashboardHref} onNavigate={onNavigate} />
            <ProjectLinks />
          </div>
          <AnalysisScene />
        </section>

        <section className="nv-landing-evidence" aria-labelledby="nv-evidence-title">
          <div className="nv-landing-evidence-intro">
            <img src={markOnStageUrl} alt="" aria-hidden="true" />
            <h2 id="nv-evidence-title">У каждого эпизода есть основание</h2>
            <p>Откройте нужную секунду, сверьте кадр с нормой и зафиксируйте решение. Подтвердите или отклоните находку и сохраните заметку к разбору.</p>
          </div>
          <div className="nv-landing-evidence-card">
            <article className="nv-landing-evidence-frame">
              <div className="nv-landing-evidence-thumb"><img src={deskSceneUrl} alt="Иллюстрация кадра кассовой операции с визуальной разметкой" /></div>
              <div><span className="nv-landing-timecode">00:44,2</span><h3>Действия собраны в один проверяемый эпизод</h3><p>Кадр остаётся рядом с описанием, чтобы вывод можно было перепроверить.</p></div>
            </article>
            <article className="nv-landing-evidence-source">
              <span>Сверка с нормой</span>
              <h3>Сначала последовательность, затем основание</h3>
              <p>Lado учитывает действия до и после кадра. В разборе подходящее правило открывается вместе с документом и пунктом.</p>
            </article>
            <div className="nv-landing-review-choice" aria-label="Пример решений проверяющего">
              <span>Проверяющий решает</span>
              <div><b>Подтвердить</b><b>Не нарушение</b><b>Не видно</b></div>
              <p>Решение сохраняется отдельно от автоматического вывода.</p>
            </div>
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
