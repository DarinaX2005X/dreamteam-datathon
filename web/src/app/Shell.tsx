import { Link, NavLink, Outlet } from "react-router-dom";
import logoUrl from "../../../identity/logo/lado-logo.svg";
import product from "../../../contracts/product.json";
export function Shell() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link className="brand" to="/">
          <img src={logoUrl} alt={product.name} />
        </Link>
        <nav aria-label="Основная навигация">
          <NavLink to="/app" end>
            Новая проверка
          </NavLink>
          <NavLink to="/app/norms">Нормы</NavLink>
        </nav>
      </header>
      <main id="main">
        <Outlet />
      </main>
      <div className="sr-only" role="status" aria-live="polite" />
    </div>
  );
}
