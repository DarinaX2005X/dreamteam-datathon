import { Route, Routes } from "react-router-dom";
import { Landing } from "../../../landing/src";
import { Shell } from "./Shell";
import { IntakePage } from "../features/intake/IntakePage";
import { CheckPage } from "../features/review/CheckPage";
import { SheetPage } from "../features/report/SheetPage";
import { NormsPage } from "../features/norms/NormsPage";
export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Landing dashboardHref="/app" rules={[]} />} />
      <Route element={<Shell />}>
        <Route path="/app" element={<IntakePage />} />
        <Route path="/app/checks/:id" element={<CheckPage />} />
        <Route path="/app/checks/:id/sheet" element={<SheetPage />} />
        <Route path="/app/norms" element={<NormsPage />} />
      </Route>
      <Route
        path="*"
        element={
          <main className="not-found">
            <p>Ошибка 404</p>
            <h1>Такой страницы нет</h1>
            <a href="/app">К новой проверке</a>
          </main>
        }
      />
    </Routes>
  );
}
