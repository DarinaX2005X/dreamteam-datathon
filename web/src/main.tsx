import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./app/routes";
import "@fontsource-variable/open-sans/wght.css";
import "../../identity/tokens/tokens.css";
import "./styles/base.css";
import product from "../../contracts/product.json";
document.title = `${product.name} — ${product.description}`;
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  </React.StrictMode>,
);
