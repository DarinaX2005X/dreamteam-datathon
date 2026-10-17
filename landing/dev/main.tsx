import React from "react";
import ReactDOM from "react-dom/client";
import { Landing } from "../src";
import rules from "./rules.sample.json";
import "../../identity/tokens/tokens.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><Landing dashboardHref="/app" rules={rules} /></React.StrictMode>,
);
