import { describe, expect, it } from "vitest";
import { formatDuration, formatTimecode } from "../src/i18n/ru";
describe("формат времени", () => {
  it("показывает десятые доли и минуты", () => {
    expect(formatTimecode(44.24)).toBe("00:44,2");
    expect(formatTimecode(125.04)).toBe("02:05,0");
  });
  it("форматирует длительность без выдуманной точности", () => {
    expect(formatDuration(125)).toBe("2 мин 5 с");
    expect(formatDuration(null)).toBe("—");
  });
});
