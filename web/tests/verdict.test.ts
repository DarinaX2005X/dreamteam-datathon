import { describe, expect, it } from "vitest";
function reviewerVerdict(
  decisions: Array<"confirmed" | "rejected" | "not_visible" | null>,
) {
  return decisions.some((v) => v !== "rejected") ? "violation" : "no_violation";
}
describe("вердикт с решениями", () => {
  it("остаётся осторожным для непросмотренных и невидимых эпизодов", () => {
    expect(reviewerVerdict([null])).toBe("violation");
    expect(reviewerVerdict(["not_visible"])).toBe("violation");
  });
  it("становится чистым только когда все эпизоды отклонены", () => {
    expect(reviewerVerdict(["rejected", "rejected"])).toBe("no_violation");
  });
});
