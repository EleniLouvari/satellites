export type HelpTopic = "openEO" | "Parcel statistics" | "EDA" | "ML classification" | "Inspection";

const GUIDE_PAGES: Record<HelpTopic, string> = {
  openEO: "openeo",
  "Parcel statistics": "statistics",
  EDA: "eda",
  "ML classification": "ml",
  Inspection: "inspection",
};

/** Open contextual pipeline guidance without losing the current form or run view. */
export default function Help({ initial }: { initial: HelpTopic }) {
  return (
    <a
      className="btn help-button"
      href={`/guide/${GUIDE_PAGES[initial]}.html`}
      target="_blank"
      rel="noopener"
      aria-label={`Help: ${initial} (opens in a new tab)`}
    >
      ? Help
    </a>
  );
}
