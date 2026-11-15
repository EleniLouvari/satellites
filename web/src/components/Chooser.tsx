import { Status } from "./ui";

export type ChooserItem = {
  id: string;
  title: string;
  meta: string;
  status: string;
};

/** Landing view: start something new, or open one of the saved items. */
export default function Chooser({
  newLabel,
  newHint,
  onNew,
  historyLabel,
  items,
  onOpen,
  emptyText,
}: {
  newLabel: string;
  newHint: string;
  onNew: () => void;
  historyLabel: string;
  items: ChooserItem[];
  onOpen: (id: string) => void;
  emptyText: string;
}) {
  return (
    <div className="chooser">
      <button type="button" className="chooser-new" onClick={onNew}>
        <b aria-hidden="true">+</b>
        <strong>{newLabel}</strong>
        <small>{newHint}</small>
      </button>
      <section className="chooser-history" aria-label={historyLabel}>
        <h3>
          {historyLabel} <span className="badge">{items.length}</span>
        </h3>
        {!items.length && <p className="muted">{emptyText}</p>}
        <ul className="chooser-list">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                className="chooser-row"
                onClick={() => onOpen(item.id)}
              >
                <span>
                  <strong>{item.title}</strong>
                  <small>{item.meta}</small>
                </span>
                <Status value={item.status} />
                <i aria-hidden="true">→</i>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

/** Link back to the landing view from a setup, detail or report view. */
export function BackLink({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <button type="button" className="linkbutton view-back" onClick={onClick}>
      ← {label}
    </button>
  );
}
