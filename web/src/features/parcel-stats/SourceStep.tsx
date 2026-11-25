import type { Account } from "../../api";
import { Status } from "../../components/ui";
import HubStatus from "../sources/hub/HubStatus";
import { SOURCES, type Source } from "./sourceControls";

/** Choose the imagery source, then Copernicus accounts (openEO) or confirm HUB access. */
export default function SourceStep({
  source,
  onSource,
  accounts,
  selected,
  onSelected,
  hubConfigured,
}: {
  source: Source;
  onSource: (source: Source) => void;
  accounts: Account[];
  selected: number[];
  onSelected: (ids: number[]) => void;
  hubConfigured?: boolean;
}) {
  return (
    <>
      <span className="eyebrow">02 · Connect to the imagery</span>
      <h2>Choose the imagery source</h2>
      <div className="sectiongap" role="radiogroup" aria-label="Imagery source">
        {SOURCES.map((item) => (
          <label className="account" key={item.id}>
            <input
              type="radio"
              name="imagery-source"
              checked={source === item.id}
              onChange={() => onSource(item.id)}
            />
            <span className="grow">
              {item.label}
              <small>{item.description}</small>
            </span>
          </label>
        ))}
      </div>
      {source === "hub" ? (
        <HubStatus configured={hubConfigured} />
      ) : (
        <AccountChoice accounts={accounts} selected={selected} onSelected={onSelected} />
      )}
    </>
  );
}

/** Ordered Copernicus account selection for openEO analyses. */
function AccountChoice({
  accounts,
  selected,
  onSelected,
}: {
  accounts: Account[];
  selected: number[];
  onSelected: (ids: number[]) => void;
}) {
  return (
    <>
      <h3 className="sectiongap">Copernicus accounts</h3>
      <p className="muted">Selection order is preserved when your analysis is submitted.</p>
      {!accounts.length && (
        <p className="note">Add an account from Copernicus accounts, then return to this draft.</p>
      )}
      {accounts.map((a) => (
        <label className="account" key={a.eo_user_id}>
          <input
            type="checkbox"
            checked={selected.includes(a.eo_user_id)}
            onChange={(e) =>
              onSelected(
                e.target.checked ? [...selected, a.eo_user_id] : selected.filter((id) => id !== a.eo_user_id),
              )
            }
          />
          <span className="grow">
            {a.label || a.username}
            <small>{a.username}</small>
          </span>
          <Status value={a.test_status} />
          {selected.includes(a.eo_user_id) && <small>#{selected.indexOf(a.eo_user_id) + 1}</small>}
        </label>
      ))}
    </>
  );
}
