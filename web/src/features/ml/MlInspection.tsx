import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../../api";
import { Button, Notice } from "../../components/ui";
import ParcelMapDialog from "./ParcelMapDialog";

type Review = {
  check_status: CheckStatus;
  approved: boolean;
  comment: string;
  updated_at: string | null;
};
type CheckStatus = "not_checked" | "satellite" | "field" | "satellite_and_field";
type Page = {
  columns: string[];
  id_column: string;
  total: number;
  offset: number;
  checked: number;
  approved: number;
  rows: { values: string[]; level: number | null; review: Review }[];
};

const CHECK_STATUSES: { value: CheckStatus; label: string }[] = [
  { value: "not_checked", label: "Not checked" },
  { value: "satellite", label: "Satellite image" },
  { value: "field", label: "Field visit" },
  { value: "satellite_and_field", label: "Satellite image and field visit" },
];
const PAGE_SIZE = 50;
// Room kept below the table for the page buttons, so the table and its scroll bars fit the window.
const BELOW_TABLE = 72;

/** Green (no inspection needed, level 0) to red (most urgent, level 1); grey when the need is unknown. */
export function needColor(level: number | null): string {
  return level == null ? "#f1f2ee" : `hsl(${Math.round(120 * (1 - level))}, 70%, 88%)`;
}

/** Shorten long decimals for reading; every other value is shown as saved. */
function display(value: string): string {
  return /^-?\d+\.\d{5,}$/.test(value) ? Number(value).toFixed(4) : value;
}

export default function MlInspection({
  mlId,
  name,
  onBack,
}: {
  mlId: string;
  name: string;
  onBack: () => void;
}) {
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Page>();
  const [comments, setComments] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const [mapParcel, setMapParcel] = useState<string>();
  // The row last opened on the map stays highlighted after the map closes.
  const [selectedParcel, setSelectedParcel] = useState<string>();
  const tableRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>();

  // Size the table from where it starts on screen down to the bottom of the window.
  useLayoutEffect(() => {
    const fit = () => {
      const top = tableRef.current?.getBoundingClientRect().top;
      if (top != null)
        setHeight(Math.max(240, window.innerHeight - top - BELOW_TABLE));
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [page !== undefined, error]);

  useEffect(() => {
    let live = true;
    setError("");
    api<Page>(`/ml/${mlId}/inspection?offset=${offset}&limit=${PAGE_SIZE}`)
      .then((value) => {
        if (!live) return;
        setPage(value);
        setComments({});
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [mlId, offset]);

  async function save(parcelId: string, change: Partial<Review>) {
    if (!page) return;
    const row = page.rows.find((r) => r.values[idIndex] === parcelId);
    if (!row) return;
    const previous = row.review;
    const next = { ...previous, ...change };
    // An unchecked parcel cannot stay approved.
    if (next.check_status === "not_checked") next.approved = false;
    const update = (review: Review, counts?: { checked: number; approved: number }) =>
      setPage((current) =>
        current && {
          ...current,
          ...(counts ?? {}),
          rows: current.rows.map((r) =>
            r.values[idIndex] === parcelId ? { ...r, review } : r,
          ),
        },
      );
    const delta = (review: Review) => ({
      checked: Number(review.check_status !== "not_checked"),
      approved: Number(review.approved),
    });
    const counts = (from: Review, to: Review) => ({
      checked: page.checked - delta(from).checked + delta(to).checked,
      approved: page.approved - delta(from).approved + delta(to).approved,
    });
    setError("");
    update(next, counts(previous, next));
    try {
      const saved = await api<Review>(`/ml/${mlId}/inspection/review`, "PUT", {
        parcel_id: parcelId,
        check_status: next.check_status,
        approved: next.approved,
        comment: next.comment,
      });
      update(saved);
    } catch (e) {
      update(previous, counts(next, previous));
      setError((e as Error).message);
    }
  }

  const idIndex = page ? page.columns.indexOf(page.id_column) : -1;
  const others = page
    ? page.columns.map((_, i) => i).filter((i) => i !== idIndex)
    : [];

  return (
    <section className="ml-inspection">
      <div className="heading">
        <div>
          <h3>Parcel inspection · {name}</h3>
          <p className="footnote">
            Every parcel with all its predictions, the ones that most need an
            inspection first (inspection need, then inspection score). Your
            choices are saved as soon as you change them; comments are saved
            when you leave the box.
          </p>
        </div>
        <div className="actions">
          <Button onClick={onBack}>Back to scenario</Button>
        </div>
      </div>
      <Notice message={error} />
      {!page ? (
        !error && <p>Loading parcels…</p>
      ) : (
        <>
          <div className="ml-inspection-bar">
            <span>
              <strong>{page.checked}</strong> of {page.total} parcels checked ·{" "}
              <strong>{page.approved}</strong> approved
            </span>
            <span className="ml-inspection-legend" aria-label="Row colours">
              <small>No inspection needed</small>
              <i />
              <small>Inspection needed</small>
            </span>
          </div>
          <div
            ref={tableRef}
            className="tablewrap panel ml-inspection-table"
            style={height ? { maxHeight: height } : undefined}
          >
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>{page.id_column}</th>
                  <th>Map</th>
                  <th>Checked</th>
                  <th>Approved</th>
                  <th>Comments</th>
                  {others.map((i) => (
                    <th key={page.columns[i]}>
                      {page.columns[i].replaceAll("_", " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {page.rows.map((row, index) => {
                  const parcelId = row.values[idIndex];
                  const comment = comments[parcelId] ?? row.review.comment;
                  const checked = row.review.check_status !== "not_checked";
                  return (
                    <tr
                      key={parcelId}
                      className={parcelId === selectedParcel ? "is-selected" : undefined}
                      aria-selected={parcelId === selectedParcel}
                      style={{ background: needColor(row.level) }}
                    >
                      <td>{page.offset + index + 1}</td>
                      <td>
                        <strong>{parcelId}</strong>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn icon-btn"
                          title="Show parcel on map"
                          aria-label={`Show parcel ${parcelId} on map`}
                          onClick={() => {
                            setSelectedParcel(parcelId);
                            setMapParcel(parcelId);
                          }}
                        >
                          <svg
                            viewBox="0 0 24 24"
                            width="18"
                            height="18"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="M12 21.5s-7-6.1-7-11.8a7 7 0 0 1 14 0c0 5.7-7 11.8-7 11.8z" />
                            <circle cx="12" cy="9.7" r="2.6" />
                          </svg>
                        </button>
                      </td>
                      <td>
                        <select
                          aria-label={`How parcel ${parcelId} was checked`}
                          value={row.review.check_status}
                          onChange={(e) =>
                            void save(parcelId, {
                              check_status: e.target.value as CheckStatus,
                            })
                          }
                        >
                          {CHECK_STATUSES.map((status) => (
                            <option key={status.value} value={status.value}>
                              {status.label}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`Parcel ${parcelId} approved`}
                          title={
                            checked
                              ? "The parcel is as the farmer declared"
                              : "Check the parcel before approving it"
                          }
                          checked={row.review.approved}
                          disabled={!checked}
                          onChange={(e) =>
                            void save(parcelId, { approved: e.target.checked })
                          }
                        />
                      </td>
                      <td>
                        <textarea
                          aria-label={`Comments on parcel ${parcelId}`}
                          className="ml-inspection-comment"
                          rows={1}
                          maxLength={2000}
                          value={comment}
                          onChange={(e) =>
                            setComments((c) => ({ ...c, [parcelId]: e.target.value }))
                          }
                          onBlur={() => {
                            if (comment !== row.review.comment)
                              void save(parcelId, { comment });
                            setComments((c) => {
                              const rest = { ...c };
                              delete rest[parcelId];
                              return rest;
                            });
                          }}
                        />
                      </td>
                      {others.map((i) => (
                        <td key={page.columns[i]}>{display(row.values[i])}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="actions sectiongap">
            <Button disabled={offset === 0} onClick={() => setOffset(0)}>
              First page
            </Button>
            <Button
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              Previous
            </Button>
            <span className="muted">
              Parcels {page.total ? offset + 1 : 0}–
              {Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
            </span>
            <Button
              disabled={offset + PAGE_SIZE >= page.total}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Next
            </Button>
            <Button
              disabled={offset + PAGE_SIZE >= page.total}
              onClick={() =>
                setOffset(Math.max(0, Math.floor((page.total - 1) / PAGE_SIZE) * PAGE_SIZE))
              }
            >
              Last page
            </Button>
          </div>
        </>
      )}
      {mapParcel !== undefined && (
        <ParcelMapDialog
          mlId={mlId}
          parcelId={mapParcel}
          onClose={() => setMapParcel(undefined)}
        />
      )}
    </section>
  );
}
