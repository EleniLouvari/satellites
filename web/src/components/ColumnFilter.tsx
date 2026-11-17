import { useId, useMemo, useState } from "react";
import { Button } from "./ui";
import {
  keywordOptions,
  matchesKeywords,
} from "../features/eda/edaControls";

/** Search text plus whole-token keyword chips, shared by the EDA and ML column pickers. */
export function useColumnFilter(names: string[]) {
  const [search, setSearch] = useState("");
  const [keywords, setKeywords] = useState<string[]>([]);
  const [mode, setMode] = useState<"any" | "all">("any");
  const choices = useMemo(() => keywordOptions(names), [names]);
  const term = search.trim().toLowerCase();
  return {
    search,
    setSearch,
    keywords,
    mode,
    setMode,
    choices,
    filtering: keywords.length > 0 || term !== "",
    toggle(keyword: string) {
      setKeywords((previous) =>
        previous.includes(keyword)
          ? previous.filter((k) => k !== keyword)
          : [...previous, keyword],
      );
    },
    clear() {
      setKeywords([]);
      setSearch("");
    },
    matches: (name: string) =>
      name.toLowerCase().includes(term) &&
      matchesKeywords(name, keywords, mode),
  };
}

export type ColumnFilterState = ReturnType<typeof useColumnFilter>;

/**
 * Search box, keyword chips and the filtered-selection actions. ``visible`` lists the
 * matching columns the actions may change; the caller re-adds or drops locked columns.
 */
export default function ColumnFilter({
  filter,
  visible,
  selected,
  onChoose,
}: {
  filter: ColumnFilterState;
  visible: string[];
  selected: string[];
  onChoose: (columns: string[]) => void;
}) {
  const radio = useId();
  return (
    <>
      <input
        aria-label="Search columns"
        placeholder="Search columns…"
        value={filter.search}
        onChange={(e) => filter.setSearch(e.target.value)}
      />
      {filter.choices.length > 0 && (
        <div className="eda-keywords">
          <div className="eda-keywords-head">
            <span>Filter by keyword</span>
            <label>
              <input
                type="radio"
                name={radio}
                checked={filter.mode === "any"}
                onChange={() => filter.setMode("any")}
              />
              any keyword
            </label>
            <label>
              <input
                type="radio"
                name={radio}
                checked={filter.mode === "all"}
                onChange={() => filter.setMode("all")}
              />
              all keywords
            </label>
          </div>
          <div className="eda-keyword-chips">
            {filter.choices.map(({ keyword, count }) => (
              <button
                type="button"
                key={keyword}
                className={
                  filter.keywords.includes(keyword) ? "chip active" : "chip"
                }
                aria-pressed={filter.keywords.includes(keyword)}
                onClick={() => filter.toggle(keyword)}
              >
                {keyword} <small>{count}</small>
              </button>
            ))}
          </div>
        </div>
      )}
      {filter.filtering && (
        <div className="eda-filter-actions">
          <span>
            {visible.length} column{visible.length === 1 ? "" : "s"} match
          </span>
          <Button
            disabled={!visible.length}
            primary
            onClick={() => onChoose(visible)}
          >
            Select only filtered
          </Button>
          <Button
            disabled={!visible.length}
            onClick={() => onChoose([...selected, ...visible])}
          >
            Add filtered
          </Button>
          <Button
            disabled={!visible.length}
            onClick={() => onChoose(selected.filter((n) => !visible.includes(n)))}
          >
            Remove filtered
          </Button>
          <Button onClick={filter.clear}>Clear filter</Button>
        </div>
      )}
    </>
  );
}
