import { useEffect, useState } from "react";

import { downloadJson, sanitizeConfig, type Schema } from "./api";

import { Button, Notice } from "./ui";
import { activeArrayErrors, fieldDisabledReason } from "./configControls";

export default function ConfigForm({
  schema,
  value,
  onChange,
  onValidityChange,
}: {
  schema: Schema;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  onValidityChange?: (valid: boolean) => void;
}) {
  const [error, setError] = useState("");
  const [arrayDrafts, setArrayDrafts] = useState<Record<string, string>>({});
  const fields = schema.fields;
  const effective = { ...schema.defaults, ...value };
  const arrayErrorNames = activeArrayErrors(
    fields,
    arrayDrafts,
    effective,
  ).join(", ");
  useEffect(() => {
    onValidityChange?.(!arrayErrorNames);
  }, [arrayErrorNames, onValidityChange]);
  const groups = schema.groups ?? [
    {
      id: "basic",
      label: "Processing settings",
      description: "Configure your analysis.",
    },
  ];

  function field(f: Schema["fields"][number]) {
    const current = Object.prototype.hasOwnProperty.call(value, f.name)
      ? value[f.name]
      : f.default;

    const array = f.type === "array";
    const object = f.type === "object";
    const disabledReason = fieldDisabledReason(f, effective);

    // Re-enabled controls must submit the defaults displayed in the form too.
    const set = (v: unknown) => onChange({ ...effective, [f.name]: v });

    let control;

    if (array && f.choices) {
      const selected = Array.isArray(current) ? current : [];

      control = (
        <div className="chips">
          {f.choices.map((choice, i) => (
            <label className="chip" key={i}>
              <input
                type="checkbox"
                checked={selected.includes(choice)}
                onChange={(e) =>
                  set(
                    e.target.checked
                      ? [...selected, choice]
                      : selected.filter((v) => v !== choice),
                  )
                }
              />
              {String(choice)}
            </label>
          ))}
        </div>
      );
    } else if (f.choices) {
      control = (
        <select
          value={String(current ?? "")}
          onChange={(e) =>
            set(f.choices!.find((c) => String(c ?? "") === e.target.value))
          }
        >
          {f.choices.map((c, i) => (
            <option key={i} value={String(c ?? "")}>
              {c == null ? "None" : String(c)}
            </option>
          ))}
        </select>
      );
    } else if (f.type === "boolean") {
      control = (
        <select
          value={String(current)}
          onChange={(e) => set(e.target.value === "true")}
        >
          <option value="true">Enabled</option>
          <option value="false">Disabled</option>
        </select>
      );
    } else if (array || object) {
      control = (
        <textarea
          value={
            arrayDrafts[f.name] ??
            (current == null ? "" : JSON.stringify(current))
          }
          placeholder={object ? 'e.g. {"executor-memory":"2G"}' : "e.g. [3, 5, 7]"}
          rows={2}
          onChange={(e) => {
            setArrayDrafts({ ...arrayDrafts, [f.name]: e.target.value });
            try {
              const parsed = e.target.value.trim()
                ? JSON.parse(e.target.value)
                : null;
              if (parsed !== null && (object
                ? typeof parsed !== "object" || Array.isArray(parsed)
                : !Array.isArray(parsed))) throw new Error();
              set(parsed);
            } catch {
              // Keep unfinished text locally until it is valid JSON.
            }
          }}
        />
      );
    } else {
      control = (
        <input
          type={
            f.type === "number" ? "number" : f.type === "date" ? "date" : "text"
          }
          step="any"
          value={current == null ? "" : String(current)}
          onChange={(e) =>
            set(
              e.target.value === ""
                ? null
                : f.type === "number"
                  ? Number(e.target.value)
                  : e.target.value,
            )
          }
        />
      );
    }

    return (
      <div
        key={f.name}
        className={`config-field ${disabledReason ? "is-disabled" : ""}`}
      >
        <fieldset
          className="config-fieldset"
          disabled={!!disabledReason}
          aria-describedby={`help-${f.name}`}
        >
          <legend>{f.name.replaceAll("_", " ")}</legend>
          {control}
        </fieldset>
        <small className="fieldhelp" id={`help-${f.name}`}>
          {disabledReason ||
            f.description ||
            (array ? "Select values or enter a JSON array." : f.type)}
        </small>
      </div>
    );
  }

  return (
    <>
      <Notice message={error} />
      <Notice
        message={
          arrayErrorNames
            ? `Error: Enter valid JSON of the requested type for ${arrayErrorNames}.`
            : ""
        }
      />
      <div className="config-groups">
        {groups.map((group, index) => {
          const members = fields.filter(
            (f) => (f.group ?? "basic") === group.id,
          );
          if (!members.length) return null;
          const active = members.filter(
            (f) => !fieldDisabledReason(f, effective),
          ).length;
          return (
            <details key={group.id} className="config-group" open>
              <summary>
                <span className="config-group-number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="config-group-heading">
                  <strong>{group.label}</strong>
                  <small>{group.description}</small>
                </span>
                <span className="config-group-count">
                  {active} / {members.length} active
                </span>
              </summary>
              <div className="formgrid config-group-fields">
                {members.map(field)}
              </div>
              {group.id === "interpolation" &&
                effective.fill_nulls === true &&
                effective.interpolation_method === "nearest" && (
                  <p className="config-group-note">
                    Nearest uses the closest valid raster cell. No distance or
                    variogram settings are needed.
                  </p>
                )}
            </details>
          );
        })}
      </div>
      <div className="actions sectiongap">
        <Button
          onClick={() =>
            downloadJson(
              sanitizeConfig(effective, schema),
              "parcel-settings.json",
            )
          }
        >
          Export settings
        </Button>
        <label className="btn file-button">
          Import settings
          <input
            type="file"
            accept=".json,application/json"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              try {
                const raw = JSON.parse(await file.text());
                onChange({
                  ...schema.defaults,
                  ...sanitizeConfig(raw.processing ?? raw, schema),
                });
                setArrayDrafts({});
                onValidityChange?.(true);
                setError("");
              } catch (err) {
                setError((err as Error).message);
              }
              e.target.value = "";
            }}
          />
        </label>
      </div>
    </>
  );
}
