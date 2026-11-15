import { useEffect, useRef, useState, type ReactNode } from "react";
import openeoFlow from "./images/openeo_cleaning_flowchart.png";
import edaFlow from "./images/eda_pipeline_flowchart.png";
import featureFlow from "./images/feature_selection_workflow.png";
import mlFlow from "./images/ml_full_workflow.png";
import oofFlow from "./images/ml_oof_reliability_workflow.png";

export type HelpTopic = "Parcel statistics" | "EDA" | "ML classification";
const TOPICS: HelpTopic[] = ["Parcel statistics", "EDA", "ML classification"];

/** Workflow image that opens at full resolution in a new tab. */
function Figure({ src, caption }: { src: string; caption: string }) {
  return (
    <figure className="help-figure">
      <a href={src} target="_blank" rel="noopener" title="Open full size">
        <img src={src} alt={caption} loading="lazy" />
      </a>
      <figcaption>{caption} · click to open full size</figcaption>
    </figure>
  );
}

function ParcelStatsHelp() {
  return (
    <>
      <p>
        Downloads Sentinel-1/2 imagery from Copernicus through openEO, cleans and
        fills the bands, calculates indices and writes one row per parcel with
        dated statistics columns (for example <code>NDVI_median__20240701</code>).
      </p>
      <ol>
        <li>Add one or more Copernicus accounts under <b>Copernicus accounts</b>.</li>
        <li>
          <b>New analysis</b>: upload a GeoPackage, GeoParquet or GeoJSON, then
          choose the layer, the unique parcel identifier and optionally a
          partition column.
        </li>
        <li>
          Select accounts and configure the run: Sentinel imagery, outlier
          removal (IQR), gap filling (temporal, 3×3, 5×5), interpolation, tiles
          and storage. Disabled options are not submitted.
        </li>
        <li>Save the draft, validate it, then submit the validated revision.</li>
        <li>
          Follow progress in <b>My runs</b>. Remote jobs at 100% do not mean the
          run is finished: download, local processing and publication follow. A
          run is complete only after its results are published.
        </li>
        <li>
          Download results, cleaning reports and logs, or <b>Download all files
          (.zip)</b>. A completed run can then be explored with <b>EDA</b>.
        </li>
      </ol>
      <Figure
        src={openeoFlow}
        caption="openEO acquisition, cleaning and final GeoParquet"
      />
    </>
  );
}

function EdaHelp() {
  return (
    <>
      <p>
        Profiles the parcel statistics of a completed run and produces an HTML
        report, diagnostic tables, plots, annotated data and an initial feature
        proposal.
      </p>
      <ol>
        <li>Open <b>EDA</b> from a completed run, then <b>New EDA analysis</b>.</li>
        <li>
          Choose the columns to analyse. Set the target and ID roles; use
          <code>target_task=classification</code> for numeric class codes.
          Include the geometry column for geospatial summaries.
        </li>
        <li>
          Review the grouped settings and submit. The analysis keeps running when
          the browser is closed.
        </li>
        <li>
          Open the report, or download the tables, plots and ZIP from the
          analysis details.
        </li>
      </ol>
      <p className="help-note">
        Feature proposals are exploratory diagnostics. Final feature selection
        requires train-only validation, which the ML pipeline performs.
      </p>
      <Figure src={edaFlow} caption="EDA pipeline: input data to diagnostics and reports" />
      <Figure src={featureFlow} caption="Feature screening workflow" />
    </>
  );
}

function MlHelp() {
  return (
    <>
      <p>
        Trains and compares classifiers on the parcel results, then predicts a
        class, probabilities and an inspection priority for every parcel. Each
        completed EDA analysis can hold several <i>scenarios</i>: one combination
        of features, models and settings.
      </p>
      <ol>
        <li>Open <b>ML classification</b> from a completed EDA analysis, then <b>New scenario</b>.</li>
        <li>
          Name the scenario and choose the features. EDA-proposed columns are
          pre-selected.
        </li>
        <li>
          Review the settings. Each option has a short explanation;
          hyperparameter grids are edited per model as JSON. Settings can be
          exported and imported.
        </li>
        <li>Run all five steps, or one step at a time.</li>
      </ol>
      <ul className="help-steps">
        <li><b>Check</b> validates the input and target.</li>
        <li><b>Prepare</b> builds the train/test split (optionally spatial) and CV folds.</li>
        <li><b>Train</b> tunes the selected models.</li>
        <li><b>Evaluate</b> scores models on the test set and chooses the strategy.</li>
        <li><b>Predict</b> refits and predicts all rows, including unlabelled ones.</li>
      </ul>
      <p className="help-note">
        After a failure, <b>Retry step</b> continues from the step that stopped and
        reuses trained models. Settings cannot be changed afterwards: use
        <b> Use as a new scenario</b> instead.
      </p>
      <Figure src={mlFlow} caption="Full classification and inspection workflow" />
      <Figure src={oofFlow} caption="Out-of-fold and class reliability workflow" />
    </>
  );
}

const CONTENT: Record<HelpTopic, () => ReactNode> = {
  "Parcel statistics": ParcelStatsHelp,
  EDA: EdaHelp,
  "ML classification": MlHelp,
};

/** Help button that opens pipeline guidance over the current page. */
export default function Help({ initial }: { initial: HelpTopic }) {
  const [open, setOpen] = useState(false);
  const [topic, setTopic] = useState<HelpTopic>(initial);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open) dialog.current?.showModal();
  }, [open]);
  const Content = CONTENT[topic];
  return (
    <>
      <button
        className="btn help-button"
        onClick={() => {
          setTopic(initial);
          setOpen(true);
        }}
      >
        ? Help
      </button>
      {open && (
        <dialog
          ref={dialog}
          className="confirm-dialog help-dialog"
          aria-labelledby="help-title"
          onCancel={() => setOpen(false)}
        >
          <div className="help-head">
            <h2 id="help-title">Help</h2>
            <button className="btn" autoFocus onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          <div className="tabs" role="tablist">
            {TOPICS.map((t) => (
              <button
                key={t}
                role="tab"
                aria-selected={topic === t}
                className={topic === t ? "active" : ""}
                onClick={() => setTopic(t)}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="help-body">
            <Content />
          </div>
        </dialog>
      )}
    </>
  );
}
