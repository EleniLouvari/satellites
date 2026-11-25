/** Open the HUB acquisition guide without losing the current wizard draft. */
function HubGuideLink() {
  return (
    <a href="/guide/hub.html" target="_blank" rel="noopener">
      Read the HUB acquisition guide ↗
    </a>
  );
}

/** Show whether the server has HUB catalog access; credentials never reach the browser. */
export default function HubStatus({ configured }: { configured?: boolean }) {
  if (configured === undefined)
    return <p className="note">Checking HUB catalog access…</p>;
  if (!configured)
    return (
      <div className="notice">
        HUB catalog access is not configured on this server. Ask the operator
        to set the HUB catalog settings for the worker, or choose Copernicus
        openEO. <HubGuideLink />
      </div>
    );
  return (
    <p className="note">
      HUB catalog access is configured on the server. Analysis-ready Sentinel
      imagery is read directly; no Copernicus accounts are used.{" "}
      <HubGuideLink />
    </p>
  );
}
