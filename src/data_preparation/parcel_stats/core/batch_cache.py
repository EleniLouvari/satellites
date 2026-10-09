"""Durable tabular batch results allow completed raster files to be removed."""

import hashlib
import json
from pathlib import Path

import pandas as pd


class BatchResultCache:
    """Commit statistics and their audit report before releasing raster storage."""

    def _statistics_signature(self):
        """Fingerprint processing options without including cleanup preferences."""
        configuration = {
            "format": 1,
            "run": self._run_signature(),
            "cleaning": self._cleaning_checkpoint_signature(),
            "statistics": self.spatial_statistics,
        }
        return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]

    def _batch_statistics_paths(self, netcdf_path):
        """Return signature-specific statistics and report paths beside a cube."""
        source = Path(netcdf_path)
        stem = f"{source.stem}_stats_{self._statistics_signature()}"
        return source.with_name(f"{stem}.parquet"), source.with_name(f"{stem}_report.parquet")

    def _has_batch_statistics(self, netcdf_path):
        """Check that both committed Parquet files exist for this configuration.

        Partial files are excluded. This checks presence only; reading the
        checkpoints performs Parquet validation.
        """
        return all(path.is_file() for path in self._batch_statistics_paths(netcdf_path))

    def _batch_available(self, netcdf_path):
        """Return whether a raw cube or saved statistics make acquisition unnecessary."""
        # Either artifact can satisfy downstream processing, depending on cleanup mode.
        return Path(netcdf_path).is_file() or self._has_batch_statistics(netcdf_path)

    def _save_batch_statistics(self, netcdf_path, result, report):
        """Persist the report, then commit statistics as the completion marker.

        Each DataFrame is written to a temporary Parquet file before replacing
        its destination. Write failures propagate without removing any rasters.
        """
        result_path, report_path = self._batch_statistics_paths(netcdf_path)
        # The result file commits last and marks the pair as complete. A crash
        # during either write leaves the NetCDF intact for the next attempt.
        for path, data in ((report_path, report), (result_path, result)):
            temporary = path.with_suffix(".partial.parquet")
            data.to_parquet(temporary, index=False)
            temporary.replace(path)

    def _remove_completed_batch_netcdfs(self, netcdf_path):
        """Remove this batch's known raster files when cleanup is enabled.

        Call only after its statistics and report are safely persisted. Raw,
        cleaned, final and partial NetCDFs are eligible; tabular checkpoints
        and unrelated files are retained. Missing files are harmless and other
        deletion errors are logged so saved statistics remain usable.
        """
        if not getattr(self, "remove_nc_after_completion", False):
            return
        source = Path(netcdf_path)
        cleaned, _ = self._cleaning_checkpoint_paths(source)
        final = self._final_checkpoint_path(source)
        # Enumerate only this batch's known raster paths, never the directory.
        # Full cleanup intentionally overrides keep_cleaned_checkpoint.
        for path in (source, cleaned, final):
            for candidate in (path, path.with_suffix(".partial.nc")):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError as error:
                    self.logger.warning("Could not remove completed raster %s: %s", candidate, error)

    def _cached_batch_statistics(self, netcdf_path):
        """Read and return the saved statistics and cleaning-report DataFrames."""
        result_path, report_path = self._batch_statistics_paths(netcdf_path)
        return pd.read_parquet(result_path), pd.read_parquet(report_path)
