"""Shared result assembly and partition-completion checkpoints."""
from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

from data_preparation.features.temporal import reduce_annual_median_features
from shared.io import write_data


class ParcelStatsResults:
    """Finalize acquired parcel batches independently of their source service."""

    def _partition_completion_path(self):
        """Return the manifest path recording partition status and configuration."""
        return self.output_dir / "satellite_partition_completion.json"

    def _partition_completion_record(self, status):
        """Build completion metadata; source adapters may add inventory provenance."""
        return {"status": status, "signature": self._statistics_signature()}

    def _write_partition_completion(self, status):
        """Atomically record the processing signature and supplied partition status.

        Call with ``processing`` before assembling outputs and ``complete``
        only after both GeoParquets and the cleaning CSV have been saved.
        """
        path = self._partition_completion_path()
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(self._partition_completion_record(status)), encoding="utf-8")
        temporary.replace(path)

    def _load_completed_partition(self):
        """Return reusable partition results, or ``None`` when processing is needed.

        Require explicit resume opt-in, all final files, matching parcel IDs,
        and a matching completion manifest when present. Legacy outputs instead
        require the matching raw-cache directory; the caller must preserve the
        original local processing options, which legacy outputs cannot verify.
        Successful reuse also restores ``self.cleaning_report`` from CSV.
        """
        if not self.resume_completed_partitions:
            return None
        outputs = [
            self.output_dir / self.PARCEL_OUTPUT_FILE_NAME,
            self.output_dir / self.REDUCED_PARCEL_OUTPUT_FILE_NAME,
            self.output_dir / "satellite_pixel_cleaning_report.csv",
        ]
        if not all(path.is_file() for path in outputs):
            return None
        # Marker validation guards against reusing outputs produced with different settings.
        marker = self._partition_completion_path()
        if marker.exists():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved != self._partition_completion_record("complete"):
                return None
        elif not (self.output_dir / "monthly_cubes" / self._run_signature()).is_dir():
            return None
        final = gpd.read_parquet(outputs[0])
        reduced = gpd.read_parquet(outputs[1])
        expected = set(self.parcels[self.PARCEL_ID_FIELD].astype(str))
        for frame in (final, reduced):
            if self.PARCEL_ID_FIELD not in frame or len(frame) != len(expected):
                return None
            if set(frame[self.PARCEL_ID_FIELD].astype(str)) != expected:
                return None
        self.cleaning_report = pd.read_csv(outputs[2])
        self.logger.info("Reusing completed partition outputs in %s.", self.output_dir)
        return final

    def run_from_batches(self, batches) -> gpd.GeoDataFrame:
        """Calculate and save partition statistics from already acquired cubes."""
        self._write_partition_completion("processing")
        # Run local statistics for each batch in parallel, returning a list of DataFrames and cleaning reports.
        completed_results, cleaning_reports = self._run_batch_statistics(batches)
        # Concatenate the batch results into a single DataFrame, sort by parcel and period, and add derived statistics.
        final = pd.concat(completed_results, ignore_index=True)
        final = final.sort_values([self.PARCEL_ID_FIELD, "period_start"]).reset_index(drop=True)
        final = self._add_derived_stats(final)
        # Pivot all temporal features and attach parcel attributes and geometry once.
        final = self._reshape_time_series_for_ml(final)
        # Persist the final statistics and cleaning report in the output directory.
        parquet_output = self.output_dir / self.PARCEL_OUTPUT_FILE_NAME
        write_data(final, str(parquet_output))
        # Create a compact sibling dataset for ML. The period count is inferred
        # from this run, while all configured Sentinel-2 sources must share it.
        reduction_sources = (*self.sentinel2_bands, *self.sentinel2_indices, *self.sentinel1_bands, *self.sentinel1_indices)
        reduced = reduce_annual_median_features(final, temporal_sources=reduction_sources, expected_periods=None)
        reduced_output = self.output_dir / self.REDUCED_PARCEL_OUTPUT_FILE_NAME
        temporary_reduced_output = reduced_output.with_name(f".{reduced_output.stem}.tmp{reduced_output.suffix}")
        try:
            write_data(reduced, str(temporary_reduced_output))
            temporary_reduced_output.replace(reduced_output)
        finally:
            if temporary_reduced_output.exists():
                temporary_reduced_output.unlink()
        # Keep one partition-level report with every batch's pixel-cleaning diagnostics.
        self.cleaning_report = pd.concat(cleaning_reports, ignore_index=True)
        cleaning_output = self.output_dir / "satellite_pixel_cleaning_report.csv"
        write_data(self.cleaning_report, str(cleaning_output), plain_csv=True)
        self._write_partition_completion("complete")
        self.logger.info(
            "Created %s (%s columns), %s (%s columns), and %s (%s rows).",
            parquet_output,
            len(final.columns),
            reduced_output,
            len(reduced.columns),
            cleaning_output,
            len(final),
        )
        return final
