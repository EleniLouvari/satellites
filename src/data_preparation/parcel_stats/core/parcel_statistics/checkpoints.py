"""Checkpoint, local-index, and cleaned-cube persistence helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shared.io import read_data, write_data


class ParcelStatisticsCheckpointMixin:
    """Methods for checkpoint signatures and checkpoint persistence."""

    def _cleaning_checkpoint_signature(self) -> str:
        """Fingerprint local operations without invalidating reusable raw downloads."""
        configuration = {
            "checkpoint_format": "cleaned_plus_fill_provenance_v3",
            "remove_outliers": self.remove_outliers,
            "iqr_quantiles": self.iqr_quantiles,
            "iqr_multiplier": self.iqr_multiplier,
            "iqr_min_valid_pixels": self.iqr_min_valid_pixels,
            "fill_nulls": getattr(self, "fill_nulls", False),
            "temporal_fill_mode": self.temporal_fill_mode,
            "minimum_observed_fraction_for_fill": self.minimum_observed_fraction_for_fill,
            "spatial_fill_window_sizes": getattr(self, "spatial_fill_window_sizes", (3, 5)),
            "interpolation_method": self.interpolation_method,
            "interpolation_max_distance_in_meters": self.interpolation_max_distance_in_meters,
            "interpolation_variogram_lags": self.interpolation_variogram_lags,
            "interpolation_variogram_max_distance_in_meters": self.interpolation_variogram_max_distance_in_meters,
        }
        temporal_windows = getattr(self, "temporal_fill_window_sizes", None)
        if temporal_windows is not None:
            configuration["temporal_fill_window_sizes"] = temporal_windows
        payload = json.dumps(configuration, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _safe_ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Divide arrays while representing zero or invalid denominators as nulls."""
        result = np.full(numerator.shape, np.nan, dtype="float32")
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    def _calculate_local_optical_index(self, values_by_band, index_name):
        """Translate source band identifiers to spectral roles before shared algebra."""
        values_by_role = {role: values_by_band[name] for role, name in self.OPTICAL_BAND_ROLES.items() if name in values_by_band}
        return self._calculate_optical_index(values_by_role, index_name)

    def _calculate_optical_index(self, values_by_role: dict[str, np.ndarray], index_name: str) -> np.ndarray:
        """Calculate an optical index from cleaned reflectance arrays keyed by spectral role."""
        band = values_by_role.__getitem__
        index_calculators = {
            "NDVI": lambda: self._safe_ratio(band("nir") - band("red"), band("nir") + band("red")),
            "NDWI": lambda: self._safe_ratio(band("green") - band("nir"), band("green") + band("nir")),
            "MNDWI": lambda: self._safe_ratio(band("green") - band("swir_1"), band("green") + band("swir_1")),
            "NDMI": lambda: self._safe_ratio(band("nir") - band("swir_1"), band("nir") + band("swir_1")),
            "NBR": lambda: self._safe_ratio(band("nir") - band("swir_2"), band("nir") + band("swir_2")),
            "GNDVI": lambda: self._safe_ratio(band("nir") - band("green"), band("nir") + band("green")),
            "EVI": lambda: self._safe_ratio(
                2.5 * (band("nir") - band("red")),
                band("nir") + 6.0 * band("red") - 7.5 * band("blue") + 1.0,
            ),
            "SAVI": lambda: self._safe_ratio(1.5 * (band("nir") - band("red")), band("nir") + band("red") + 0.5),
            "MSAVI": lambda: self._calculate_msavi(band("nir"), band("red")),
            "NDRE": lambda: self._safe_ratio(band("nir") - band("red_edge"), band("nir") + band("red_edge")),
            "PSRI": lambda: self._safe_ratio(band("red") - band("blue"), band("red_edge_2")),
            "CI": lambda: self._calculate_ci(band("nir"), band("red_edge")),
        }
        calculator = index_calculators.get(index_name)
        if calculator is None:
            raise ValueError(f"Error: Unsupported index: {index_name}")
        return calculator()

    def _calculate_msavi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """Calculate MSAVI with NaN output for invalid discriminant values."""
        doubled_nir_plus_one = 2.0 * nir + 1.0
        discriminant = doubled_nir_plus_one**2 - 8.0 * (nir - red)
        result = np.full(discriminant.shape, np.nan, dtype="float32")
        valid = np.isfinite(discriminant) & (discriminant >= 0)
        result[valid] = (doubled_nir_plus_one[valid] - np.sqrt(discriminant[valid])) / 2.0
        return result

    def _calculate_ci(self, nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """Calculate chlorophyll index from NIR and red-edge bands."""
        ratio = self._safe_ratio(nir, red_edge)
        return np.where(np.isfinite(ratio), ratio - 1.0, np.nan).astype("float32")

    def _calculate_local_sentinel1_index(self, values_by_band: dict[str, np.ndarray], index_name: str) -> np.ndarray:
        """Calculate one Sentinel-1 index per pixel from linear-power sigma0."""
        vv = values_by_band["VV"]
        vh = values_by_band["VH"]
        finite_negative = (np.isfinite(vv) & (vv < 0)) | (np.isfinite(vh) & (vh < 0))
        if np.any(finite_negative):
            raise ValueError(
                "Error: Sentinel-1 R and RVI require non-negative linear-power sigma0 values; negative values suggest dB-scaled or physically invalid input."
            )
        if index_name == "R":
            return self._safe_ratio(vv, vh)
        if index_name == "RVI":
            return self._safe_ratio(4.0 * vh, vv + vh)
        raise ValueError(f"Error: Unsupported Sentinel-1 index: {index_name}")

    def _build_local_output_cube(self, values: np.ndarray, cube_variables: list[str]) -> np.ndarray:
        """Select requested bands and append indices derived from cleaned source bands."""
        values_by_band = {name: values[:, index] for index, name in enumerate(cube_variables)}
        output_layers = [values_by_band[name] for name in self.optical_bands]
        output_layers.extend(self._calculate_local_optical_index(values_by_band, name) for name in self.optical_indices)
        output_layers.extend(values_by_band[name] for name in self.sentinel1_bands)
        output_layers.extend(self._calculate_local_sentinel1_index(values_by_band, name) for name in self.sentinel1_indices)
        return np.stack(output_layers, axis=1).astype("float32", copy=False)

    def _cleaning_checkpoint_paths(self, netcdf_path) -> tuple[Path, Path]:
        """Return deterministic cleaned-raster and audit paths for one raw cube."""
        source = Path(netcdf_path)
        base_stem = source.stem
        return (source.with_name(f"{base_stem}_cleaned.nc"), source.with_name(f"{base_stem}_cleaning_report.parquet"))

    def _final_checkpoint_path(self, netcdf_path) -> Path:
        """Return the final post-index NetCDF path for one raw cube."""
        source = Path(netcdf_path)
        base_stem = source.stem
        return source.with_name(f"{base_stem}_final.nc")

    def _load_final_checkpoint(
        self, netcdf_path, periods: list[str], variables: list[str], expected_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame] | None:
        """Load final bands and indices when the post-index checkpoint is complete."""
        final_path = self._final_checkpoint_path(netcdf_path)
        _, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        if not final_path.exists() or not report_path.exists():
            return None
        with xr.open_dataset(final_path, mask_and_scale=True) as checkpoint:
            if checkpoint.attrs.get("cleaning_signature") != self._cleaning_checkpoint_signature():
                self.parcel_logger.info(f"Ignoring stale post-index checkpoint {final_path}.")
                return None
            saved_periods = checkpoint["period"].astype(str).values.tolist()
            saved_variables = checkpoint["variable"].astype(str).values.tolist()
            cleaned = np.asarray(checkpoint["cleaned"].values, dtype="float32")
            observed_mask = np.asarray(checkpoint["observed_mask"].values, dtype=bool)
            temporal_filled_mask = np.asarray(checkpoint["temporal_filled_mask"].values, dtype=bool)
            observed = np.where(observed_mask, cleaned, np.nan).astype("float32", copy=False)
        if saved_periods != periods or saved_variables != variables:
            raise ValueError(f"Error: Final checkpoint metadata does not match the configured outputs: {final_path}")
        if cleaned.shape != expected_shape or observed.shape != expected_shape or temporal_filled_mask.shape != expected_shape:
            raise ValueError(f"Error: Final checkpoint shape does not match the raw cube: {final_path}")
        self.parcel_logger.info(f"Using completed post-index checkpoint {final_path}.")
        return cleaned, observed, temporal_filled_mask, read_data(str(report_path))

    def _save_final_checkpoint(
        self,
        netcdf_path,
        cleaned: np.ndarray,
        observed: np.ndarray,
        periods: list[str],
        variables: list[str],
        temporally_filled: np.ndarray | None = None,
    ) -> None:
        """Atomically save final requested bands and locally calculated indices."""
        final_path = self._final_checkpoint_path(netcdf_path)
        temporary_path = final_path.with_suffix(".partial.nc")
        dimensions = ("period", "variable", "y", "x")
        if temporally_filled is None:
            temporally_filled = observed
        temporal_filled_mask = ~np.isfinite(observed) & np.isfinite(temporally_filled)
        xr.Dataset(
            {
                "cleaned": (dimensions, cleaned.astype("float32", copy=False)),
                "observed_mask": (dimensions, np.isfinite(observed).astype("uint8")),
                "temporal_filled_mask": (dimensions, temporal_filled_mask.astype("uint8")),
            },
            coords={"period": periods, "variable": variables},
            attrs={"cleaning_signature": self._cleaning_checkpoint_signature()},
        ).to_netcdf(temporary_path)
        temporary_path.replace(final_path)
        self.parcel_logger.info(f"Saved completed post-index checkpoint {final_path}.")

    def _load_cleaning_checkpoint(
        self, netcdf_path, periods: list[str], variables: list[str], expected_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
        """Load a complete and compatible cleaning checkpoint when available."""
        cleaned_path, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        if not cleaned_path.exists() or not report_path.exists():
            return None

        with xr.open_dataset(cleaned_path, mask_and_scale=True) as checkpoint:
            if checkpoint.attrs.get("cleaning_signature") != self._cleaning_checkpoint_signature():
                self.parcel_logger.info(f"Ignoring stale cleaning checkpoint {cleaned_path}.")
                return None
            saved_periods = checkpoint["period"].astype(str).values.tolist()
            saved_variables = checkpoint["variable"].astype(str).values.tolist()
            cleaned = np.asarray(checkpoint["cleaned"].values, dtype="float32")
            observed_mask = np.asarray(checkpoint["observed_mask"].values, dtype=bool)
            observed = np.where(observed_mask, cleaned, np.nan).astype("float32", copy=False)

        if saved_periods != periods or saved_variables != variables:
            raise ValueError(f"Error: Cleaning checkpoint metadata does not match the raw cube: {cleaned_path}")
        if cleaned.shape != expected_shape or observed.shape != expected_shape:
            raise ValueError(f"Error: Cleaning checkpoint shape does not match the raw cube: {cleaned_path}")

        report = read_data(str(report_path))
        self.parcel_logger.info(f"Using completed cleaning checkpoint {cleaned_path}.")
        return cleaned, observed, report

    def _save_cleaning_checkpoint(
        self,
        netcdf_path,
        cleaned: np.ndarray,
        observed: np.ndarray,
        report: pd.DataFrame,
        periods: list[str],
        variables: list[str],
    ) -> None:
        """Atomically persist cleaned pixels, observation state, and their audit."""
        cleaned_path, report_path = self._cleaning_checkpoint_paths(netcdf_path)
        temporary_cleaned = cleaned_path.with_suffix(".partial.nc")
        temporary_report = report_path.with_suffix(".partial.parquet")
        dimensions = ("period", "variable", "y", "x")
        checkpoint = xr.Dataset(
            {
                "cleaned": (dimensions, cleaned.astype("float32", copy=False)),
                "observed_mask": (dimensions, np.isfinite(observed).astype("uint8")),
            },
            coords={"period": periods, "variable": variables},
            attrs={"cleaning_signature": self._cleaning_checkpoint_signature()},
        )
        checkpoint.to_netcdf(temporary_cleaned)
        write_data(report, str(temporary_report))
        temporary_report.replace(report_path)
        temporary_cleaned.replace(cleaned_path)
        self.parcel_logger.info(f"Saved completed cleaning checkpoint {cleaned_path}.")

    def _remove_cleaned_raster_checkpoint(self, netcdf_path) -> None:
        """Remove the redundant intermediate raster after final output commits."""
        cleaned_path, _ = self._cleaning_checkpoint_paths(netcdf_path)
        if cleaned_path.exists():
            cleaned_path.unlink()
            self.parcel_logger.info(f"Removed intermediate cleaning raster {cleaned_path}; raw data and audit remain available.")
