"""Distance and contiguity neighborhoods for spatial autocorrelation."""


import numpy as np
from libpysal.weights import DistanceBand, Queen
from libpysal.weights.util import fill_diagonal

import shared.geometry as geom_l


def calculate_spatial_weights_by_dist(gdf, search_distance_in_ft, standardization="R"):
    """Calculate spatial weights matrix based on distance threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    search_distance_in_ft : float
        The distance threshold in feet for defining spatial neighbors.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.

    Returns
    -------
    weights.DistanceBand
        The spatial weights matrix.

    """
    # Extract coordinates from GeoDataFrame
    gdf = gdf.copy()
    gdf = geom_l.convert_geometries_to_points(gdf)
    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    # Define neighbors based on distance threshold (e.g., distance less than a certain threshold)
    search_distance = geom_l.convert_value_in_ft_to_df_units(gdf, search_distance_in_ft)

    # Within the radius, alpha=-1 creates inverse-distance weights instead of equal neighbor weights.
    # Coordinates and the converted threshold must share a suitable projected CRS; this is not geodesic distance.
    spatial_weights_matrix = DistanceBand(coordinates, threshold=search_distance, binary=False, alpha=-1.0,
                                          silence_warnings=True)
    # Request the caller's weight transform before diagonal removal; R denotes row standardization.
    spatial_weights_matrix.transform = standardization
    # Exclude self-influence. Disconnected observations remain islands rather than receiving invented neighbors.
    spatial_weights_matrix = fill_diagonal(spatial_weights_matrix, 0)
    return spatial_weights_matrix


def calculate_spatial_weights_for_polygons(gdf, standardization="R"):
    """Calculate spatial weights matrix based on polygon contiguity.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing polygon geometries.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.

    Returns
    -------
    weights.Queen
        The spatial weights matrix based on Queen contiguity.

    """
    gdf = gdf.copy()
    # Queen adjacency includes shared vertices as well as edges; dataframe indexes provide the weight IDs.
    # libpysal emits a UserWarning when the polygon set contains islands or disconnected components;
    # that informational warning is expected for a valid neighbor graph and should remain visible to users.
    spatial_weights_matrix = Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
    spatial_weights_matrix.transform = standardization
    spatial_weights_matrix = fill_diagonal(spatial_weights_matrix, 0)
    return spatial_weights_matrix


def get_spatial_weights(gdf, search_distance_in_ft, standardization):
    """Get spatial weights matrix based on geometry type.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    search_distance_in_ft : float
        The search distance in feet (used for point-based analyses).
    standardization : str
        The type of spatial weights standardization ('R' or 'B').

    Returns
    -------
    weights object
        The spatial weights matrix (Queen for polygons, DistanceBand for points).

    """
    # Routing uses a case-sensitive test on the first geometry's type string.
    # Standard Shapely Polygon names are capitalized and currently fall through to the distance branch.
    if "polygon" in gdf.iloc[0]['geometry'].geom_type:
        print("Calculate spatial weights for polygons")
        if standardization.upper() == "B":
            print("The standardization is 'B'. Consider using 'R' for polygons.")
        spatial_weights_matrix = calculate_spatial_weights_for_polygons(gdf, standardization)
    else:
        print("Calculate spatial weights for points")
        spatial_weights_matrix = calculate_spatial_weights_by_dist(gdf, search_distance_in_ft, standardization)
    return spatial_weights_matrix
