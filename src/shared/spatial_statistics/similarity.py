"""Buffered spatial extent and rowwise overlap comparison helpers."""

from IPython.display import display

import shared.geometry as geom_l
import shared.tabular as cm_l


def spatial_extent_similarity(gdf1, gdf2, buffer_radius):
    """Calculate spatial extent similarity between two GeoDataFrames.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame.
    buffer_radius : float
        The buffer radius to apply to the geometries.

    Returns
    -------
    None
        Prints the overlap percentage and displays the polygon difference.

    """
    gdf1 = gdf1.copy()
    gdf2 = gdf2.copy()
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)

    # This helper returns feet from native units, yet buffering consumes native units; preserve the existing conversion.
    buffer_dist = geom_l.convert_value_in_df_units_from_ft(gdf1, buffer_radius)

    outer_polygon1 = gdf1.copy()
    outer_polygon1['geometry'] = outer_polygon1['geometry'].buffer(buffer_dist)
    # Convex hulls fill internal gaps, so this measures enclosing extents rather than detailed occupied area.
    outer_polygon1 = outer_polygon1.geometry.unary_union.convex_hull

    outer_polygon2 = gdf2.copy()
    outer_polygon2['geometry'] = outer_polygon2['geometry'].buffer(buffer_dist)
    outer_polygon2 = outer_polygon2.geometry.unary_union.convex_hull

    # The displayed difference is directional (second minus first), whereas intersection-over-union is symmetric.
    pol_diference = outer_polygon2.difference(outer_polygon1)
    pol_intersection = outer_polygon2.intersection(outer_polygon1)
    pol_union = outer_polygon2.union(outer_polygon1)

    # A nonzero union area is assumed; degenerate extents have no defined area-overlap ratio.
    overlap = round(100*pol_intersection.area/pol_union.area, 2)
    print(f"Overlap percentage: {overlap}%")
    display(pol_diference)


def spatial_jaccard_similarity(gdf1, gdf2, buffer_radius):
    """Calculate Jaccard similarity index between two GeoDataFrames.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame.
    buffer_radius : float
        The buffer radius to apply to the geometries.

    Returns
    -------
    float
        The Jaccard similarity index.

    """
    gdf1 = gdf1.copy()
    gdf2 = gdf2.copy()
    cm_l.reset_index(gdf1, gdf2)
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)

    buffer_dist = geom_l.convert_value_in_df_units_from_ft(gdf1, buffer_radius)
    gdf1['geometry'] = gdf1['geometry'].buffer(buffer_dist)

    # intersects returns a row-aligned Boolean Series; len counts all results, not the number of True intersections.
    # This legacy count is therefore not a geometric Jaccard intersection size.
    intersection_count = len(gdf1.intersects(gdf2))
    union_count = len(gdf1) + len(gdf2) - intersection_count
    jaccard_index = intersection_count / union_count

    if jaccard_index == 0:
        print("The datasets have no spatial overlap.")
    elif jaccard_index == 1:
        print("The datasets are identical in terms of spatial extent.")
    else:
        print(f"The Jaccard similarity index is {jaccard_index:.2f}.")

    return jaccard_index
