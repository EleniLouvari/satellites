"""Decrypt the Copernicus accounts frozen on one analysis run."""

from parcel_app.core import db
from parcel_app.core.security import cipher


def run_credentials(run):
    """Decrypt exactly the ordered frozen account identities, never all owner accounts."""
    rows = db.all_rows(
        """SELECT ra.username AS frozen_username,e.username,e.password
        FROM run_accounts ra JOIN eo_users e ON e.eo_user_id=ra.eo_user_id
        JOIN app_users u ON u.app_user_id=e.app_user_id
        WHERE ra.run_id=%s AND e.app_user_id=%s AND u.is_active ORDER BY ra.position""",
        (str(run["run_id"]), run["app_user_id"]),
    )
    if len(rows) != len(run["account_ids"]) or not rows:
        raise ValueError("Error: Selected Copernicus accounts are unavailable.")
    if any(row["username"] != row["frozen_username"] for row in rows):
        raise ValueError("Error: Copernicus account identity changed; restore the original account.")
    return [(row["frozen_username"], cipher().decrypt(row["password"].encode()).decode()) for row in rows]
