# version_extractor.py
#
# Finds the nearest release version tag for a given datetime.
# Used by the Chunker to annotate every chunk with the version that
# was current when the source object was created.
#
# WHY THIS MATTERS:
# The critic uses version_tag to detect version mismatches.
# "This answer is based on a chunk from v0.2 but you asked about v0.4"
# Only possible if every chunk carries its version context.
#
# HOW THE VERSION MAP WORKS:
# The orchestrator builds a dict: {release_created_at: tag_name}
# sorted by datetime. For any issue/PR created_at, we find the most
# recent release that came BEFORE it. That's the version it belonged to.

from datetime import datetime
from typing import Optional


def extract_version_tag(
    created_at: datetime,
    version_map: dict[datetime, str],
) -> Optional[str]:
    """
    Finds the nearest release version that was current when created_at occurred.

    Example:
        version_map = {
            datetime(2023, 1, 1): "v0.1.0",
            datetime(2023, 6, 1): "v0.2.0",
            datetime(2024, 1, 1): "v0.3.0",
        }
        extract_version_tag(datetime(2023, 8, 15), version_map)
        → "v0.2.0"   (most recent release before Aug 2023)

    Args:
        created_at:  When the GitHub object was created.
        version_map: {release_datetime: tag_name} — built from RawRelease list.

    Returns:
        Version tag string (e.g. "v0.3.1") or None if no releases predate created_at.
    """
    if not version_map:
        return None

    # Find releases that came before or at the same time as created_at
    prior_releases = {
        dt: tag for dt, tag in version_map.items()
        if dt <= created_at
    }

    if not prior_releases:
        return None

    # Return the tag of the most recent prior release
    most_recent_dt = max(prior_releases.keys())
    return prior_releases[most_recent_dt]


def build_version_map(releases: list) -> dict[datetime, str]:
    """
    Builds a version map from a list of RawRelease objects.
    Called by IngestionOrchestrator before chunking starts.

    Args:
        releases: List of RawRelease from GitHubFetcher.

    Returns:
        {release_created_at: tag_name} dict.
    """
    return {
        release.created_at: release.tag_name
        for release in releases
        if release.tag_name  # skip releases without tags
    }