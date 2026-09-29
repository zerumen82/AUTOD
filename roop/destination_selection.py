"""Look up the destination faces selected for a specific video file."""

import os


def video_reference_bank(references, video_path):
    """Return valid references for this path, preserving old pathless entries."""
    basename = os.path.basename(video_path)
    stored = (references or {}).get(f"selected_face_ref_{basename}")
    if isinstance(stored, dict):
        stored = [stored]
    if not isinstance(stored, (list, tuple)):
        return []

    bank = [ref for ref in stored if isinstance(ref, dict)
            and ref.get('embedding') is not None]
    if not any(ref.get('source_path') for ref in bank):
        return bank

    exact_path = os.path.normcase(os.path.abspath(video_path))
    return [ref for ref in bank if ref.get('source_path')
            and os.path.normcase(os.path.abspath(ref['source_path'])) == exact_path]


def missing_video_references(video_paths, references):
    """List videos that would render without a selected destination face."""
    return [path for path in video_paths
            if not video_reference_bank(references, path)]
