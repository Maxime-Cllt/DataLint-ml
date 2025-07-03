from pathlib import Path


def get_project_root(marker="README.md") -> Path:
    """
    Get the root path of the project by searching for a marker file.
    :param marker:  The name of the marker file to identify the project root (default is "README.md").
    :return:  Path object pointing to the project root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'")
