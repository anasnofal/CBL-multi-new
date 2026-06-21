import os
import shutil
from pathlib import Path

import kagglehub


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))
    os.makedirs(project_data_dir, exist_ok=True)

    dataset_id = "anasnofal/uk-police-crime-dataset-mar-2023-feb-2025"

    # kagglehub downloads/extracts into its cache and returns that path
    path = kagglehub.dataset_download(dataset_id)

    print("Downloaded dataset to:", project_data_dir)
    print("Path to dataset files:", path)

    # If the library downloaded into a cache location, copy files into project data dir
    try:
        src = Path(path)
        dest = Path(project_data_dir)
        if src.exists():
            # copytree for directories, copy for files
            if src.is_dir():
                for item in src.iterdir():
                    target = dest / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
            else:
                shutil.copy2(src, dest / src.name)
            print("Copied dataset files into project data directory.")
        else:
            print("Warning: downloaded path does not exist:", path)
    except Exception as exc:  # noqa: BLE001 - surface user-facing error only
        print("Failed to copy dataset into project data directory:", exc)


if __name__ == "__main__":
    main()
