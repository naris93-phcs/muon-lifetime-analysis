from pathlib import Path

import awkward as ak
import uproot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_DATA_DIR = PROJECT_ROOT / "data" / "root"


def find_root_files(directory: Path) -> list[Path]:
    """Return all ROOT files found inside the given directory."""
    return sorted(directory.glob("*.root"))


def inspect_root_file(file_path: Path) -> None:
    """Inspect the structure and basic contents of a ROOT file."""

    print("=" * 70)
    print("ROOT FILE INSPECTION")
    print("=" * 70)
    print(f"File: {file_path.name}")
    print(f"Path: {file_path}")
    print()

    with uproot.open(file_path) as root_file:
        print("Top-level objects:")
        print("-" * 70)

        for key, class_name in root_file.classnames().items():
            print(f"{key:<40} {class_name}")

        print()

        for key, class_name in root_file.classnames().items():
            if "TTree" not in class_name:
                continue

            tree = root_file[key]

            print("=" * 70)
            print(f"TTree: {key}")
            print("=" * 70)
            print(f"Number of entries/events: {tree.num_entries}")
            print()

            print("Branches:")
            print("-" * 70)

            for branch_name, branch in tree.items():
                print(
                    f"{branch_name:<35} "
                    f"typename={branch.typename}"
                )

            print()

            print("Testing the first event:")
            print("-" * 70)

            try:
                first_event = tree.arrays(
                    entry_start=0,
                    entry_stop=1,
                    library="ak",
                )
            except Exception as error:
                print(f"Could not read first event: {error}")
                continue

            for field in first_event.fields:
                values = first_event[field]

                print(f"\nBranch: {field}")
                print(f"Awkward type: {ak.type(values)}")

                try:
                    first_value = values[0]
                except (IndexError, TypeError):
                    print("No value found in the first event.")
                    continue

                try:
                    value_length = len(first_value)
                except TypeError:
                    value_length = None

                if value_length is None:
                    print(f"First value: {first_value}")
                else:
                    print(f"Length in first event: {value_length}")
                    print(f"First 10 values: {first_value[:10]}")

            print()


def main() -> None:
    root_files = find_root_files(ROOT_DATA_DIR)

    if not root_files:
        raise FileNotFoundError(
            f"No ROOT files were found in:\n{ROOT_DATA_DIR}"
        )

    print("Available ROOT files:")
    print("-" * 70)

    for index, file_path in enumerate(root_files):
        size_mb = file_path.stat().st_size / (1024 ** 2)
        print(f"[{index}] {file_path.name} ({size_mb:.2f} MB)")

    print()

    selected_index = int(input("Select ROOT file index: "))

    if selected_index < 0 or selected_index >= len(root_files):
        raise IndexError("Invalid ROOT file index.")

    inspect_root_file(root_files[selected_index])


if __name__ == "__main__":
    main()