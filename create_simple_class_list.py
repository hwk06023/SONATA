import json
import sys


def create_simple_list(ontology_file, output_file):
    with open(ontology_file, "r") as f:
        ontology = json.load(f)

    class_names = []
    for item in ontology:
        if "abstract" not in item.get("restrictions", []):
            class_names.append(item["name"])

    class_names.sort()

    with open(output_file, "w") as f:
        f.write("# AudioSet Classes\n")
        f.write("# Format: Class Name\n")
        f.write("# Total classes: {}\n\n".format(len(class_names)))
        for name in class_names:
            f.write(f"{name}\n")

    print(f"Created simple class list with {len(class_names)} entries at {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_simple_class_list.py <ontology_file> [output_file]")
        sys.exit(1)

    ontology_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "audioset_classes_list.txt"

    create_simple_list(ontology_file, output_file)
