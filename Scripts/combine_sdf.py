import os

input_folder = "ligands.sdf"  # this is your folder with all the individual .sdf files
output_file = "ligands_merged.sdf"

with open(output_file, 'w') as outfile:
    for file in os.listdir(input_folder):
        if file.endswith('.sdf'):
            file_path = os.path.join(input_folder, file)
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content.endswith("$$$$"):
                    content += "\n$$$$"
                outfile.write(content + "\n")

print(f"✅ Merged into: {output_file}")
