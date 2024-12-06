import os

def export_folder_structure(root_dir, output_file, max_files=30):
    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Write the current folder's path
            depth = dirpath.replace(root_dir, "").count(os.sep)
            indent = "    " * depth
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")
            
            # Write up to `max_files` filenames in the current folder
            for filename in filenames[:max_files]:
                f.write(f"{indent}    {filename}\n")

export_folder_structure('/nethome/mma81/storage/CrossNews', '/nethome/mma81/storage/CrossNews/file_structure.txt')