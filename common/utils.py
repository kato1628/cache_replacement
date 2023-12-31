import os
import shutil

def create_directory(path: str, overwrite=False):
    print(f"Creating directory: {path}")

    if os.path.exists(path):
        if overwrite:
            print(f"Overwriting existing directory: {path}")
            shutil.rmtree(path)
        else:
            raise ValueError(f"Directory already exists: {path}")
    os.makedirs(path)
            
