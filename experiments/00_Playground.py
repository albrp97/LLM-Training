# list directories
import os

print("Current working directory:", os.getcwd())
print("List of directories:")
for item in os.listdir():
    if os.path.isdir(item):
        print(" -", item)
        