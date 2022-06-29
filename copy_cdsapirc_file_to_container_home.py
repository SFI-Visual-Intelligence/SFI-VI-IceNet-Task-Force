import os
import shutil


print(f"Home directory contents before copying: {os.listdir(os.path.expanduser('~'))}")
print(f"Does '.cdsapirc' exist in cwd?: {os.path.exists('.cdsapirc')}")
# if os.path.exists('.cdsapirc'):
shutil.copy(".cdsapirc", os.path.expanduser('~'))

print(f"Home directory contents after copying: {os.listdir(os.path.expanduser('~'))}")
