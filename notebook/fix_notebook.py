import json

# Read the notebook
with open('yiedl_crypto_model.ipynb', 'r') as f:
    notebook = json.load(f)

# Add empty outputs array to any code cell missing it
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'outputs' not in cell:
        cell['outputs'] = []
    if 'execution_count' not in cell and cell['cell_type'] == 'code':
        cell['execution_count'] = None

# Save the fixed notebook
with open('yiedl_crypto_model.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook has been fixed!")