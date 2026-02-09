import csv
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Helper to find files in the same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def get_soul():
    # 1. Read the JSON (Core Academy Info)
    with open(os.path.join(BASE_DIR, 'data.json'), 'r') as f:
        core_data = json.load(f)

    # 2. Read the CSV (The Energy/Safety Modules)
    modules_from_csv = []
    csv_path = os.path.join(BASE_DIR, 'modules.csv')
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # This ensures the key names match what the "Face" expects
                modules_from_csv.append({
                    "title": row.get('title', 'Untitled'),
                    "status": row.get('status', 'Upcoming')
                })

    # 3. Stitch them together
    # We replace the JSON modules with the fresh CSV modules
    core_data['academy_info']['modules'] = modules_from_csv
            
    return jsonify(core_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
