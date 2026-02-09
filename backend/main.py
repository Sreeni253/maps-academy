import csv
import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def get_soul():
    modules = []
    csv_path = os.path.join(BASE_DIR, 'modules.csv')
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We skip empty rows and header rows
                if row.get('Sub-Skill'):
                    modules.append({
                        "title": row['Sub-Skill'],
                        "category": row.get('Module/Sub-Module', 'General'),
                        "status": "Verified Professional Skill"
                    })
            
    return jsonify({
        "academy_info": {
            "tagline": "Mapping and Advancing Professional Skills",
            "modules": modules[:20] # Taking the first 20 for a clean look
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
