import csv
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_csv_data():
    csv_path = os.path.join(BASE_DIR, 'modules.csv')
    if not os.path.exists(csv_path):
        return None, []
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        return reader.fieldnames, list(reader)

@app.route('/')
def get_soul():
    target_cert = request.args.get('cert', '').strip()
    headers, rows = get_csv_data()
    
    # If no cert is requested, return the list of available cert columns
    if not target_cert:
        # Columns 3 onwards are usually the certifications in your CSV
        cert_list = headers[2:] 
        return jsonify({"available_certs": cert_list})

    modules = []
    for row in rows:
        level = row.get(target_cert, "").strip()
        if level in ["A", "S", "AD"]:
            modules.append({
                "title": row.get('Sub-Skill', 'Unnamed'),
                "category": row.get('Module/Sub-Module', 'General'),
                "level": level
            })
            
    return jsonify({"selected_certification": target_cert, "modules": modules})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
