import csv
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def get_soul():
    # Get the certification filter from the URL (e.g., ?cert=BEE CEM)
    target_cert = request.args.get('cert', 'BEE CEM') 
    
    modules = []
    csv_path = os.path.join(BASE_DIR, 'modules.csv')
    
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if this skill is required (A, S, or AD) for the selected cert
                level = row.get(target_cert, "").strip()
                if level in ["A", "S", "AD"]:
                    modules.append({
                        "title": row.get('Sub-Skill', 'Unnamed Skill'),
                        "category": row.get('Module/Sub-Module', 'General'),
                        "level": level, # Awareness, Skill, or Advanced
                        "status": f"Level: {level}"
                    })
            
    return jsonify({
        "selected_certification": target_cert,
        "tagline": f"Professional Path: {target_cert}",
        "modules": modules
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
