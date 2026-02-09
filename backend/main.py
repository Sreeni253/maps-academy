import csv
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def get_soul():
    # 1. Get the cert from the URL, default to 'BEE CEM'
    target_cert = request.args.get('cert', 'BEE CEM').strip() 
    
    modules = []
    csv_path = os.path.join(BASE_DIR, 'modules.csv')
    
    if os.path.exists(csv_path):
        # Using 'utf-8-sig' handles invisible characters often added by Excel
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            # Clean up column names (strip spaces)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            for row in reader:
                # Get the value for the requested certification column
                level = row.get(target_cert, "").strip()
                
                # If we find A, S, or AD, add it to the list
                if level in ["A", "S", "AD"]:
                    modules.append({
                        "title": row.get('Sub-Skill', 'Unnamed Skill'),
                        "category": row.get('Module/Sub-Module', 'General'),
                        "level": level
                    })
            
    # CRUCIAL: Always return a 'modules' key even if it's empty
    return jsonify({
        "selected_certification": target_cert,
        "tagline": f"Path: {target_cert}",
        "modules": modules  # This prevents the 'undefined' error
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
