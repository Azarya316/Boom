import json
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Clé Groq (fixée côté serveur) ──
GROQ_API_KEY = "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"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/scan', methods=['POST'])
def scan_images():
    odds_file = request.files.get('odds')
    rank_file = request.files.get('rank')

    if not odds_file and not rank_file:
        return jsonify({"error": "Aucune image fournie"}), 400

    prompt_text = """Tu analyses des captures d'écran de l'appli de paris sportifs Bet261 (version afrique/CAN virtuelle).
Réponds UNIQUEMENT en JSON valide, rien d'autre, sans backticks.

Format de réponse attendu:
{
  "matches": [
    {
      "teamA": "Nom équipe 1",
      "teamB": "Nom équipe 2",
      "cote1": 1.03,
      "coteX": 12.56,
      "cote2": 99.04,
      "over25": 2.10,
      "under25": 1.72,
      "over15": 1.45,
      "under15": 2.60,
      "coteGG": 1.80,
      "coteNG": 1.95,
      "dc1x": 1.22,
      "dcx2": 1.55
    }
  ],
  "rankings": [
    { "team": "Nom équipe", "rank": 1, "pts": 27, "form": ["W","W","W","W","D"] }
  ]
}

Instructions:
- Pour les cotes: extrais les valeurs 1X2 pour chaque match visible, ainsi que Over/Under, GG/NG et Double Chance si présentes.
- Pour le classement: extrais rang, nom équipe, points, et historique des 5 derniers matchs (W/D/L).
- Si une image n'est pas fournie, retourne un tableau vide pour ce champ.
- Normalise les noms en français (ex: Ivory Coast -> Côte d'Ivoire, Egypt -> Égypte, Algeria -> Algérie, South Africa -> Afrique du Sud).
- Ne rajoute AUCUN texte autour du JSON."""

    try:
        messages_content = [{"type": "text", "text": prompt_text}]

        if odds_file:
            img_data = base64.b64encode(odds_file.read()).decode('utf-8')
            mime = odds_file.mimetype or 'image/jpeg'
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_data}"}
            })
            messages_content.append({
                "type": "text",
                "text": "Image ci-dessus: capture des COTES 1X2 (et Over/Under, GG/NG, Double Chance si visibles)."
            })

        if rank_file:
            img_data = base64.b64encode(rank_file.read()).decode('utf-8')
            mime = rank_file.mimetype or 'image/jpeg'
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_data}"}
            })
            messages_content.append({
                "type": "text",
                "text": "Image ci-dessus: capture du CLASSEMENT des équipes (rang, points, forme récente)."
            })

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": messages_content}],
            "temperature": 0.1,
            "max_tokens": 2000
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )

        resp_json = resp.json()

        if not resp.ok:
            err = resp_json.get('error', {}).get('message', str(resp_json))
            return jsonify({"error": f"Erreur Groq ({resp.status_code}): {err}"}), 500

        raw_text = resp_json['choices'][0]['message']['content']

        # Nettoyage robuste du JSON retourné
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]
        clean_text = clean_text.strip().rstrip("```").strip()

        parsed = json.loads(clean_text)
        return jsonify(parsed)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Réponse invalide (JSON mal formé): {str(e)}. Brut: {raw_text[:300]}"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ Serveur Safe Boom démarré → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)