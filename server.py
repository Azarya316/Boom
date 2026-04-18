import json
import base64
import os
import re
import traceback
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time

# ── ML ──
try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn non installé — ML désactivé. Lancez : pip install scikit-learn joblib numpy")

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Clé Groq (fixée côté serveur) ──
GROQ_API_KEY = "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"

MODEL_PATH        = "model.pkl"
SCORE_MODEL_PATH  = "score_model.pkl"
FEEDBACK_LOG      = "feedback.jsonl"
STRATEGY_LOG      = "strategy.json"

LABELS       = ["1", "X", "2"]

# ════════════════════════════════════════════════
#  SCORES POSSIBLES (prédicton de score exact)
# ════════════════════════════════════════════════
SCORE_LABELS = [
    "0-0", "1-0", "0-1", "1-1", "2-0", "0-2",
    "2-1", "1-2", "2-2", "3-0", "0-3", "3-1",
    "1-3", "3-2", "2-3", "3-3", "4-0", "0-4",
    "4-1", "1-4", "4-2", "2-4"
]

def score_to_result(score: str) -> int:
    """Convertit un score '2-1' en résultat 0=domicile, 1=nul, 2=visiteur."""
    try:
        a, b = map(int, score.split("-"))
        if a > b: return 0
        if a == b: return 1
        return 2
    except:
        return -1

def score_to_index(score: str) -> int:
    try:
        return SCORE_LABELS.index(score)
    except:
        return -1

# ════════════════════════════════════════════════
#  STRATÉGIES APPRISES (écrites dans strategy.json)
# ════════════════════════════════════════════════

DEFAULT_STRATEGY = {
    "version": 1,
    "trained_on": 0,
    "accuracy_result": 0.0,
    "accuracy_score": 0.0,
    "score_weights": {
        "0-0": 0.05, "1-0": 0.12, "0-1": 0.10,
        "1-1": 0.14, "2-0": 0.08, "0-2": 0.07,
        "2-1": 0.09, "1-2": 0.08, "2-2": 0.06,
        "3-0": 0.04, "0-3": 0.03, "3-1": 0.04,
        "1-3": 0.03, "3-2": 0.03, "2-3": 0.02,
        "3-3": 0.01, "4-0": 0.01, "0-4": 0.01,
        "4-1": 0.01, "1-4": 0.01, "4-2": 0.01, "2-4": 0.01
    },
    "confidence_threshold": 60.0,
    "reliable_threshold": 65.0,
    "use_calibrated": True,
    "feature_importance": {}
}

def load_strategy() -> dict:
    if os.path.exists(STRATEGY_LOG):
        try:
            with open(STRATEGY_LOG, encoding="utf-8") as f:
                s = json.load(f)
                # Fusion avec défaut pour les clés manquantes
                for k, v in DEFAULT_STRATEGY.items():
                    if k not in s:
                        s[k] = v
                return s
        except:
            pass
    return dict(DEFAULT_STRATEGY)

def save_strategy(strategy: dict):
    with open(STRATEGY_LOG, "w", encoding="utf-8") as f:
        json.dump(strategy, f, ensure_ascii=False, indent=2)

STRATEGY = load_strategy()

# ════════════════════════════════════════════════
#  FEATURE ENGINEERING (enrichi — 22 features)
# ════════════════════════════════════════════════

def _find_rank(team: str, rankings: list) -> int:
    if not team:
        return 10
    t = team.lower().strip()
    for r in rankings:
        name = r.get("team", "").lower()
        if t[:4] in name or name[:4] in t:
            return int(r.get("rank", 10))
    return 10

def _find_pts(team: str, rankings: list) -> float:
    if not team:
        return 20.0
    t = team.lower().strip()
    for r in rankings:
        name = r.get("team", "").lower()
        if t[:4] in name or name[:4] in t:
            return float(r.get("pts", 20))
    return 20.0

def _form_score(form: list) -> float:
    mapping = {"W": 3, "D": 1, "L": 0}
    scores = [mapping.get(r, 1) for r in (form or [])[-5:]]
    return sum(scores) / (3 * len(scores)) if scores else 0.5

def _form_goals_approx(form: list) -> float:
    """Estime un score moyen de buts à partir de la forme W/D/L."""
    mapping = {"W": 1.8, "D": 1.0, "L": 0.5}
    scores = [mapping.get(r, 1.0) for r in (form or [])[-5:]]
    return sum(scores) / len(scores) if scores else 1.0

def _streak(form: list) -> int:
    """Calcule la série actuelle : +N si N victoires, -N si N défaites."""
    if not form:
        return 0
    last = form[-1]
    streak = 0
    for r in reversed(form):
        if r == last:
            streak += 1
        else:
            break
    return streak if last == "W" else -streak

def extract_features(match: dict, rankings: list) -> np.ndarray:
    """
    Vecteur de 22 features :
     [0-2]   Probabilités implicites normalisées p1, px, p2
     [3-5]   Log-cotes log(c1), log(cx), log(c2)
     [6-8]   Rang A, Rang B, Écart de rang
     [9-11]  Points A, Points B, Différentiel de points
     [12-13] Forme A, Forme B (score 0-1)
     [14]    Différentiel de forme A-B
     [15-16] Série victoires/défaites A, B (streak)
     [17]    Probabilité Over 2.5
     [18]    Probabilité GG (les deux marquent)
     [19]    Probabilité Over 1.5
     [20]    Double chance 1X  (probabilité implicite)
     [21]    Double chance X2  (probabilité implicite)
    """
    c1 = max(float(match.get("cote1") or 3.0), 1.01)
    cx = max(float(match.get("coteX") or 3.5), 1.01)
    c2 = max(float(match.get("cote2") or 3.0), 1.01)

    margin = 1/c1 + 1/cx + 1/c2
    p1 = (1/c1) / margin
    px = (1/cx) / margin
    p2 = (1/c2) / margin

    rank_a = _find_rank(match.get("teamA", ""), rankings)
    rank_b = _find_rank(match.get("teamB", ""), rankings)
    gap    = abs(rank_a - rank_b)

    pts_a = _find_pts(match.get("teamA", ""), rankings)
    pts_b = _find_pts(match.get("teamB", ""), rankings)
    pts_diff = pts_a - pts_b

    form_a = _form_score(match.get("formA", []))
    form_b = _form_score(match.get("formB", []))

    streak_a = _streak(match.get("formA", []))
    streak_b = _streak(match.get("formB", []))

    over25  = max(float(match.get("over25")  or 2.0), 1.01)
    under25 = max(float(match.get("under25") or 1.8), 1.01)
    p_over  = (1/over25) / (1/over25 + 1/under25)

    cgg = max(float(match.get("coteGG") or 1.9), 1.01)
    cng = max(float(match.get("coteNG") or 1.9), 1.01)
    p_gg = (1/cgg) / (1/cgg + 1/cng)

    over15  = max(float(match.get("over15")  or 1.45), 1.01)
    under15 = max(float(match.get("under15") or 2.6),  1.01)
    p_o15   = (1/over15) / (1/over15 + 1/under15)

    dc1x = max(float(match.get("dc1x") or 1.2), 1.01)
    dcx2 = max(float(match.get("dcx2") or 1.5), 1.01)
    p_dc1x = 1 / dc1x
    p_dcx2 = 1 / dcx2

    return np.array([
        p1, px, p2,
        np.log(c1), np.log(cx), np.log(c2),
        rank_a, rank_b, gap,
        pts_a, pts_b, pts_diff,
        form_a, form_b, form_a - form_b,
        streak_a, streak_b,
        p_over, p_gg, p_o15,
        p_dc1x, p_dcx2
    ], dtype=np.float32)

def score_features(match: dict, rankings: list) -> np.ndarray:
    """
    Features supplémentaires pour la prédiction de score exact.
    Reprend les 22 features de base + probabilité Over et forme de buts estimée.
    """
    base = extract_features(match, rankings)
    ga = _form_goals_approx(match.get("formA", []))
    gb = _form_goals_approx(match.get("formB", []))
    extra = np.array([ga, gb, ga - gb, ga + gb], dtype=np.float32)
    return np.concatenate([base, extra])

# ════════════════════════════════════════════════
#  CHARGEMENT DES MODÈLES
# ════════════════════════════════════════════════

def load_model():
    if not SKLEARN_OK:
        return None
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            print(f"✅ Modèle résultat chargé depuis {MODEL_PATH}")
            return m
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle résultat : {e}")
    return None

def load_score_model():
    if not SKLEARN_OK:
        return None
    if os.path.exists(SCORE_MODEL_PATH):
        try:
            m = joblib.load(SCORE_MODEL_PATH)
            print(f"✅ Modèle score chargé depuis {SCORE_MODEL_PATH}")
            return m
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle score : {e}")
    return None

MODEL       = load_model()
SCORE_MODEL = load_score_model()
_train_lock = threading.Lock()

# ════════════════════════════════════════════════
#  ENTRAÎNEMENT
# ════════════════════════════════════════════════

def train_from_feedback():
    """
    Entraîne DEUX modèles :
      1. Modèle résultat  → prédit 1/X/2
      2. Modèle score     → prédit le score exact (0-0, 1-0, etc.)
    Réécrit strategy.json avec les nouvelles stats apprises.
    Retourne (model_result, model_score, n_samples, accuracy_result, accuracy_score)
    """
    global STRATEGY

    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn non disponible")

    if not os.path.exists(FEEDBACK_LOG):
        raise FileNotFoundError(f"{FEEDBACK_LOG} introuvable — aucun résultat enregistré.")

    rows = []
    with open(FEEDBACK_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) < 10:
        raise ValueError(f"Seulement {len(rows)} exemples — minimum 10 requis pour entraîner.")

    # ── Modèle 1 : Résultat (1/X/2) ──
    X_res = np.array([extract_features(r["match"], r.get("rankings", [])) for r in rows])
    y_res = np.array([int(r["result"]) for r in rows])

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        alpha=0.001
    )
    xgb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    rf  = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    lr  = LogisticRegression(max_iter=500, random_state=42, C=0.5)

    ensemble_res = VotingClassifier(
        estimators=[("mlp", mlp), ("xgb", xgb), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[4, 4, 2, 1]
    )

    pipeline_res = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    ensemble_res)
    ])
    pipeline_res.fit(X_res, y_res)

    # Cross-validation pour estimer l'accuracy réelle
    try:
        cv_scores_res = cross_val_score(pipeline_res, X_res, y_res, cv=min(5, len(rows)//5), scoring="accuracy")
        acc_res = float(cv_scores_res.mean())
    except:
        acc_res = 0.0

    joblib.dump(pipeline_res, MODEL_PATH)
    print(f"✅ Modèle résultat entraîné — accuracy CV : {acc_res*100:.1f}%")

    # ── Modèle 2 : Score exact ──
    score_rows = [r for r in rows if r.get("score")]
    pipeline_score = None
    acc_score = 0.0

    if len(score_rows) >= 10:
        X_sc = np.array([score_features(r["match"], r.get("rankings", [])) for r in score_rows])
        y_sc_raw = [score_to_index(r["score"]) for r in score_rows]
        valid = [(x, y) for x, y in zip(X_sc, y_sc_raw) if y >= 0]

        if len(valid) >= 10:
            X_sc_v = np.array([v[0] for v in valid])
            y_sc_v = np.array([v[1] for v in valid])

            mlp_sc  = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42, early_stopping=True)
            xgb_sc  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
            rf_sc   = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)

            ensemble_sc = VotingClassifier(
                estimators=[("mlp", mlp_sc), ("xgb", xgb_sc), ("rf", rf_sc)],
                voting="soft",
                weights=[4, 4, 2]
            )

            pipeline_score = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    ensemble_sc)
            ])
            pipeline_score.fit(X_sc_v, y_sc_v)

            try:
                cv_scores_sc = cross_val_score(pipeline_score, X_sc_v, y_sc_v, cv=min(5, len(valid)//5), scoring="accuracy")
                acc_score = float(cv_scores_sc.mean())
            except:
                acc_score = 0.0

            joblib.dump(pipeline_score, SCORE_MODEL_PATH)
            print(f"✅ Modèle score entraîné — accuracy CV : {acc_score*100:.1f}%")
    else:
        print(f"ℹ️  Seulement {len(score_rows)} matchs avec score — modèle score non entraîné encore.")

    # ── Calcul des poids de score à partir des données réelles ──
    score_counts = {}
    for r in rows:
        s = r.get("score")
        if s and s in SCORE_LABELS:
            score_counts[s] = score_counts.get(s, 0) + 1

    if score_counts:
        total_sc = sum(score_counts.values())
        learned_weights = {s: round(score_counts.get(s, 0) / total_sc, 4) for s in SCORE_LABELS}
    else:
        learned_weights = DEFAULT_STRATEGY["score_weights"]

    # ── Mise à jour de la stratégie ──
    new_conf_threshold = max(55.0, min(75.0, acc_res * 100 * 0.85))
    new_reliable_threshold = max(60.0, min(80.0, acc_res * 100 * 0.90))

    STRATEGY.update({
        "version":             STRATEGY.get("version", 1) + 1,
        "trained_on":          len(rows),
        "accuracy_result":     round(acc_res * 100, 1),
        "accuracy_score":      round(acc_score * 100, 1),
        "score_weights":       learned_weights,
        "confidence_threshold": round(new_conf_threshold, 1),
        "reliable_threshold":   round(new_reliable_threshold, 1),
    })
    save_strategy(STRATEGY)
    print(f"✅ strategy.json mis à jour (v{STRATEGY['version']}) — {len(rows)} matchs, acc={acc_res*100:.1f}%")

    return pipeline_res, pipeline_score, len(rows), acc_res, acc_score

# ════════════════════════════════════════════════
#  AUTO-ENTRAÎNEMENT EN ARRIÈRE-PLAN
# ════════════════════════════════════════════════

def auto_train_loop():
    """
    Vérifie toutes les 60s si de nouveaux exemples sont disponibles.
    Lance un réentraînement automatique dès que 10+ exemples existent
    ET que le modèle n'est pas à jour.
    """
    global MODEL, SCORE_MODEL, STRATEGY
    last_count = 0

    while True:
        try:
            if os.path.exists(FEEDBACK_LOG):
                count = sum(1 for line in open(FEEDBACK_LOG, encoding="utf-8") if line.strip())
                if count >= 10 and count != last_count:
                    print(f"🤖 Auto-training lancé ({count} exemples)...")
                    with _train_lock:
                        MODEL, SCORE_MODEL, n, a_res, a_sc = train_from_feedback()
                        last_count = count
                        print(f"   Résultat accuracy : {a_res*100:.1f}% | Score accuracy : {a_sc*100:.1f}%")
        except Exception as e:
            print("Auto-train error:", e)
        time.sleep(60)

_auto_thread = threading.Thread(target=auto_train_loop, daemon=True)
_auto_thread.start()

# ════════════════════════════════════════════════
#  PRÉDICTION DE SCORE (heuristique + ML)
# ════════════════════════════════════════════════

def predict_score_heuristic(match: dict, rankings: list, result_probs: list) -> list:
    """
    Prédit un classement de scores possibles basé sur :
    - les probabilités de résultat (p1, px, p2)
    - les probabilités Over/Under
    - les poids appris dans STRATEGY
    - la forme des équipes
    """
    p1, px, p2 = result_probs

    over25  = max(float(match.get("over25")  or 2.0), 1.01)
    under25 = max(float(match.get("under25") or 1.8), 1.01)
    p_over  = (1/over25) / (1/over25 + 1/under25)

    cgg = max(float(match.get("coteGG") or 1.9), 1.01)
    cng = max(float(match.get("coteNG") or 1.9), 1.01)
    p_gg = (1/cgg) / (1/cgg + 1/cng)

    over15  = max(float(match.get("over15")  or 1.45), 1.01)
    under15 = max(float(match.get("under15") or 2.6),  1.01)
    p_o15   = (1/over15) / (1/over15 + 1/under15)

    form_a = _form_score(match.get("formA", []))
    form_b = _form_score(match.get("formB", []))

    weights = STRATEGY.get("score_weights", DEFAULT_STRATEGY["score_weights"])

    scores_prob = {}
    for score in SCORE_LABELS:
        a, b = map(int, score.split("-"))
        total = a + b

        # Probabilité de résultat
        if a > b:   res_p = p1
        elif a == b: res_p = px
        else:        res_p = p2

        # Probabilité de volume de buts
        if total == 0:
            vol_p = (1 - p_o15) * 0.8
        elif total == 1:
            vol_p = (p_o15 - p_over) * 0.7 + (1 - p_o15) * 0.2
        elif total == 2:
            vol_p = (p_over * 0.5 + (1 - p_gg) * p_over * 0.3)
        elif total == 3:
            vol_p = p_over * p_gg * 0.6
        elif total == 4:
            vol_p = p_over * p_gg * 0.3
        else:
            vol_p = p_over * p_gg * 0.1

        # Probabilité de GG (les deux marquent)
        if a > 0 and b > 0:
            gg_p = p_gg
        else:
            gg_p = 1 - p_gg

        # Forme des équipes
        if a > 0:
            att_a = 0.5 + form_a * 0.5
        else:
            att_a = 1.0

        if b > 0:
            att_b = 0.5 + form_b * 0.5
        else:
            att_b = 1.0

        base = weights.get(score, 0.01)
        prob = base * res_p * vol_p * gg_p * att_a * att_b
        scores_prob[score] = max(prob, 0.0001)

    # Normaliser
    total = sum(scores_prob.values())
    scores_prob = {s: round(v / total * 100, 2) for s, v in scores_prob.items()}

    # Trier par probabilité décroissante
    sorted_scores = sorted(scores_prob.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:10]

# ════════════════════════════════════════════════
#  ROUTES STATIQUES
# ════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# ════════════════════════════════════════════════
#  ROUTE /scan  (OCR via Groq — inchangé)
# ════════════════════════════════════════════════

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

    raw_text = ""
    try:
        messages_content = [{"type": "text", "text": prompt_text}]

        if odds_file:
            img_data = base64.b64encode(odds_file.read()).decode('utf-8')
            mime = odds_file.mimetype or 'image/jpeg'
            messages_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}})
            messages_content.append({"type": "text", "text": "Image ci-dessus: capture des COTES 1X2 (et Over/Under, GG/NG, Double Chance si visibles)."})

        if rank_file:
            img_data = base64.b64encode(rank_file.read()).decode('utf-8')
            mime = rank_file.mimetype or 'image/jpeg'
            messages_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}})
            messages_content.append({"type": "text", "text": "Image ci-dessus: capture du CLASSEMENT des équipes (rang, points, forme récente)."})

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": messages_content}],
            "temperature": 0.1,
            "max_tokens": 2000
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        resp_json = resp.json()
        if not resp.ok:
            err = resp_json.get('error', {}).get('message', str(resp_json))
            return jsonify({"error": f"Erreur Groq ({resp.status_code}): {err}"}), 500

        raw_text = resp_json['choices'][0]['message']['content']

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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /scan-results  (OCR résultats → auto-apprentissage)
# ════════════════════════════════════════════════

@app.route('/scan-results', methods=['POST'])
def scan_results():
    """
    Reçoit une image de résultats de matchs (capture Bet261).
    Extrait automatiquement : teamA, teamB, score final (ex: "2-1").
    Retourne : { "results": [{"teamA":"...", "teamB":"...", "score":"2-1"}, ...] }

    Le front-end croise ensuite ces résultats avec les cartes existantes
    et envoie le tout à /import-results pour entraîner l'IA.
    """
    results_file = request.files.get('results_img')
    team_a_hint  = request.form.get('teamA', '')
    team_b_hint  = request.form.get('teamB', '')

    if not results_file:
        return jsonify({"error": "Aucune image de résultats fournie"}), 400

    # Construire le prompt selon qu'on a des hints d'équipes ou non
    if team_a_hint and team_b_hint:
        context = f"Le match est entre '{team_a_hint}' (domicile) et '{team_b_hint}' (visiteur)."
    else:
        context = "Extrais TOUS les matchs visibles dans l'image."

    prompt_text = f"""Tu analyses une capture d'écran montrant des résultats de matchs de football virtuels (app Bet261).
{context}
Réponds UNIQUEMENT en JSON valide, sans backticks, sans explication.

Format EXACT attendu:
{{
  "results": [
    {{
      "teamA": "Nom équipe domicile",
      "teamB": "Nom équipe visiteur",
      "score": "2-1"
    }}
  ]
}}

Règles strictes:
- "score" doit toujours être au format "BUTS_DOMICILE-BUTS_VISITEUR" (ex: "0-0", "1-0", "2-3")
- Extrais TOUS les matchs terminés visibles avec leur score final
- Normalise les noms (Egypt→Égypte, Morocco→Maroc, Ivory Coast→Côte d'Ivoire, Senegal→Sénégal, etc.)
- Si tu ne vois pas de score clair pour un match, ne l'inclus pas
- Ne rajoute AUCUN texte hors du JSON"""

    raw_text = ""
    try:
        img_data = base64.b64encode(results_file.read()).decode('utf-8')
        mime = results_file.mimetype or 'image/jpeg'

        messages_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}},
            {"type": "text", "text": "Image ci-dessus : résultats des matchs terminés."}
        ]

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": messages_content}],
            "temperature": 0.05,
            "max_tokens": 1500
        }

        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        resp_json = resp.json()
        if not resp.ok:
            err = resp_json.get('error', {}).get('message', str(resp_json))
            return jsonify({"error": f"Erreur Groq ({resp.status_code}): {err}"}), 500

        raw_text = resp_json['choices'][0]['message']['content']

        # Nettoyage JSON
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("```").strip()

        parsed = json.loads(clean)
        results_list = parsed.get("results", [])

        # Validation et normalisation des scores
        validated = []
        for r in results_list:
            score = str(r.get("score", "")).strip()
            # Accepter formats: "2-1", "2:1", "2 1"
            score = re.sub(r'[:\s]', '-', score)
            parts = re.findall(r'\d+', score)
            if len(parts) >= 2:
                score = f"{parts[0]}-{parts[1]}"
                validated.append({
                    "teamA": r.get("teamA", ""),
                    "teamB": r.get("teamB", ""),
                    "score": score
                })

        print(f"✅ /scan-results : {len(validated)} résultat(s) extrait(s)")
        return jsonify({"results": validated, "raw_count": len(results_list)})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON mal formé: {str(e)}. Brut: {raw_text[:300]}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /predict  (prédiction ML enrichie)
# ════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    """
    Corps JSON :
      { "match": {...}, "rankings": [...] }

    Retourne :
      {
        "prediction": "1"|"X"|"2",
        "confidence": 74.3,
        "reliable": true,
        "probabilities": { "1": 74.3, "X": 18.1, "2": 7.6 },
        "scores": [["1-0", 22.4], ["2-0", 15.1], ...],
        "top_score": "1-0",
        "ml_active": true,
        "strategy_version": 3,
        "trained_on": 150
      }
    """
    global MODEL, SCORE_MODEL

    data     = request.get_json(force=True) or {}
    match    = data.get("match", {})
    rankings = data.get("rankings", [])

    threshold   = STRATEGY.get("confidence_threshold", 60.0)
    reliable_th = STRATEGY.get("reliable_threshold",   65.0)

    if MODEL is None:
        return _fallback_predict(match, rankings)

    try:
        feat  = extract_features(match, rankings).reshape(1, -1)
        probs = MODEL.predict_proba(feat)[0]
        best  = int(np.argmax(probs))
        conf  = float(probs[best]) * 100

        probs_dict = {
            "1": round(float(probs[0]) * 100, 1),
            "X": round(float(probs[1]) * 100, 1),
            "2": round(float(probs[2]) * 100, 1),
        }

        # ── Prédiction de score ──
        if SCORE_MODEL is not None:
            sf = score_features(match, rankings).reshape(1, -1)
            sc_probs = SCORE_MODEL.predict_proba(sf)[0]
            # Associer labels et probabilités
            classes = SCORE_MODEL.classes_
            score_list = []
            for i, cls in enumerate(classes):
                if cls < len(SCORE_LABELS):
                    score_list.append((SCORE_LABELS[cls], round(float(sc_probs[i]) * 100, 2)))
            score_list.sort(key=lambda x: x[1], reverse=True)
            top_scores = score_list[:10]
        else:
            # Fallback heuristique
            top_scores = predict_score_heuristic(match, rankings, [probs[0], probs[1], probs[2]])

        top_score = top_scores[0][0] if top_scores else "1-0"

        return jsonify({
            "prediction":       LABELS[best],
            "confidence":       round(conf, 1),
            "reliable":         conf >= reliable_th,
            "probabilities":    probs_dict,
            "scores":           top_scores,
            "top_score":        top_score,
            "ml_active":        True,
            "score_ml_active":  SCORE_MODEL is not None,
            "strategy_version": STRATEGY.get("version", 1),
            "trained_on":       STRATEGY.get("trained_on", 0),
            "accuracy_result":  STRATEGY.get("accuracy_result", 0),
            "accuracy_score":   STRATEGY.get("accuracy_score", 0),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _fallback_predict(match: dict, rankings: list):
    """Logique déterministe de secours quand aucun modèle n'est encore disponible."""
    c1 = float(match.get("cote1") or 3.0)
    c2 = float(match.get("cote2") or 3.0)
    cote_fav = min(c1, c2)
    outcome  = "1" if c1 <= c2 else "2"

    if   cote_fav <= 1.20: conf = 90.0
    elif cote_fav <  1.50: conf = 75.0
    elif cote_fav <= 2.20: conf = 60.0
    else:                  conf, outcome = 40.0, "X"

    p1 = conf/100 if outcome == "1" else (1 - conf/100) / 2
    px = conf/100 if outcome == "X" else (1 - conf/100) / 2
    p2 = conf/100 if outcome == "2" else (1 - conf/100) / 2

    top_scores = predict_score_heuristic(match, rankings, [p1, px, p2])

    return jsonify({
        "prediction":      outcome,
        "confidence":      conf,
        "reliable":        conf >= 60.0,
        "probabilities":   None,
        "scores":          top_scores,
        "top_score":       top_scores[0][0] if top_scores else "1-0",
        "ml_active":       False,
        "score_ml_active": False,
        "note":            "Modèle ML non encore entraîné — prédiction déterministe.",
        "strategy_version": STRATEGY.get("version", 1),
        "trained_on":       STRATEGY.get("trained_on", 0),
    })

# ════════════════════════════════════════════════
#  ROUTE /feedback  (enregistrement d'un résultat réel)
# ════════════════════════════════════════════════

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Corps JSON :
      {
        "match":    { ...données du match... },
        "rankings": [ ...classement... ],
        "result":   0 | 1 | 2,         (0=Victoire A, 1=Nul, 2=Victoire B)
        "score":    "2-1"              (optionnel mais TRÈS important pour apprendre les scores)
      }
    """
    data = request.get_json(force=True) or {}

    if "match" not in data or "result" not in data:
        return jsonify({"error": "Champs requis manquants : match, result"}), 400

    try:
        result = int(data["result"])
        if result not in (0, 1, 2):
            raise ValueError
    except:
        return jsonify({"error": "result doit être 0, 1 ou 2"}), 400

    score = data.get("score", "")
    if score and score not in SCORE_LABELS:
        # Essai de normalisation
        parts = re.findall(r'\d+', score)
        if len(parts) >= 2:
            score = f"{parts[0]}-{parts[1]}"

    record = {
        "match":    data["match"],
        "rankings": data.get("rankings", []),
        "result":   result,
        "score":    score if score else None
    }

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    count = sum(1 for line in open(FEEDBACK_LOG, encoding="utf-8") if line.strip())
    return jsonify({
        "ok":      True,
        "saved":   count,
        "message": f"{count} exemple(s). {'Réentraînement automatique en cours...' if count >= 10 else 'Encore ' + str(10-count) + ' exemples avant le premier entraînement.'}"
    })

# ════════════════════════════════════════════════
#  ROUTE /import-results  (importer des résultats en masse)
# ════════════════════════════════════════════════

@app.route('/import-results', methods=['POST'])
def import_results():
    """
    Importe une liste de matchs avec leurs vrais résultats.
    Corps JSON :
    {
      "results": [
        {
          "match":    { teamA, teamB, cote1, coteX, cote2, ... },
          "rankings": [...],
          "result":   0|1|2,
          "score":    "2-1"   (optionnel)
        },
        ...
      ]
    }
    Après import, déclenche automatiquement un réentraînement.
    """
    global MODEL, SCORE_MODEL

    data = request.get_json(force=True) or {}
    results = data.get("results", [])

    if not results:
        return jsonify({"error": "Aucun résultat dans 'results'"}), 400

    saved = 0
    errors = []

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        for i, row in enumerate(results):
            try:
                result = int(row.get("result", -1))
                if result not in (0, 1, 2):
                    # Essai de déduction depuis le score
                    score = row.get("score", "")
                    if score:
                        result = score_to_result(score)
                    if result < 0:
                        errors.append(f"Ligne {i}: result invalide")
                        continue

                score = row.get("score", "")
                if score and score not in SCORE_LABELS:
                    parts = re.findall(r'\d+', str(score))
                    if len(parts) >= 2:
                        score = f"{parts[0]}-{parts[1]}"
                    else:
                        score = None

                record = {
                    "match":    row.get("match", {}),
                    "rankings": row.get("rankings", []),
                    "result":   result,
                    "score":    score if score else None
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
            except Exception as e:
                errors.append(f"Ligne {i}: {str(e)}")

    if saved == 0:
        return jsonify({"error": "Aucun résultat valide importé", "details": errors}), 400

    # Déclencher l'entraînement immédiatement
    train_result = {}
    try:
        with _train_lock:
            MODEL, SCORE_MODEL, n, a_res, a_sc = train_from_feedback()
        train_result = {
            "trained": True,
            "samples": n,
            "accuracy_result": round(a_res * 100, 1),
            "accuracy_score":  round(a_sc * 100, 1),
            "strategy_version": STRATEGY.get("version", 1),
        }
    except Exception as e:
        train_result = {"trained": False, "reason": str(e)}

    return jsonify({
        "ok":           True,
        "imported":     saved,
        "errors":       errors,
        "training":     train_result,
        "total_in_log": sum(1 for line in open(FEEDBACK_LOG, encoding="utf-8") if line.strip()),
        "message":      f"✅ {saved} résultats importés et modèle réentraîné !"
    })

# ════════════════════════════════════════════════
#  ROUTE /train  (forcer un entraînement)
# ════════════════════════════════════════════════

@app.route('/train', methods=['POST'])
def train():
    global MODEL, SCORE_MODEL

    if not SKLEARN_OK:
        return jsonify({"error": "scikit-learn non installé sur le serveur."}), 503

    try:
        with _train_lock:
            MODEL, SCORE_MODEL, n, a_res, a_sc = train_from_feedback()
        return jsonify({
            "ok":               True,
            "samples":          n,
            "accuracy_result":  round(a_res * 100, 1),
            "accuracy_score":   round(a_sc * 100, 1),
            "strategy_version": STRATEGY.get("version", 1),
            "message":          f"✅ Modèle entraîné sur {n} matchs. Résultat: {a_res*100:.1f}% | Score: {a_sc*100:.1f}%"
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /status  (état complet du système)
# ════════════════════════════════════════════════

@app.route('/status', methods=['GET'])
def status():
    nb_feedback = 0
    nb_with_score = 0
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    nb_feedback += 1
                    try:
                        r = json.loads(line)
                        if r.get("score"):
                            nb_with_score += 1
                    except:
                        pass

    return jsonify({
        "ml_active":          MODEL is not None,
        "score_ml_active":    SCORE_MODEL is not None,
        "sklearn_ok":         SKLEARN_OK,
        "model_file":         os.path.exists(MODEL_PATH),
        "score_model_file":   os.path.exists(SCORE_MODEL_PATH),
        "feedback_count":     nb_feedback,
        "with_score":         nb_with_score,
        "ready_to_train":     nb_feedback >= 10,
        "strategy": {
            "version":            STRATEGY.get("version", 1),
            "trained_on":         STRATEGY.get("trained_on", 0),
            "accuracy_result":    STRATEGY.get("accuracy_result", 0),
            "accuracy_score":     STRATEGY.get("accuracy_score", 0),
            "confidence_threshold": STRATEGY.get("confidence_threshold", 60.0),
            "reliable_threshold": STRATEGY.get("reliable_threshold", 65.0),
        },
        "score_labels":       SCORE_LABELS,
    })

# ════════════════════════════════════════════════
#  ROUTE /strategy  (lire / réécrire la stratégie)
# ════════════════════════════════════════════════

@app.route('/strategy', methods=['GET'])
def get_strategy():
    return jsonify(STRATEGY)

@app.route('/strategy', methods=['POST'])
def update_strategy():
    """Permet de forcer manuellement certains paramètres de stratégie."""
    global STRATEGY
    updates = request.get_json(force=True) or {}
    STRATEGY.update(updates)
    save_strategy(STRATEGY)
    return jsonify({"ok": True, "strategy": STRATEGY})

# ════════════════════════════════════════════════
#  DÉMARRAGE
# ════════════════════════════════════════════════

if __name__ == '__main__':
    print("✅ Serveur Safe Boom démarré → http://localhost:5000")
    print(f"   ML résultat actif  : {MODEL is not None}")
    print(f"   ML score actif     : {SCORE_MODEL is not None}")
    print(f"   sklearn dispo      : {SKLEARN_OK}")
    print(f"   Stratégie v{STRATEGY.get('version', 1)} — entraîné sur {STRATEGY.get('trained_on', 0)} matchs")
    print(f"   Accuracy résultat  : {STRATEGY.get('accuracy_result', 0)}%")
    print(f"   Accuracy score     : {STRATEGY.get('accuracy_score', 0)}%")
    app.run(host='0.0.0.0', port=5000, debug=True)
