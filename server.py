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
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn non installé — ML désactivé. Lancez : pip install scikit-learn joblib numpy")

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Clé Groq ──
GROQ_API_KEY = "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"

MODEL_PATH        = "model.pkl"
SCORE_MODEL_PATH  = "score_model.pkl"
FEEDBACK_LOG      = "feedback.jsonl"
STRATEGY_LOG      = "strategy.json"

LABELS       = ["1", "X", "2"]

# ════════════════════════════════════════════════
#  SCORES POSSIBLES
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
#  STRATÉGIES APPRISES
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
#  FEATURE ENGINEERING
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
    mapping = {"W": 1.8, "D": 1.0, "L": 0.5}
    scores = [mapping.get(r, 1.0) for r in (form or [])[-5:]]
    return sum(scores) / len(scores) if scores else 1.0

def _streak(form: list) -> int:
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

def load_feedback_rows() -> list:
    """Charge le feedback en dédupliquant par (teamA, teamB, score)."""
    if not os.path.exists(FEEDBACK_LOG):
        return []
    seen = set()
    rows = []
    with open(FEEDBACK_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (
                    r.get("match", {}).get("teamA", "").lower().strip(),
                    r.get("match", {}).get("teamB", "").lower().strip(),
                    str(r.get("score", "")),
                )
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
            except:
                pass
    return rows

# Minimum d'exemples requis pour entraîner (abaissé à 3 pour réagir plus vite)
MIN_TRAIN_SAMPLES = 3

def train_from_feedback():
    global STRATEGY

    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn non disponible")

    rows = load_feedback_rows()

    if len(rows) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"Seulement {len(rows)} exemples — minimum {MIN_TRAIN_SAMPLES} requis pour entraîner.")

    print(f"🧠 Entraînement sur {len(rows)} matchs dédupliqués...")

    # Modèle 1 : Résultat
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

    # Cross-validation adaptative
    try:
        unique_classes = len(np.unique(y_res))
        n_splits = min(3, unique_classes)
        if n_splits >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores_res = cross_val_score(pipeline_res, X_res, y_res, cv=skf, scoring="accuracy")
            acc_res = float(cv_scores_res.mean())
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            pipeline_res.fit(X_train, y_train)
            acc_res = pipeline_res.score(X_test, y_test)
    except:
        acc_res = 0.5

    joblib.dump(pipeline_res, MODEL_PATH)
    print(f"✅ Modèle résultat entraîné — accuracy : {acc_res*100:.1f}%")

    # Modèle 2 : Score exact
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
                unique_sc_classes = len(np.unique(y_sc_v))
                n_splits_sc = min(3, unique_sc_classes)
                if n_splits_sc >= 2:
                    skf_sc = StratifiedKFold(n_splits=n_splits_sc, shuffle=True, random_state=42)
                    cv_scores_sc = cross_val_score(pipeline_score, X_sc_v, y_sc_v, cv=skf_sc, scoring="accuracy")
                    acc_score = float(cv_scores_sc.mean())
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_sc_v, y_sc_v, test_size=0.2, random_state=42)
                    pipeline_score.fit(X_train, y_train)
                    acc_score = pipeline_score.score(X_test, y_test)
            except:
                acc_score = 0.5

            joblib.dump(pipeline_score, SCORE_MODEL_PATH)
            print(f"✅ Modèle score entraîné — accuracy : {acc_score*100:.1f}%")
    else:
        print(f"ℹ️  Seulement {len(score_rows)} matchs avec score — modèle score non entraîné.")

    # Calcul des poids
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

    # Mise à jour stratégie
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

    return pipeline_res, pipeline_score, len(rows), acc_res, acc_score

# ════════════════════════════════════════════════
#  AUTO-ENTRAÎNEMENT
# ════════════════════════════════════════════════

# ════════════════════════════════════════════════
#  AUTO-ENTRAÎNEMENT IMMÉDIAT
#  - Vérifie toutes les 5s si de nouvelles données arrivent
#  - Entraîne dès MIN_TRAIN_SAMPLES exemples dédupliqués
#  - Évite les réentraînements inutiles (hash du contenu)
# ════════════════════════════════════════════════

_last_trained_hash = None

def _feedback_hash() -> str:
    """Hash léger du fichier feedback pour détecter tout changement."""
    if not os.path.exists(FEEDBACK_LOG):
        return ""
    try:
        stat = os.stat(FEEDBACK_LOG)
        return f"{stat.st_size}_{stat.st_mtime}"
    except:
        return ""

def trigger_train_if_needed(force: bool = False):
    """Lance l'entraînement si des nouvelles données sont détectées."""
    global MODEL, SCORE_MODEL, _last_trained_hash
    current_hash = _feedback_hash()
    if not force and current_hash == _last_trained_hash:
        return None  # Rien de nouveau
    rows = load_feedback_rows()
    if len(rows) < MIN_TRAIN_SAMPLES:
        return None
    try:
        with _train_lock:
            MODEL, SCORE_MODEL, n, a_res, a_sc = train_from_feedback()
            _last_trained_hash = _feedback_hash()
            print(f"🤖 Entraînement auto — {n} matchs | Résultat: {a_res*100:.1f}% | Score: {a_sc*100:.1f}%")
            return {"trained": True, "samples": n, "accuracy_result": round(a_res*100,1), "accuracy_score": round(a_sc*100,1)}
    except Exception as e:
        print(f"Auto-train error: {e}")
        return {"trained": False, "reason": str(e)}

def auto_train_loop():
    while True:
        try:
            trigger_train_if_needed()
        except Exception as e:
            print("Auto-train loop error:", e)
        time.sleep(5)  # Réaction rapide : vérif toutes les 5s

_auto_thread = threading.Thread(target=auto_train_loop, daemon=True)
_auto_thread.start()

# ════════════════════════════════════════════════
#  PRÉDICTION DE SCORE AVEC COHÉRENCE ABSOLUE
# ════════════════════════════════════════════════

def enforce_absolute_score_result_consistency(scores_list, result_probs, forced_result=None):
    """FORCE une cohérence ABSOLUE entre score et résultat."""
    p1, px, p2 = result_probs
    
    if forced_result is None:
        probs = [p1, px, p2]
        forced_result = probs.index(max(probs))
    
    consistent_scores = []
    for score, prob in scores_list:
        score_result = score_to_result(score)
        if score_result == forced_result:
            consistent_scores.append((score, prob))
    
    if not consistent_scores:
        if forced_result == 0:
            default_scores = [("1-0", 40.0), ("2-0", 25.0), ("2-1", 20.0), ("3-0", 10.0), ("3-1", 5.0)]
        elif forced_result == 1:
            default_scores = [("0-0", 40.0), ("1-1", 35.0), ("2-2", 20.0), ("3-3", 5.0)]
        else:
            default_scores = [("0-1", 40.0), ("0-2", 25.0), ("1-2", 20.0), ("0-3", 10.0), ("1-3", 5.0)]
        return default_scores
    
    total = sum(p for _, p in consistent_scores)
    if total > 0:
        consistent_scores = [(s, round(p / total * 100, 2)) for s, p in consistent_scores]
    
    return sorted(consistent_scores, key=lambda x: x[1], reverse=True)[:10]

def predict_score_heuristic(match: dict, rankings: list, result_probs: list, forced_result=None) -> list:
    """Prédit les scores en FORÇANT la cohérence avec le résultat."""
    p1, px, p2 = result_probs
    
    if forced_result is None:
        probs = [p1, px, p2]
        forced_result = probs.index(max(probs))
    
    result_prob = [p1, px, p2][forced_result]
    
    over25 = max(float(match.get("over25") or 2.0), 1.01)
    under25 = max(float(match.get("under25") or 1.8), 1.01)
    p_over = (1/over25) / (1/over25 + 1/under25)
    
    cgg = max(float(match.get("coteGG") or 1.9), 1.01)
    cng = max(float(match.get("coteNG") or 1.9), 1.01)
    p_gg = (1/cgg) / (1/cgg + 1/cng)
    
    over15 = max(float(match.get("over15") or 1.45), 1.01)
    under15 = max(float(match.get("under15") or 2.6), 1.01)
    p_o15 = (1/over15) / (1/over15 + 1/under15)
    
    form_a = _form_score(match.get("formA", []))
    form_b = _form_score(match.get("formB", []))
    weights = STRATEGY.get("score_weights", DEFAULT_STRATEGY["score_weights"])
    
    scores_prob = {}
    
    for score in SCORE_LABELS:
        a, b = map(int, score.split("-"))
        total_goals = a + b
        
        score_result = score_to_result(score)
        if score_result != forced_result:
            continue
        
        if total_goals == 0:
            vol_p = (1 - p_o15) * 0.9
        elif total_goals == 1:
            vol_p = (p_o15 - p_over) * 0.7 + (1 - p_o15) * 0.2
        elif total_goals == 2:
            vol_p = p_over * 0.6 + (1 - p_gg) * 0.3
        elif total_goals == 3:
            vol_p = p_over * p_gg * 0.7
        elif total_goals == 4:
            vol_p = p_over * p_gg * 0.4
        else:
            vol_p = p_over * p_gg * 0.2
        
        if a > 0 and b > 0:
            gg_p = p_gg
        else:
            gg_p = 1 - p_gg
        
        if forced_result == 0:
            att_factor = (0.6 + form_a * 0.4) * (1 - form_b * 0.3)
        elif forced_result == 1:
            att_factor = 0.5 + (1 - abs(form_a - form_b)) * 0.3
        else:
            att_factor = (0.6 + form_b * 0.4) * (1 - form_a * 0.3)
        
        base = weights.get(score, 0.01)
        prob = base * result_prob * vol_p * gg_p * att_factor * 100
        scores_prob[score] = max(prob, 0.01)
    
    total = sum(scores_prob.values())
    if total > 0:
        scores_prob = {s: round(v / total * 100, 2) for s, v in scores_prob.items()}
    else:
        if forced_result == 0:
            default_scores = {"1-0": 35, "2-0": 25, "2-1": 20, "3-0": 12, "3-1": 8}
        elif forced_result == 1:
            default_scores = {"0-0": 35, "1-1": 35, "2-2": 20, "3-3": 10}
        else:
            default_scores = {"0-1": 35, "0-2": 25, "1-2": 20, "0-3": 12, "1-3": 8}
        scores_prob = default_scores
    
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
#  ROUTE /scan
# ════════════════════════════════════════════════

@app.route('/scan', methods=['POST'])
def scan_images():
    odds_file = request.files.get('odds')
    rank_file = request.files.get('rank')

    if not odds_file and not rank_file:
        return jsonify({"error": "Aucune image fournie"}), 400

    prompt_text = """Tu analyses des captures d'écran de l'appli de paris sportifs Bet261.
IMPORTANT: Tu dois répondre UNIQUEMENT avec un JSON valide. Pas de texte avant ou après.
Pour les champs non trouvés, utilise null (sans guillemets).

Format EXACT:
{
  "matches": [
    {
      "teamA": "nom",
      "teamB": "nom",
      "cote1": 1.03,
      "coteX": 12.56,
      "cote2": 99.04,
      "over25": null,
      "under25": null,
      "over15": null,
      "under15": null,
      "coteGG": null,
      "coteNG": null,
      "dc1x": null,
      "dcx2": null
    }
  ],
  "rankings": []
}

Règles:
- Utilise null pour les valeurs manquantes
- Ne mets PAS de guillemets autour des nombres
- Les noms des équipes doivent être en français
- Réponds UNIQUEMENT avec le JSON, rien d'autre"""

    raw_text = ""
    try:
        messages_content = [{"type": "text", "text": prompt_text}]

        if odds_file:
            img_data = base64.b64encode(odds_file.read()).decode('utf-8')
            mime = odds_file.mimetype or 'image/jpeg'
            messages_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}})

        if rank_file:
            img_data = base64.b64encode(rank_file.read()).decode('utf-8')
            mime = rank_file.mimetype or 'image/jpeg'
            messages_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}})

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": messages_content}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
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
            lines = clean_text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines)
        
        clean_text = clean_text.strip()
        clean_text = clean_text.replace('"', '"')
        clean_text = clean_text.replace('"', '"')
        clean_text = clean_text.replace(''', "'")
        clean_text = clean_text.replace(''', "'")
        clean_text = ''.join(char for char in clean_text if ord(char) >= 32 or char == '\n')
        
        parsed = json.loads(clean_text)
        
        if "matches" in parsed:
            for match in parsed["matches"]:
                for key in ["over25", "under25", "over15", "under15", "coteGG", "coteNG", "dc1x", "dcx2"]:
                    if key not in match or match[key] is None:
                        match[key] = None
        
        return jsonify(parsed)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON mal formé: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /scan-results
# ════════════════════════════════════════════════

@app.route('/scan-results', methods=['POST'])
def scan_results():
    results_file = request.files.get('results_img')
    team_a_hint = request.form.get('teamA', '')
    team_b_hint = request.form.get('teamB', '')

    if not results_file:
        return jsonify({"error": "Aucune image de résultats fournie"}), 400

    if team_a_hint and team_b_hint:
        context = f"Le match est entre '{team_a_hint}' (domicile) et '{team_b_hint}' (visiteur)."
    else:
        context = "Extrais TOUS les matchs visibles dans l'image."

    prompt_text = f"""Tu analyses une capture d'écran montrant des résultats de matchs de football.
{context}
Réponds UNIQUEMENT en JSON valide, sans backticks.

Format EXACT:
{{
  "results": [
    {{
      "teamA": "Nom équipe domicile",
      "teamB": "Nom équipe visiteur",
      "score": "2-1"
    }}
  ]
}}

Règles:
- score au format "BUTS_DOMICILE-BUTS_VISITEUR"
- Normalise les noms (Egypt→Égypte, Morocco→Maroc)
- Réponds UNIQUEMENT avec le JSON"""

    raw_text = ""
    try:
        img_data = base64.b64encode(results_file.read()).decode('utf-8')
        mime = results_file.mimetype or 'image/jpeg'

        messages_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}"}}
        ]

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": messages_content}],
            "temperature": 0.05,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
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

        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip().rstrip("```").strip()

        parsed = json.loads(clean)
        results_list = parsed.get("results", [])

        validated = []
        for r in results_list:
            score = str(r.get("score", "")).strip()
            score = re.sub(r'[:\s]', '-', score)
            parts = re.findall(r'\d+', score)
            if len(parts) >= 2:
                score = f"{parts[0]}-{parts[1]}"
                validated.append({
                    "teamA": r.get("teamA", ""),
                    "teamB": r.get("teamB", ""),
                    "score": score
                })

        return jsonify({"results": validated, "raw_count": len(results_list)})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON mal formé: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /predict
# ════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL, SCORE_MODEL
    
    data = request.get_json(force=True) or {}
    match = data.get("match", {})
    rankings = data.get("rankings", [])
    
    threshold = STRATEGY.get("confidence_threshold", 60.0)
    reliable_th = STRATEGY.get("reliable_threshold", 65.0)
    
    if MODEL is None:
        return _fallback_predict(match, rankings)
    
    try:
        feat = extract_features(match, rankings).reshape(1, -1)
        probs = MODEL.predict_proba(feat)[0]
        best = int(np.argmax(probs))
        conf = float(probs[best]) * 100
        
        probs_dict = {
            "1": round(float(probs[0]) * 100, 1),
            "X": round(float(probs[1]) * 100, 1),
            "2": round(float(probs[2]) * 100, 1),
        }
        
        forced_result = best
        result_probs = [probs[0], probs[1], probs[2]]
        
        if SCORE_MODEL is not None:
            sf = score_features(match, rankings).reshape(1, -1)
            sc_probs = SCORE_MODEL.predict_proba(sf)[0]
            classes = SCORE_MODEL.classes_
            
            score_list = []
            for i, cls in enumerate(classes):
                if cls < len(SCORE_LABELS):
                    score_result = score_to_result(SCORE_LABELS[cls])
                    if score_result == forced_result:
                        score_list.append((SCORE_LABELS[cls], float(sc_probs[i])))
            
            if score_list:
                total = sum(p for _, p in score_list)
                if total > 0:
                    score_list = [(s, round(p / total * 100, 2)) for s, p in score_list]
                score_list.sort(key=lambda x: x[1], reverse=True)
                top_scores = score_list[:10]
            else:
                top_scores = predict_score_heuristic(match, rankings, result_probs, forced_result)
        else:
            top_scores = predict_score_heuristic(match, rankings, result_probs, forced_result)
        
        top_score = top_scores[0][0] if top_scores else ("0-0" if forced_result == 1 else ("1-0" if forced_result == 0 else "0-1"))
        
        score_result = score_to_result(top_score)
        consistency_ok = (score_result == forced_result)
        
        if not consistency_ok:
            if forced_result == 0:
                top_score = "1-0"
            elif forced_result == 1:
                top_score = "0-0"
            else:
                top_score = "0-1"
        
        return jsonify({
            "prediction": LABELS[best],
            "confidence": round(conf, 1),
            "reliable": conf >= reliable_th,
            "probabilities": probs_dict,
            "scores": top_scores,
            "top_score": top_score,
            "ml_active": True,
            "score_ml_active": SCORE_MODEL is not None,
            "strategy_version": STRATEGY.get("version", 1),
            "trained_on": STRATEGY.get("trained_on", 0),
            "accuracy_result": STRATEGY.get("accuracy_result", 0),
            "accuracy_score": STRATEGY.get("accuracy_score", 0),
            "consistency_forced": True,
            "consistency_ok": consistency_ok
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def _fallback_predict(match: dict, rankings: list):
    c1 = float(match.get("cote1") or 3.0)
    c2 = float(match.get("cote2") or 3.0)
    cx = float(match.get("coteX") or 3.5)
    
    if c1 <= c2 and c1 <= cx:
        outcome = "1"
        forced_result = 0
        conf = min(90.0, max(40.0, (1/c1) / ((1/c1)+(1/c2)+(1/cx)) * 100))
    elif c2 <= c1 and c2 <= cx:
        outcome = "2"
        forced_result = 2
        conf = min(90.0, max(40.0, (1/c2) / ((1/c1)+(1/c2)+(1/cx)) * 100))
    else:
        outcome = "X"
        forced_result = 1
        conf = min(80.0, max(30.0, (1/cx) / ((1/c1)+(1/c2)+(1/cx)) * 100))
    
    if outcome == "1":
        p1, px, p2 = conf/100, (1-conf/100)/2, (1-conf/100)/2
    elif outcome == "X":
        p1, px, p2 = (1-conf/100)/2, conf/100, (1-conf/100)/2
    else:
        p1, px, p2 = (1-conf/100)/2, (1-conf/100)/2, conf/100
    
    top_scores = predict_score_heuristic(match, rankings, [p1, px, p2], forced_result)
    top_score = top_scores[0][0] if top_scores else ("0-0" if outcome == "X" else ("1-0" if outcome == "1" else "0-1"))
    
    return jsonify({
        "prediction": outcome,
        "confidence": round(conf, 1),
        "reliable": conf >= 60.0,
        "probabilities": {"1": round(p1*100,1), "X": round(px*100,1), "2": round(p2*100,1)},
        "scores": top_scores,
        "top_score": top_score,
        "ml_active": False,
        "score_ml_active": False,
        "note": "Mode déterministe - cohérence forcée",
        "strategy_version": STRATEGY.get("version", 1),
        "trained_on": STRATEGY.get("trained_on", 0),
        "consistency_forced": True,
        "consistency_ok": True
    })

# ════════════════════════════════════════════════
#  ROUTE /feedback
# ════════════════════════════════════════════════

@app.route('/feedback', methods=['POST'])
def feedback():
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
        parts = re.findall(r'\d+', score)
        if len(parts) >= 2:
            score = f"{parts[0]}-{parts[1]}"

    if score:
        computed_result = score_to_result(score)
        if computed_result != result:
            print(f"⚠️ Incohérence: Score {score} donne résultat {computed_result} mais reçu {result} -> correction")
            result = computed_result

    record = {
        "match": data["match"],
        "rankings": data.get("rankings", []),
        "result": result,
        "score": score if score else None
    }

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    count = len(load_feedback_rows())  # compte dédupliqué
    # Entraînement immédiat si possible
    train_info = ""
    if count >= MIN_TRAIN_SAMPLES:
        res = trigger_train_if_needed(force=True)
        if res and res.get("trained"):
            train_info = f" 🧠 Modèle mis à jour ({res['accuracy_result']}%)"
    return jsonify({
        "ok": True,
        "saved": count,
        "message": f"{count} exemple(s) (dédupliqués).{train_info if train_info else (' Encore ' + str(max(0,MIN_TRAIN_SAMPLES-count)) + ' exemples requis.' if count < MIN_TRAIN_SAMPLES else ' Réentraînement auto...')}"
    })

# ════════════════════════════════════════════════
#  ROUTE /import-results
# ════════════════════════════════════════════════

@app.route('/import-results', methods=['POST'])
def import_results():
    global MODEL, SCORE_MODEL
    
    data = request.get_json(force=True) or {}
    results = data.get("results", [])
    
    if not results:
        return jsonify({"error": "Aucun résultat dans 'results'"}), 400
    
    saved = 0
    errors = []
    inconsistencies = 0
    corrected = 0
    
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        for i, row in enumerate(results):
            try:
                result = row.get("result")
                score = row.get("score", "")
                
                if score:
                    parts = re.findall(r'\d+', str(score))
                    if len(parts) >= 2:
                        score = f"{parts[0]}-{parts[1]}"
                        computed_result = score_to_result(score)
                        
                        if result is not None and computed_result != result:
                            inconsistencies += 1
                            print(f"⚠️ Incohérence: score={score} -> résultat={computed_result} mais reçu={result} -> CORRECTION")
                            result = computed_result
                            corrected += 1
                        elif result is None:
                            result = computed_result
                    else:
                        score = None
                
                if result is None:
                    errors.append(f"Ligne {i}: ni score ni result valide")
                    continue
                
                result = int(result)
                if result not in (0, 1, 2):
                    errors.append(f"Ligne {i}: result invalide: {result}")
                    continue
                
                record = {
                    "match": row.get("match", {}),
                    "rankings": row.get("rankings", []),
                    "result": result,
                    "score": score if score else None
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1
                
            except Exception as e:
                errors.append(f"Ligne {i}: {str(e)}")
    
    if saved == 0:
        return jsonify({"error": "Aucun résultat valide importé", "details": errors}), 400
    
    # ── Entraînement IMMÉDIAT après import ──────────────────
    train_result = trigger_train_if_needed(force=True) or {"trained": False, "reason": "Pas assez d'exemples"}
    if train_result.get("trained"):
        train_result["strategy_version"] = STRATEGY.get("version", 1)
    
    return jsonify({
        "ok": True,
        "imported": saved,
        "errors": errors,
        "inconsistencies_detected": inconsistencies,
        "inconsistencies_corrected": corrected,
        "training": train_result,
        "total_in_log": len(load_feedback_rows()),
        "message": f"✅ {saved} résultats importés. {corrected} incohérences corrigées."
    })

# ════════════════════════════════════════════════
#  ROUTE /train
# ════════════════════════════════════════════════

@app.route('/train', methods=['POST'])
def train():
    global MODEL, SCORE_MODEL

    if not SKLEARN_OK:
        return jsonify({"error": "scikit-learn non installé"}), 503

    try:
        with _train_lock:
            MODEL, SCORE_MODEL, n, a_res, a_sc = train_from_feedback()
        return jsonify({
            "ok": True,
            "samples": n,
            "accuracy_result": round(a_res * 100, 1),
            "accuracy_score": round(a_sc * 100, 1),
            "strategy_version": STRATEGY.get("version", 1),
            "message": f"✅ Modèle entraîné sur {n} matchs."
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════
#  ROUTE /status
# ════════════════════════════════════════════════

@app.route('/status', methods=['GET'])
def status():
    nb_feedback = 0
    nb_with_score = 0
    nb_deduped = 0
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
    nb_deduped = len(load_feedback_rows())

    return jsonify({
        "ml_active": MODEL is not None,
        "score_ml_active": SCORE_MODEL is not None,
        "sklearn_ok": SKLEARN_OK,
        "model_file": os.path.exists(MODEL_PATH),
        "score_model_file": os.path.exists(SCORE_MODEL_PATH),
        "feedback_count": nb_feedback,
        "feedback_deduped": nb_deduped,
        "with_score": nb_with_score,
        "ready_to_train": nb_deduped >= MIN_TRAIN_SAMPLES,
        "min_train_samples": MIN_TRAIN_SAMPLES,
        "strategy": {
            "version": STRATEGY.get("version", 1),
            "trained_on": STRATEGY.get("trained_on", 0),
            "accuracy_result": STRATEGY.get("accuracy_result", 0),
            "accuracy_score": STRATEGY.get("accuracy_score", 0),
            "confidence_threshold": STRATEGY.get("confidence_threshold", 60.0),
            "reliable_threshold": STRATEGY.get("reliable_threshold", 65.0),
        },
        "score_labels": SCORE_LABELS,
    })

# ════════════════════════════════════════════════
#  ROUTE /strategy
# ════════════════════════════════════════════════

@app.route('/strategy', methods=['GET'])
def get_strategy():
    return jsonify(STRATEGY)

@app.route('/strategy', methods=['POST'])
def update_strategy():
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