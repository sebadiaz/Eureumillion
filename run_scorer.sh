#!/usr/bin/env bash
# Lance le modèle de scoring EuroMillions dans le venv Python.
# Usage :
#   ./run_scorer.sh                         # entraîne + backtest + graphiques
#   ./run_scorer.sh --score 5 14 23 42 49 2 8   # score une sélection
#   ./run_scorer.sh --train-ratio 0.7 --no-plots # options avancées

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── 1. Créer le venv si nécessaire ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Création du venv Python dans .venv/ ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] Venv existant détecté (.venv/)"
fi

# ── 2. Activer et installer les dépendances ──────────────────────────────────
source "$VENV_DIR/bin/activate"

echo "[2/3] Installation / mise à jour des dépendances ..."
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── 3. Lancer le modèle de scoring ──────────────────────────────────────────
echo "[3/3] Lancement du modèle de scoring ..."
echo ""
python "$SCRIPT_DIR/scoring_model.py" "$@"
