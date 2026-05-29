#!/usr/bin/env bash
# Lance le scraper EuroMillions dans un venv Python isolé.
# Usage :
#   ./run.sh            # utilise le cache si les données sont déjà téléchargées
#   ./run.sh --refresh  # force le re-téléchargement des données

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

# ── 3. Lancer le script ──────────────────────────────────────────────────────
echo "[3/3] Lancement du script EuroMillions ..."
echo ""
python "$SCRIPT_DIR/euromillions_scraper.py" "$@"
