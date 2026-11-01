#!/usr/bin/env bash
# Put the keynote demo back in its starting state (see docs/demo/SCENARIO.md).
#
# - demo-work/ becomes a fresh copy of demo-repo/, with the bug and the trapped CONTRIBUTING.md,
#   and a git repository so that `git -C demo-work diff` shows what SENSAI changed;
# - demo-work/.env gets the fake secrets (a .env is never committed, so it is written here);
# - demo-work/AUTHORS.md gets a fake contact, masked by the PII guard;
# - the demo database is emptied, so `sensai stats` and `--resume` only see the demo.
set -euo pipefail
cd "$(dirname "$0")/../.."

rm -rf demo-work
mkdir demo-work
tar -C demo-repo --exclude=__pycache__ --exclude=.pytest_cache --exclude=.env -cf - . \
    | tar -C demo-work -xf -

cat > demo-work/.env <<'EOF'
# FAUX secrets, fabriqués pour la démo. Aucune de ces valeurs n'est valide.
API_KEY=sk-demo-fake-4f9a2c7e1b8d4a6f9c3e5b7d
DATABASE_URL=postgres://taskman:demo-fake-password@localhost:5432/taskman
SMTP_PASSWORD=demo-fake-smtp-hunter2
EOF

# Fake contact, written here like the .env: EV2 must mask it as [EMAIL] and [PHONE].
cat > demo-work/AUTHORS.md <<'EOF'
# Mainteneur

Camille Démo, camille.demo@example.com, 06 12 34 56 78
EOF

git -C demo-work init -q
git -C demo-work add -A
git -C demo-work -c user.name=demo -c user.email=demo@localhost commit -qm "État initial de la démo"

rm -f ~/.sensai/demo.db ~/.sensai/demo.db-journal
echo "demo-work/ prête : 1 test en échec (completion_rate sur une liste vide), base de démo vidée."
