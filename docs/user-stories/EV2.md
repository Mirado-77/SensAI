# EV2 Garde-fous contre l'injection et les fuites de secrets

## Story 1

En tant que développeuse qui laisse l'agent lire son dépôt, je veux qu'une instruction cachée
dans le message que j'envoie ou dans un fichier lu ne fasse pas changer de comportement à l'agent,
afin qu'un fichier piégé ne puisse pas le détourner.

Critères d'acceptation :
- Un message utilisateur qui tente de contourner les instructions (par exemple « ignore all
  previous instructions ») est refusé avant tout appel au modèle : le tour se termine en
  `run.failed` avec le message « injection de prompt détectée », et rien n'est enregistré dans la
  session.
- Un résultat d'outil qui contient une telle tentative (fichier lu, sortie MCP, sortie du sandbox)
  est remplacé, avant d'atteindre le modèle, par un message d'erreur qui indique que la sortie de
  l'outil a été bloquée. Le texte piégé n'apparaît ni dans le contexte du modèle ni dans
  `tool.finished`.
- Chaque blocage émet un événement `guard.blocked` avec `stage` (`entry` ou `tool`), `reason` et,
  pour un outil, `tool`.
- Les espaces multiples, les retours à la ligne et les caractères invisibles ne permettent pas de
  contourner la détection d'une phrase.

## Story 2

En tant que développeuse qui a un fichier `.env` dans son dépôt, je veux que ses valeurs ne
sortent jamais dans les réponses de l'agent ni dans le contexte du modèle, afin que mes clés
restent secrètes.

Critères d'acceptation :
- Les valeurs du fichier `[guards] env_file` (valeurs d'au moins 8 caractères) sont remplacées par
  `[REDACTED]` dans les sorties d'outils, avant que le modèle ne les reçoive.
- Les mêmes valeurs sont remplacées dans la réponse finale, qui est aussi celle qui est
  enregistrée dans la session.
- Les formats de clés connus (`sk-…`, `AKIA…`, `ghp_…`) sont remplacés même sans fichier `.env`.
- Chaque remplacement émet `guard.redacted` avec `stage` (`answer` ou `tool`), `count` et, pour un
  outil, `tool`.
- `[guards] enabled = false` désactive les deux garde-fous, sans toucher au code.

## Limites connues

- La détection d'injection repose sur une liste de formulations (français et anglais). Elle ne
  couvre pas toutes les formulations possibles et peut refuser un texte légitime qui les contient
  (par exemple un fichier du projet qui documente ces phrases).
- Le texte streamé vers la console est affiché au fil de l'eau : la rédaction s'applique à la
  réponse finale, pas aux morceaux déjà affichés.

## Tests

- `tests/unit/test_ev2_guards.py` : parcours de bout en bout sur le runtime (entrée, sortie
  d'outil, réponse, désactivation).
- `tests/unit/test_safety_injection.py` : détection des formulations et contournements.
- `tests/unit/test_safety_secrets.py` : lecture de `.env` et rédaction.
