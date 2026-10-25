# EV1 Juge LLM hors ligne sur un corpus de questions

## Story 1

En tant que développeur qui modifie le prompt ou le modèle, je veux faire répondre l'agent à un
corpus de questions sur le dépôt de démo et faire noter chaque réponse par un juge LLM, afin de
savoir en une commande si la qualité des réponses a baissé.

Critères d'acceptation :
- `make eval-quality` lit les questions de `evals/questions/*.toml` (un fichier par question, avec
  `id`, `question`, `reference` et `criteria`). Une question avec un champ inconnu ou un TOML
  invalide est signalée, et la commande s'arrête avec le code 2 sans appeler le modèle.
- Chaque question est posée dans une nouvelle session, sur une copie en lecture seule de
  `demo-repo/`, avec la version de prompt active (A4) et l'outil `search_code` (R1) quand la copie
  a pu être indexée. Sinon, la raison est écrite dans le rapport et l'évaluation continue.
- Le juge est un appel de modèle séparé, en sortie structurée (T5) : il reçoit la question, la
  référence, les critères et la réponse, et renvoie un verdict (`correct`, `partial`,
  `incorrect`), un score de 1 à 5 et une raison d'une phrase.
- Une réponse vide est notée `incorrect` sans appeler le juge. Si le juge ne produit pas de note
  valide après la réparation, la réponse est marquée « non notée » au lieu de faire échouer le
  lancement.

## Story 2

En tant que développeur qui compare deux versions de prompt, je veux un rapport JSON et Markdown
horodaté qui indique la version évaluée et le score moyen, afin de comparer deux lancements sans
relancer le modèle (A7 s'appuie dessus).

Critères d'acceptation :
- Chaque lancement écrit `ev1-<horodatage>.json` et `ev1-<horodatage>.md` dans `evals/reports/`.
- Le rapport indique le modèle, le modèle du juge, la version de prompt `<nom>@<sha>`, le nombre
  de réponses par verdict, les réponses non notées et le score moyen.
- Pour chaque question, le rapport donne le verdict, le score, la raison du juge, les outils
  appelés, un extrait de la réponse et, si le tour a échoué, son erreur (délai, limite d'étapes).
- `--prompt <version>` évalue une autre version de prompt, `--judge-model <nom>` change le modèle
  du juge, `--only <id>` ne lance qu'une question et `--no-index` lance sans `search_code`.

## Tests

Les tests portent l'ID de la fonctionnalité avec `@pytest.mark.feature("EV1")`.

- `tests/unit/test_ev1.py` : format du corpus, appel du juge, réponse vide, juge en échec,
  version de prompt envoyée, notes de configuration, progression, rapport Markdown (avec l'erreur
  d'un tour échoué) et code de sortie 2.
