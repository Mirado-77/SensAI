# EV3 Logs structurés et métriques

## Story 1

En tant que développeuse qui présente SENSAI en démo, je veux voir en une commande combien coûte
un tour (durée, temps jusqu'au premier token, tokens), afin d'expliquer d'où vient la latence et
de vérifier qu'un changement de modèle ou de prompt ne la dégrade pas.

Critères d'acceptation :
- `sensai stats` affiche le nombre de tours par statut (terminés, interrompus, échoués), les
  tokens au total et par tour, avec le détail prompt et réponse.
- La durée du tour et le temps jusqu'au premier token sont donnés en moyenne, p50, p95 et max,
  calculés sur les tours terminés seulement (un tour échoué ou interrompu fausserait la mesure).
- `--since 30m`, `2h`, `1d`... limite les chiffres aux tours démarrés dans cette période ; une
  période invalide est refusée avec un message et le code 2.
- Sans aucun tour enregistré, la commande affiche « No run recorded yet. » et rend 0.
- `--json` donne les mêmes chiffres en JSON sur stdout, pour un autre outil.

## Story 2

En tant que développeur qui reprend une session qui s'est mal passée, je veux voir quels outils
ont été appelés, lesquels ont échoué ou ont été refusés, et les dernières erreurs, afin de savoir
quoi corriger sans relire tout le journal.

Critères d'acceptation :
- Un tableau par outil donne les appels, les succès, les erreurs, les refus et la durée moyenne
  en millisecondes ; un refus de l'utilisateur (`denied`) n'est pas compté comme une erreur.
- Les dernières erreurs (`run.failed` et `tool.failed` hors refus) sont listées de la plus
  récente à la plus ancienne, avec la date, le type et le message ; `--errors N` règle leur
  nombre (5 par défaut).
- Les blocages et masquages des garde-fous (EV2, `guard.blocked` et `guard.redacted`) sont
  comptés.
- Une base de données inutilisable est signalée sur stderr avec le code 2.
- Tout est lu dans le journal d'événements existant (`runs`, `tool_calls`, `events`) : aucune
  migration, aucun changement de la boucle.

## Tests

Les tests portent l'ID de la fonctionnalité avec `@pytest.mark.feature("EV3")`.

- `tests/unit/test_stats.py` (tours par statut et tokens, durée et premier token sur les tours
  terminés, percentile, outils avec erreurs et refus, erreurs récentes sans les refus, garde-fous,
  filtre `--since`, journal vide, commande `sensai stats` après un vrai chat, sortie JSON, base
  inutilisable, période invalide)
