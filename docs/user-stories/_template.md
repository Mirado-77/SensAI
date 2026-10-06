# <ID> <Nom de la fonctionnalité>

Chaque fonctionnalité a au moins deux user stories. Les critères d'acceptation sont des
comportements observables : ce qu'un utilisateur voit ou ce qu'un test peut vérifier.

## Story 1

En tant que <type d'utilisateur>, je veux <action>, afin de <résultat>.

Critères d'acceptation :
- <Comportement observable 1>
- <Comportement observable 2>
- <Comportement observable 3>

## Story 2

En tant que <type d'utilisateur>, je veux <action>, afin de <résultat>.

Critères d'acceptation :
- <Comportement observable 1>
- <Comportement observable 2>

## Tests

Les tests portent l'ID de la fonctionnalité avec `@pytest.mark.feature("<ID>")`.

- `tests/<chemin>::<nom du test>`
