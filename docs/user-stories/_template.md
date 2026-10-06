# <ID> <Feature name>

Each feature has at least two user stories. Acceptance criteria are observable behaviors: what
a user sees or what a test can assert.

## Story 1

As a <type of user>, I want to <action>, so that <outcome>.

Acceptance criteria:
- <Observable behavior 1>
- <Observable behavior 2>
- <Observable behavior 3>

## Story 2

As a <type of user>, I want to <action>, so that <outcome>.

Acceptance criteria:
- <Observable behavior 1>
- <Observable behavior 2>

## Tests

Tests carry the feature ID with `@pytest.mark.feature("<ID>")`.

- `tests/<path>::<test name>`
