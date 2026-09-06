# OPFython conventions

These conventions adapt cpmux's phitrain-derived code style to OPFython. Keep the existing scientific API and
domain structure rather than copying cpmux's application architecture, dependencies, or license.

## Compatibility and ownership

- Preserve public module paths, reexports, class identities, runtime arguments, and list/tuple/`None` returns.
- Keep Python 3.11 support until an explicit support-policy change. The requested modern annotation syntax works
  on 3.11. Do not introduce 3.12-only syntax merely to match another project's interpreter floor.
- Keep the Apache 2.0 license. Every Python file starts with these two lines, followed by a blank line:

  ```python
  # Copyright (c) 2020-2026 Gustavo de Rosa.
  # Licensed under the Apache License, Version 2.0.
  ```

- Treat numerical inputs as borrowed storage. Numerical guards must not modify caller arrays or their views.
  Document intentional mutation, including the training/validation exchanges performed by `learn()`.
- Keep seeded split ordering stable while isolating splitter randomness from NumPy's global generator.
- Preserve the custom exception hierarchy and first-use console/file logging policy. Do not replace them with
  framework conventions as an incidental cleanup.
- Keep persisted model identities and state compatible with the supported release. Internal reuse must not change
  class bases, callback order, heap ties, label propagation, or prediction return types.

## Python style

- Use `X | None`, not `Optional[X]`, and builtin generics such as `list[int]` and `dict[str, Any]`.
- Import ABCs such as `Callable` and `Iterable` from `collections.abc`. Import only necessary typing-specific names
  such as `Any`, `Literal`, and `TypeVar` from `typing`.
- Imports are top-level and absolute. Separate stdlib, third-party, and OPFython imports with blank lines.
  Internal consumers import defining modules. Existing package reexports remain part of the public API.
- Use double quotes. Keep code and readable prose within 120 characters.
- Use `if`/`raise` with a specific exception for validation, never `assert`. Bare `except:` is forbidden.
- Comments explain why, not what. Prefer no comment or one line, with a three-line maximum.
  Do not add banner/section separators or trailing periods to ordinary comments.
- Insert one blank line at each phase transition in function bodies with at least 12 lines of code.
  Separate validation from committing state and group constructor fields by responsibility.
- Inline first. Extract a helper, constant, or parameter when a second call site establishes a shared responsibility.

## Documentation

- Public functions, classes, and their explicit constructors use Google-style docstrings.
- Use a single-sentence summary. A regular class has only its class summary, and constructor arguments are documented
  on `__init__`. Keep scientific references and useful algorithm notes in the appropriate public method or guide.
- Private helpers and framework-dispatched hooks have no docstrings. Document properties on their getters rather
  than duplicating their contracts on descriptor setters.
- Keep short summaries on one line, matching cpmux and Black. Multiline docstrings keep one blank line before
  their closing `"""`. Keep one blank line after every docstring before the next statement or field.
- Use one line per `Args:`, `Returns:`, and `Raises:` entry. Do not add semicolons or `defaults to <X>` tails.
- State shapes, label/index meanings, return ordering, mutation, persistence, ownership, and failure modes where
  relevant. An annotation is not runtime validation and does not establish an array's shape.
- Data classes without explicit constructors document every field in `Attributes:`, one line per field.
- Preserve useful references and examples. Do not replace explanatory contracts with summaries merely to reduce size.
- Sphinx/Napoleon remains the documentation stack, and generated class documentation includes constructor contracts.

## Diagnostics and errors

- Library modules use `get_logger(__name__)` from `opfython.utils.logging`, never `print()`.
  Example scripts are a presentation surface and may print their results without introducing CLI dependencies.
- `logger.warning` and `logger.error` identify a backticked offender and end with a period:
  ``f"`name=value` <verb-phrase>."``. Info/debug messages remain plain.
- Raised messages identify a backticked name and end with a period:
  ``f"`name` <verb-phrase>[, but got <value>]."``. Use `is None`/`is True` prose.
- Avoid dumping feature arrays into diagnostics. Report the relevant type, shape, index, or configuration value.
- Exception constructors preserve caller-supplied messages. Format messages at library raise sites rather than
  silently rewriting messages supplied by applications.

## Tests and delivery

- Keep the existing test layout. Test functions are plain functions without docstrings or type annotations.
- Name tests after the behavior under test. Use bare assertions without redundant failure-message strings.
- Add focused regression cases for intentional behavior changes, and retain existing meaningful assertions.
- Check writable/read-only/view inputs, seeded ordering, public failure contracts, and fitted-model persistence
  where the affected operation makes those cases relevant.
- Use the existing pytest, Black, isort, Flake8, and Sphinx tools. Do not add a framework or tool migration solely
  to imitate another repository.
- Run quality hooks and documentation builds in CI. A passing formatter does not establish logical grouping or
  documentation completeness, so review those explicitly.
