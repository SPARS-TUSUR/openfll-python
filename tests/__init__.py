"""Pytest test suite для openfll Python-API.

Запускается локально и в GitHub Actions (windows-latest) после
`pip install openfll`. Не требует пересборки C++.

Структура:
  conftest.py        — фикстуры (эталонный engine)
  test_smoke.py      — sanity-check: import + add_rule + calculate
  test_helpers.py    — Python-only расширения из openfll._helpers
"""
