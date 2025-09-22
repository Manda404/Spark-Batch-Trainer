from datetime import date

today = date.today().strftime("%Y-%m-%d")

template = f"""
Changelog
=========

v1.0.0 ({today})
-----------------
- Première release publique
- Support des frameworks XGBoost, CatBoost et LightGBM
- Support complet du batch training avec Spark DataFrames
"""

with open("docs/changelog.rst", "w", encoding="utf-8") as f:
    f.write(template)
