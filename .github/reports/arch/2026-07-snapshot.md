# Architecture Snapshot

_Generated: 2026-07-01_

**30 files** | **1,034 Python LOC**

## Directory Tree

```
ai-data-analyst-agent/
├── data/
│   └── sample/
│       ├── iris.csv
│       ├── sales.csv
│       └── titanic.csv
├── plots/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── analyst_agent.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py
│   │   └── pattern_detector.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── task_inference.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plot_generator.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_eda.py
│   ├── test_pattern_detector.py
│   └── test_task_inference.py
├── .gitignore
├── config.yaml
├── README.md
└── requirements.txt
```

## Module Size (Python LOC)

| Module | LOC |
|--------|-----|
| `src` | 838 |
| `tests` | 196 |

## Imports

| Module | Uses |
|--------|------|
| `src` | `argparse`, `datetime`, `logging`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, ... (+5) |
| `tests` | `numpy`, `pandas`, `pytest`, `src` |

## File types

| Extension | Files |
|-----------|------:|
| `.py` | 21 |
| `(none)` | 3 |
| `.csv` | 3 |
| `.md` | 1 |
| `.txt` | 1 |
| `.yaml` | 1 |

---