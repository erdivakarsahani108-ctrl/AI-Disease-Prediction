"""
scripts/generate_notebooks.py
===============================
Programmatically builds the 4 required Jupyter notebooks
(notebooks/EDA.ipynb, preprocessing.ipynb, model_training.ipynb,
evaluation.ipynb) using nbformat, then executes them end-to-end with
nbclient so the committed notebooks contain real, reproducible outputs
(tables, metrics, interactive Plotly charts) rather than empty cells.

Run:
    python scripts/generate_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = ROOT / "notebooks"


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SETUP_CELL = (
    "import sys, pathlib\n"
    "ROOT = pathlib.Path.cwd().parent if (pathlib.Path.cwd() / '..' / 'src').exists() else pathlib.Path.cwd()\n"
    "sys.path.insert(0, str(ROOT))\n"
    "sys.path.insert(0, str(ROOT / 'scripts'))\n"
    "import pandas as pd\n"
    "pd.set_option('display.max_columns', 20)\n"
)


def build_eda_notebook():
    cells = [
        md(
            "# Exploratory Data Analysis (EDA)\n\n"
            "AI-Based Disease Prediction & Intelligent Health Assistant\n\n"
            "This notebook explores the processed `data/processed/disease_dataset.csv` "
            "dataset: shape, class distribution, symptom frequency, disease-symptom "
            "relationships and class imbalance, using the same chart functions used "
            "by the Streamlit Data Science Dashboard (`src/visualization/eda_charts.py`)."
        ),
        code(SETUP_CELL),
        code(
            "from src.ml.data_loader import load_dataset, get_feature_columns\n"
            "from src.visualization import eda_charts\n\n"
            "df = load_dataset()\n"
            "feature_cols = get_feature_columns(df)\n"
            "print('Shape:', df.shape)\n"
            "print('Diseases:', df['disease'].nunique())\n"
            "print('Symptom features:', len(feature_cols))\n"
            "print('Missing values:', df.isna().sum().sum())\n"
            "print('Duplicate rows:', df.duplicated().sum())\n"
        ),
        md("## Disease Distribution"),
        code("eda_charts.disease_distribution_chart(df)"),
        md("## Disease Category Distribution"),
        code("eda_charts.category_distribution_chart(df)"),
        md("## Most Frequent Symptoms"),
        code("eda_charts.symptom_frequency_chart(df)"),
        md("## Disease vs Symptom Heatmap"),
        code("eda_charts.disease_symptom_heatmap(df)"),
        md("## Class Imbalance"),
        code("eda_charts.class_imbalance_chart(df)\ndf['disease'].value_counts().describe()"),
        md("## Real vs Synthetic Records"),
        code("eda_charts.real_vs_synthetic_chart(df)"),
        md("## Symptoms-per-Record Distribution"),
        code("eda_charts.symptom_count_distribution(df)"),
        md(
            "## Summary\n\n"
            "- The dataset combines a small set of REAL disease-symptom combinations "
            "from a public source with documented SYNTHETIC augmentation (see "
            "`docs/dataset_methodology.md`).\n"
            "- Class imbalance is real and expected (see `docs/data_quality_report.md`) "
            "because different diseases have very different numbers of validated real "
            "symptom combinations - this is handled downstream via `class_weight='balanced'` "
            "and macro-averaged evaluation metrics."
        ),
    ]
    return _nb(cells)


def build_preprocessing_notebook():
    cells = [
        md(
            "# Data Preprocessing Pipeline\n\n"
            "Runs the full reproducible pipeline: Raw Data -> Cleaning -> Integration -> "
            "Deduplication -> Symptom Standardization -> Synthetic Augmentation -> "
            "Feature Engineering -> Validation -> Final Dataset.\n\n"
            "This notebook calls the SAME functions used by `scripts/prepare_data.py` "
            "(no duplicated logic), so the notebook and the CLI script always stay "
            "in sync."
        ),
        code(SETUP_CELL),
        code(
            "from src.preprocessing.clean import (\n"
            "    load_symptom_vocabulary, load_real_disease_symptom_sets, build_disease_symptom_pool\n"
            ")\n\n"
            "vocab_df = load_symptom_vocabulary()\n"
            "real_df = load_real_disease_symptom_sets()\n"
            "print('Symptom vocabulary size:', len(vocab_df))\n"
            "print('Unique real disease-symptom combinations:', len(real_df))\n"
            "print('Duplicate rows removed from raw source:', real_df.attrs.get('duplicate_rows_removed'))\n"
            "real_df.head()"
        ),
        md("## Disease -> Validated Real Symptom Pool (sample)"),
        code(
            "pool = build_disease_symptom_pool(real_df)\n"
            "for disease in list(pool)[:5]:\n"
            "    print(disease, '->', pool[disease])"
        ),
        md("## Synthetic Augmentation (subset resampling of validated real symptoms only)"),
        code(
            "from src.preprocessing.augment import generate_synthetic_records\n"
            "from src.utils.paths import RANDOM_SEED\n\n"
            "synth_df = generate_synthetic_records(real_df, pool, target_per_disease=50, seed=RANDOM_SEED)\n"
            "print('Synthetic records generated (demo target=50/disease):', len(synth_df))\n"
            "synth_df.head()"
        ),
        md(
            "## Full Pipeline Execution\n\n"
            "Runs the exact same `scripts/prepare_data.py` used to build the committed "
            "`data/processed/disease_dataset.csv` (target = 600 records/disease)."
        ),
        code("import prepare_data\nprepare_data.main()"),
    ]
    return _nb(cells)


def build_model_training_notebook():
    cells = [
        md(
            "# Model Training & Comparison\n\n"
            "Trains and compares 6 classical ML algorithms (Logistic Regression, "
            "Decision Tree, Random Forest, KNN, SVM, XGBoost) on the processed "
            "dataset, and selects the best model using a weighted composite of "
            "macro-F1, weighted-F1, macro-recall and cross-validated F1-macro "
            "(NOT accuracy alone).\n\n"
            "This notebook calls `scripts/train_model.py` directly so the notebook "
            "and CLI script never drift apart."
        ),
        code(SETUP_CELL),
        code("import train_model\ntrain_model.main()"),
        md("## Model Comparison Table"),
        code(
            "import pandas as pd\n"
            "comp = pd.read_csv(ROOT / 'models' / 'model_comparison.csv')\n"
            "comp"
        ),
        md("## Model Comparison Chart"),
        code(
            "from src.visualization import model_charts\n"
            "model_charts.model_comparison_bar_chart(comp)"
        ),
        code("model_charts.model_timing_chart(comp)"),
        md("## Best Model Metadata"),
        code(
            "import json\n"
            "metadata = json.loads((ROOT / 'models' / 'model_metadata.json').read_text())\n"
            "metadata"
        ),
    ]
    return _nb(cells)


def build_evaluation_notebook():
    cells = [
        md(
            "# Model Evaluation\n\n"
            "Loads the persisted best model and evaluates it on the held-out test "
            "split: per-class precision/recall/F1, confusion matrix, and a summary "
            "of the metric-selection rationale."
        ),
        code(SETUP_CELL),
        code("import evaluate_model\nevaluate_model.main()"),
        md("## Per-Class Classification Report"),
        code(
            "import pandas as pd\n"
            "report = pd.read_csv(ROOT / 'reports' / 'classification_report.csv', index_col=0)\n"
            "report"
        ),
        md("## Confusion Matrix (interactive)"),
        code(
            "from IPython.display import IFrame\n"
            "IFrame(str(ROOT / 'reports' / 'confusion_matrix.html'), width=900, height=900)"
        ),
        md(
            "## Notes\n\n"
            "Macro-averaged metrics are prioritized over raw accuracy because several "
            "diseases have far fewer real records than others (see "
            "`docs/data_quality_report.md`). See `docs/model_evaluation.md` and "
            "`docs/model_card.md` for the full write-up."
        ),
    ]
    return _nb(cells)


def execute_and_save(nb, path: Path):
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"Executed & saved -> {path}")


def main():
    execute_and_save(build_eda_notebook(), NOTEBOOKS_DIR / "EDA.ipynb")
    execute_and_save(build_preprocessing_notebook(), NOTEBOOKS_DIR / "preprocessing.ipynb")
    execute_and_save(build_model_training_notebook(), NOTEBOOKS_DIR / "model_training.ipynb")
    execute_and_save(build_evaluation_notebook(), NOTEBOOKS_DIR / "evaluation.ipynb")


if __name__ == "__main__":
    main()
