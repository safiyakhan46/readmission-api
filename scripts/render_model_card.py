"""Fill the metrics and subgroup tables in MODEL_CARD.md from models/metadata.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_block(text: str, tag: str, body: str) -> str:
    return re.sub(rf"<!-- {tag} -->.*?<!-- /{tag} -->", f"<!-- {tag} -->\n{body}\n<!-- /{tag} -->",
                  text, flags=re.S)


def main():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    m = meta["test_metrics"]
    metrics = "\n".join([
        f"Model version `{meta['model_version']}`, trained on `{meta['data']}` "
        f"(train {meta['n_train']:,} / val {meta['n_val']:,} / test {meta['n_test']:,} encounters).",
        "",
        "| Metric | Value |", "|---|---|",
        f"| ROC-AUC (95% CI) | {m['roc_auc']:.3f} ({m['roc_auc_ci_low']:.3f}–{m['roc_auc_ci_high']:.3f}) |",
        f"| PR-AUC | {m['pr_auc']:.3f} (prevalence {m['prevalence']:.3f}) |",
        f"| Brier score | {m['brier']:.3f} |",
        f"| Threshold | {meta['threshold']:.2f} |",
        f"| Precision / Recall / F1 at threshold | {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f} |",
    ])
    rows = ["| Group | n | Prevalence | ROC-AUC | Recall |", "|---|---|---|---|---|"]
    for g, s in meta.get("subgroup_metrics", {}).items():
        rows.append(f"| {g} | {s['n']:,} | {s['prevalence']:.3f} | {s['roc_auc']:.3f} | {s['recall']:.3f} |")
    card = ROOT / "MODEL_CARD.md"
    text = replace_block(card.read_text(), "METRICS", metrics)
    text = replace_block(text, "SUBGROUPS", "\n".join(rows))
    card.write_text(text)
    print("MODEL_CARD.md updated")


if __name__ == "__main__":
    main()
