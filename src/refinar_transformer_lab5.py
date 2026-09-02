from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

import optimizar_transformer_lab5 as core


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PROJECT = SCRIPT_DIRECTORY.parent if SCRIPT_DIRECTORY.name == "src" else core.PROJECT
PROJECT = Path(os.environ.get("LAB5_PROJECT", str(DEFAULT_PROJECT)))
OUT = Path(os.environ.get("LAB5_OPT_OUT", str(PROJECT / "resultados")))
ARTIFACTS = Path(os.environ.get("LAB5_ARTIFACTS_OUT", str(PROJECT / "artefactos")))
ARTIFACTS.mkdir(parents=True, exist_ok=True)
PREVIOUS = json.loads((OUT / "resultados_optimizacion_transformer.json").read_text(encoding="utf-8"))


CONFIGS = [
    core.ModelConfig(
        name="F1_label_balance15",
        label_smoothing=0.05,
        hard_weight=1.5,
        max_epochs=50,
        patience=10,
    ),
    core.ModelConfig(
        name="F2_label_balance20",
        label_smoothing=0.05,
        hard_weight=2.0,
        max_epochs=50,
        patience=10,
    ),
    core.ModelConfig(
        name="F3_label_balance30",
        label_smoothing=0.05,
        hard_weight=3.0,
        max_epochs=50,
        patience=10,
    ),
]


def grammar_reference(secret: str) -> str:
    meanings = {row["token_secreto"]: row["significado"] for row in core.DICTIONARY}
    tokens = secret.split()
    negation = tokens[-1] == "riga"
    sequence = tokens[:-1] if negation else tokens
    adjective = len(sequence) == 6
    output = [meanings[sequence[0]], meanings[sequence[1]]]
    if negation:
        output.append("no")
    if adjective:
        output.extend(
            [meanings[sequence[5]], meanings[sequence[2]], meanings[sequence[4]], meanings[sequence[3]]]
        )
    else:
        output.extend([meanings[sequence[4]], meanings[sequence[2]], meanings[sequence[3]]])
    return " ".join(output)


def main() -> None:
    stage = []
    models = {}
    print("Búsqueda enfocada", flush=True)
    for config in CONFIGS:
        model, training = core.train_model(config, core.TRAIN_INNER, core.DEV_INNER, seed=42)
        metrics = core.evaluate(model, core.DEV_INNER)
        row = {"config": asdict(config), "seed": 42, "training": training, "metrics": metrics}
        stage.append(row)
        models[config.name] = model
        print(config.name, metrics["frases_exactas"], metrics["exactitud_tokens"], training["best_epoch"], flush=True)

    top_names = [
        row["config"]["name"]
        for row in sorted(
            stage,
            key=lambda row: (
                row["metrics"]["frases_exactas"],
                row["metrics"]["exactitud_tokens"],
            ),
            reverse=True,
        )[:2]
    ]
    config_map = {config.name: config for config in CONFIGS}
    multiseed = [row for row in stage if row["config"]["name"] in top_names]
    for name in top_names:
        for seed in (17, 73):
            model, training = core.train_model(config_map[name], core.TRAIN_INNER, core.DEV_INNER, seed=seed)
            metrics = core.evaluate(model, core.DEV_INNER)
            multiseed.append(
                {"config": asdict(config_map[name]), "seed": seed, "training": training, "metrics": metrics}
            )
            print(name, seed, metrics["frases_exactas"], metrics["exactitud_tokens"], training["best_epoch"], flush=True)

    summaries = []
    for name in top_names:
        rows = [row for row in multiseed if row["config"]["name"] == name]
        summaries.append(
            {
                "config_name": name,
                "mean_sentence_exact": sum(row["metrics"]["frases_exactas"] for row in rows) / len(rows),
                "mean_token_accuracy": sum(row["metrics"]["exactitud_tokens"] for row in rows) / len(rows),
                "best_epochs": [row["training"]["best_epoch"] for row in rows],
                "parameters": rows[0]["training"]["parameters"],
            }
        )
    previous_summary = PREVIOUS["winner"]["selection_summary"]
    candidates = summaries + [previous_summary]
    winner_summary = max(
        candidates,
        key=lambda row: (
            row["mean_sentence_exact"],
            row["mean_token_accuracy"],
            -row["parameters"],
        ),
    )
    if winner_summary["config_name"] not in config_map:
        print("La búsqueda enfocada no supera al ganador anterior.", winner_summary, flush=True)
        result = {
            "stage": stage,
            "multiseed": multiseed,
            "summaries": summaries,
            "winner": "previous",
            "winner_summary": winner_summary,
        }
        (OUT / "resultados_refinamiento_transformer.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    config = config_map[winner_summary["config_name"]]
    epochs = sorted(winner_summary["best_epochs"])[1]
    print("Ganador enfocado", winner_summary, "épocas", epochs, flush=True)
    model, training = core.train_model(config, core.TRAIN_ALL, None, seed=42, fixed_epochs=epochs)
    official = core.evaluate(model, core.VALIDATION_OFFICIAL, include_predictions=True)
    subgroups = core.subgroup_metrics(official["predicciones"])
    ablation_model, ablation_training = core.train_model(
        config, core.TRAIN_ALL, None, seed=42, use_position=False, fixed_epochs=epochs
    )
    ablation = core.evaluate(ablation_model, core.VALIDATION_OFFICIAL)
    secrets = []
    for phrase in core.SECRET_PHRASES:
        translation = core.translate_greedy(model, phrase)
        reference = grammar_reference(phrase)
        secrets.append(
            {
                "secreto": phrase,
                "traduccion": translation,
                "referencia_reglas": reference,
                "coincide_referencia": translation == reference,
            }
        )

    artifact = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "use_position": True,
        "final_epochs": epochs,
        "beam_size": 1,
        "length_penalty": 0.0,
        "src_token_to_id": core.SRC_VOCAB.token_to_id,
        "tgt_token_to_id": core.TGT_VOCAB.token_to_id,
    }
    torch.save(artifact, ARTIFACTS / "modelo_transformer_optimizado.pt")
    result = {
        "stage": stage,
        "multiseed": multiseed,
        "summaries": summaries,
        "winner": "focused",
        "winner_summary": winner_summary,
        "config": asdict(config),
        "epochs": epochs,
        "training": training,
        "official": {key: value for key, value in official.items() if key != "predicciones"},
        "subgroups": subgroups,
        "ablation_training": ablation_training,
        "ablation": ablation,
        "predictions": official["predicciones"],
        "secret_translations": secrets,
    }
    (OUT / "resultados_refinamiento_transformer.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("OFICIAL", result["official"], flush=True)
    print("ABLACIÓN", ablation, flush=True)
    print("SECRETOS", sum(row["coincide_referencia"] for row in secrets), "/", len(secrets), flush=True)
    for row in secrets:
        print(row, flush=True)


if __name__ == "__main__":
    main()
