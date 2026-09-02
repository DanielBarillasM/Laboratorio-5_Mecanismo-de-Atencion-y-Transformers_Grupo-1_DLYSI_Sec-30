from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PROJECT = (
    SCRIPT_DIRECTORY.parent
    if SCRIPT_DIRECTORY.name == "src"
    else Path(r"C:\Users\Daniel Barillas\Desktop\Lab-5_Deep-Learning\Laboratorio-5_Mecanismo-de-Atencion-y-Transformers_Grupo-1_DLYSI_Sec-30")
)
PROJECT = Path(os.environ.get("LAB5_PROJECT", str(DEFAULT_PROJECT)))
DATA = PROJECT / "datos" / "CARPETA_DATOS"
OUT = Path(os.environ.get("LAB5_OPT_OUT", str(PROJECT / "resultados")))
ARTIFACTS = Path(os.environ.get("LAB5_ARTIFACTS_OUT", str(PROJECT / "artefactos")))
OUT.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(6, max(1, torch.get_num_threads())))

SECRET_PHRASES = [
    "moro seza fefo povo mefi zita",
    "fefo mefi moro somi bane",
    "fefo rino fefo posi zizi gope riga",
    "moro sesi moro mapa gope riga",
    "fefo zizi fefo posi gero gope riga",
    "fefo zizi moro povo seza rizo",
    "moro sesi moro sena seza pefe riga",
    "moro somi fefo gero gope riga",
    "moro sesi moro sena mapa gope riga",
    "fefo gero fefo ragi rino nava",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


TRAIN_ALL = read_csv(DATA / "entrenamiento.csv")
VALIDATION_OFFICIAL = read_csv(DATA / "validacion.csv")
DICTIONARY = read_csv(DATA / "diccionario.csv")


class Vocabulary:
    SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    def __init__(self, tokens: list[str]):
        unique = self.SPECIALS + sorted(set(tokens) - set(self.SPECIALS))
        self.token_to_id = {token: index for index, token in enumerate(unique)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, boundaries: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in text.split()]
        if boundaries:
            ids = [self.token_to_id["<SOS>"]] + ids + [self.token_to_id["<EOS>"]]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for index in ids:
            token = self.id_to_token[int(index)]
            if token == "<EOS>":
                break
            if token not in {"<PAD>", "<SOS>"}:
                tokens.append(token)
        return " ".join(tokens)


SRC_VOCAB = Vocabulary([token for row in TRAIN_ALL for token in row["idioma_secreto"].split()])
TGT_VOCAB = Vocabulary([token for row in TRAIN_ALL for token in row["espanol"].split()])
PAD_SRC = SRC_VOCAB.token_to_id["<PAD>"]
PAD_TGT = TGT_VOCAB.token_to_id["<PAD>"]
SOS_TGT = TGT_VOCAB.token_to_id["<SOS>"]
EOS_TGT = TGT_VOCAB.token_to_id["<EOS>"]


def structure(row: dict[str, str]) -> dict[str, object]:
    tokens = row["idioma_secreto"].split()
    negation = tokens[-1] == "riga"
    core = tokens[:-1] if negation else tokens
    adjective = len(core) == 6
    if adjective:
        subject, obj, verb, adj = core[1], core[4], core[5], core[3]
    else:
        subject, obj, verb, adj = core[1], core[3], core[4], None
    return {
        "negacion": negation,
        "adjetivo": adjective,
        "mismo_determinante": core[0] == core[2],
        "largo": len(tokens),
        "sujeto": subject,
        "objeto": obj,
        "verbo": verb,
        "token_adjetivo": adj,
    }


def stratified_split(rows: list[dict[str, str]], fraction: float = 0.20, seed: int = 2026):
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        s = structure(row)
        key = (s["negacion"], s["adjetivo"], s["mismo_determinante"], s["largo"])
        groups[key].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    target = round(len(rows) * fraction)
    raw = {key: len(indices) * fraction for key, indices in groups.items()}
    allocations = {key: int(value) for key, value in raw.items()}
    remaining = target - sum(allocations.values())
    for key in sorted(groups, key=lambda k: raw[k] - allocations[k], reverse=True)[:remaining]:
        allocations[key] += 1
    dev_indices = {index for key, indices in groups.items() for index in indices[: allocations[key]]}
    train = [row for index, row in enumerate(rows) if index not in dev_indices]
    dev = [row for index, row in enumerate(rows) if index in dev_indices]
    assert len(train) + len(dev) == len(rows) and len(dev) == target
    return train, dev


TRAIN_INNER, DEV_INNER = stratified_split(TRAIN_ALL)


class TranslationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        src = torch.tensor(SRC_VOCAB.encode(row["idioma_secreto"]), dtype=torch.long)
        tgt = torch.tensor(TGT_VOCAB.encode(row["espanol"]), dtype=torch.long)
        return src, tgt


def collate(batch):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD_SRC),
        pad_sequence(tgts, batch_first=True, padding_value=PAD_TGT),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int = 48
    heads: int = 4
    layers: int = 2
    d_ff: int = 96
    dropout: float = 0.1
    optimizer: str = "adam"
    lr: float = 2e-3
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    hard_weight: float = 1.0
    activation: str = "relu"
    norm_first: bool = False
    max_epochs: int = 40
    patience: int = 9
    scheduler: bool = True


class TranslatorTransformer(nn.Module):
    def __init__(self, config: ModelConfig, use_position: bool = True):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.use_position = use_position
        self.src_embedding = nn.Embedding(len(SRC_VOCAB), config.d_model, padding_idx=PAD_SRC)
        self.tgt_embedding = nn.Embedding(len(TGT_VOCAB), config.d_model, padding_idx=PAD_TGT)
        self.position = PositionalEncoding(config.d_model, dropout=config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.heads,
            num_encoder_layers=config.layers,
            num_decoder_layers=config.layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.output = nn.Linear(config.d_model, len(TGT_VOCAB))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_padding = src.eq(PAD_SRC)
        tgt_padding = tgt.eq(PAD_TGT)
        causal = torch.triu(torch.ones((tgt.size(1), tgt.size(1)), dtype=torch.bool, device=tgt.device), diagonal=1)
        src_x = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        if self.use_position:
            src_x = self.position(src_x)
            tgt_x = self.position(tgt_x)
        hidden = self.transformer(
            src_x,
            tgt_x,
            tgt_mask=causal,
            src_key_padding_mask=src_padding,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_padding,
        )
        return self.output(hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def hard_example(row: dict[str, str]) -> bool:
    s = structure(row)
    return bool(s["mismo_determinante"] and (s["negacion"] or s["adjetivo"]))


def make_loader(rows: list[dict[str, str]], batch_size: int, training: bool, config: ModelConfig, seed: int):
    dataset = TranslationDataset(rows)
    generator = torch.Generator().manual_seed(seed)
    if training and config.hard_weight > 1.0:
        weights = [config.hard_weight if hard_example(row) else 1.0 for row in rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, generator=generator, collate_fn=collate)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        decoder_input, expected = tgt[:, :-1], tgt[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(src, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), expected.reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        valid_tokens = expected.ne(PAD_TGT).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens
    return total_loss / max(total_tokens, 1)


def train_model(
    config: ModelConfig,
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]] | None,
    seed: int,
    use_position: bool = True,
    fixed_epochs: int | None = None,
):
    set_seed(seed)
    model = TranslatorTransformer(config, use_position=use_position).to(DEVICE)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters >= 250_000:
        raise ValueError(f"{config.name} supera el límite: {parameters}")
    optimizer_class = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = None
    if config.scheduler and dev_rows is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
        )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_TGT,
        reduction="sum",
        label_smoothing=config.label_smoothing,
    )
    train_loader = make_loader(train_rows, 128, True, config, seed)
    dev_loader = make_loader(dev_rows, 256, False, config, seed) if dev_rows is not None else None
    epochs = fixed_epochs if fixed_epochs is not None else config.max_epochs
    history = {"train": [], "val": [], "lr": []}
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, dev_loader, criterion) if dev_loader is not None else train_loss
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(val_loss)
        if fixed_epochs is None and stale >= config.patience:
            break
    model.load_state_dict(best_state if dev_rows is not None else model.state_dict())
    return model, {
        "history": history,
        "best_epoch": best_epoch if dev_rows is not None else epochs,
        "best_val_loss": best_loss if dev_rows is not None else None,
        "epochs_run": len(history["train"]),
        "parameters": parameters,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def translate_greedy(model: nn.Module, source: str, max_length: int = 16) -> str:
    model.eval()
    src = torch.tensor([SRC_VOCAB.encode(source)], dtype=torch.long, device=DEVICE)
    generated = [SOS_TGT]
    for _ in range(max_length):
        tgt = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        next_id = int(model(src, tgt)[0, -1].argmax())
        generated.append(next_id)
        if next_id == EOS_TGT:
            break
    return TGT_VOCAB.decode(generated)


@torch.no_grad()
def translate_beam(model: nn.Module, source: str, beam_size: int = 3, length_penalty: float = 0.3, max_length: int = 16) -> str:
    if beam_size <= 1:
        return translate_greedy(model, source, max_length=max_length)
    model.eval()
    source_ids = SRC_VOCAB.encode(source)
    beams: list[tuple[list[int], float, bool]] = [([SOS_TGT], 0.0, False)]
    banned = {PAD_TGT, SOS_TGT}
    for _ in range(max_length):
        candidates: list[tuple[list[int], float, bool]] = []
        active = [(tokens, score, done) for tokens, score, done in beams if not done]
        candidates.extend((tokens, score, done) for tokens, score, done in beams if done)
        if not active:
            break
        src = torch.tensor([source_ids] * len(active), dtype=torch.long, device=DEVICE)
        tgt = torch.tensor([tokens for tokens, _, _ in active], dtype=torch.long, device=DEVICE)
        log_probs = model(src, tgt)[:, -1].log_softmax(dim=-1)
        for row_index, (tokens, score, _) in enumerate(active):
            values, indices = torch.topk(log_probs[row_index], k=min(beam_size + len(banned), log_probs.size(-1)))
            kept = 0
            for value, index in zip(values.tolist(), indices.tolist()):
                if index in banned:
                    continue
                new_tokens = tokens + [int(index)]
                candidates.append((new_tokens, score + float(value), index == EOS_TGT))
                kept += 1
                if kept == beam_size:
                    break

        def rank(item):
            tokens, score, _ = item
            length = max(1, len(tokens) - 1)
            return score / (length**length_penalty)

        beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
        if all(done for _, _, done in beams):
            break
    best = max(beams, key=lambda item: item[1] / (max(1, len(item[0]) - 1) ** length_penalty))
    return TGT_VOCAB.decode(best[0])


def evaluate(model, rows, beam_size: int = 1, length_penalty: float = 0.3, include_predictions: bool = False):
    correct_tokens = 0
    total_tokens = 0
    exact = 0
    records = []
    for row in rows:
        prediction = translate_beam(model, row["idioma_secreto"], beam_size, length_penalty)
        expected = row["espanol"]
        pred_tokens, expected_tokens = prediction.split(), expected.split()
        length = max(len(pred_tokens), len(expected_tokens))
        correct = sum(a == b for a, b in zip_longest(pred_tokens, expected_tokens, fillvalue=None))
        correct_tokens += correct
        total_tokens += length
        is_exact = prediction == expected
        exact += int(is_exact)
        if include_predictions:
            records.append(
                {
                    "secreto": row["idioma_secreto"],
                    "esperado": expected,
                    "prediccion": prediction,
                    "correcta": is_exact,
                    **structure(row),
                }
            )
    return {
        "exactitud_tokens": correct_tokens / total_tokens,
        "frases_exactas": exact / len(rows),
        "frases_correctas": exact,
        "frases_totales": len(rows),
        "tokens_correctos": correct_tokens,
        "tokens_totales": total_tokens,
        **({"predicciones": records} if include_predictions else {}),
    }


def subgroup_metrics(records: list[dict]) -> dict[str, dict]:
    definitions = {
        "mismo_determinante": lambda r: bool(r["mismo_determinante"]),
        "determinante_diferente": lambda r: not bool(r["mismo_determinante"]),
        "con_negacion": lambda r: bool(r["negacion"]),
        "sin_negacion": lambda r: not bool(r["negacion"]),
        "con_adjetivo": lambda r: bool(r["adjetivo"]),
        "sin_adjetivo": lambda r: not bool(r["adjetivo"]),
        "largo_5": lambda r: r["largo"] == 5,
        "largo_6": lambda r: r["largo"] == 6,
        "largo_7": lambda r: r["largo"] == 7,
    }
    output = {}
    for name, predicate in definitions.items():
        subset = [record for record in records if predicate(record)]
        output[name] = {
            "n": len(subset),
            "frases_correctas": sum(record["correcta"] for record in subset),
            "frases_exactas": sum(record["correcta"] for record in subset) / len(subset) if subset else None,
        }
    return output


CANDIDATES = [
    ModelConfig(name="E1_extendido"),
    ModelConfig(name="E2_label005", label_smoothing=0.05),
    ModelConfig(name="E3_adamw", optimizer="adamw", lr=1e-3, weight_decay=1e-4),
    ModelConfig(name="E4_balanceado", hard_weight=2.0),
    ModelConfig(name="E5_combinado", optimizer="adamw", lr=1e-3, weight_decay=1e-4, label_smoothing=0.05, hard_weight=2.0),
    ModelConfig(name="E6_d56", d_model=56, d_ff=112, optimizer="adamw", lr=1e-3, weight_decay=1e-4, label_smoothing=0.05, hard_weight=2.0),
    ModelConfig(name="E7_d64", d_model=64, d_ff=128, optimizer="adamw", lr=1e-3, weight_decay=1e-4, label_smoothing=0.05, hard_weight=2.0),
    ModelConfig(name="E8_prenorm_gelu", optimizer="adamw", lr=1e-3, weight_decay=1e-4, label_smoothing=0.05, hard_weight=2.0, activation="gelu", norm_first=True),
]


def compact_config(config: ModelConfig) -> dict:
    return asdict(config)


def load_optimized_model(path: Path | None = None):
    """Reconstruye el ganador para inferencia sin volver a ejecutar la búsqueda."""
    artifact_path = path or (ARTIFACTS / "modelo_transformer_optimizado.pt")
    artifact = torch.load(artifact_path, map_location=DEVICE, weights_only=False)
    config = ModelConfig(**artifact["config"])
    model = TranslatorTransformer(config, use_position=artifact["use_position"]).to(DEVICE)
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return model, artifact


def main() -> None:
    print("Dispositivo:", DEVICE)
    print("Datos:", len(TRAIN_INNER), len(DEV_INNER), len(VALIDATION_OFFICIAL))
    stage_results = []
    stage_models: dict[str, nn.Module] = {}
    for index, config in enumerate(CANDIDATES, 1):
        print(f"\n[{index}/{len(CANDIDATES)}] {config.name}", flush=True)
        model, training = train_model(config, TRAIN_INNER, DEV_INNER, seed=42)
        greedy = evaluate(model, DEV_INNER, beam_size=1)
        result = {"config": compact_config(config), "training": training, "greedy": greedy}
        stage_results.append(result)
        stage_models[config.name] = model
        print(
            config.name,
            f"exact={greedy['frases_exactas']:.4f}",
            f"token={greedy['exactitud_tokens']:.4f}",
            f"epoch={training['best_epoch']}",
            f"params={training['parameters']}",
            f"sec={training['seconds']:.1f}",
            flush=True,
        )
        (OUT / "checkpoint_optimizacion_progreso.json").write_text(
            json.dumps(stage_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    ranked_stage = sorted(
        stage_results,
        key=lambda item: (
            item["greedy"]["frases_exactas"],
            item["greedy"]["exactitud_tokens"],
            -item["training"]["parameters"],
        ),
        reverse=True,
    )
    top_for_beam = ranked_stage[:3]
    beam_results = []
    print("\nEvaluación de beam search", flush=True)
    for item in top_for_beam:
        name = item["config"]["name"]
        model = stage_models[name]
        for beam_size in (1, 3, 5):
            for penalty in ((0.0, 0.3) if beam_size > 1 else (0.0,)):
                metrics = evaluate(model, DEV_INNER, beam_size=beam_size, length_penalty=penalty)
                beam_results.append(
                    {
                        "config_name": name,
                        "beam_size": beam_size,
                        "length_penalty": penalty,
                        "metrics": metrics,
                        "parameters": item["training"]["parameters"],
                    }
                )
                print(name, beam_size, penalty, metrics["frases_exactas"], metrics["exactitud_tokens"], flush=True)

    ranked_beam = sorted(
        beam_results,
        key=lambda item: (
            item["metrics"]["frases_exactas"],
            item["metrics"]["exactitud_tokens"],
            -item["parameters"],
            -item["beam_size"],
        ),
        reverse=True,
    )
    finalist_names = []
    for item in ranked_beam:
        if item["config_name"] not in finalist_names:
            finalist_names.append(item["config_name"])
        if len(finalist_names) == 2:
            break
    best_beam_for_config = {}
    for name in finalist_names:
        best_beam_for_config[name] = next(item for item in ranked_beam if item["config_name"] == name)

    seeds = [17, 42, 73]
    multiseed = []
    print("\nValidación multisemilla", finalist_names, flush=True)
    config_map = {config.name: config for config in CANDIDATES}
    for name in finalist_names:
        beam_spec = best_beam_for_config[name]
        for seed in seeds:
            model, training = train_model(config_map[name], TRAIN_INNER, DEV_INNER, seed=seed)
            metrics = evaluate(
                model,
                DEV_INNER,
                beam_size=beam_spec["beam_size"],
                length_penalty=beam_spec["length_penalty"],
            )
            multiseed.append(
                {
                    "config_name": name,
                    "seed": seed,
                    "beam_size": beam_spec["beam_size"],
                    "length_penalty": beam_spec["length_penalty"],
                    "training": training,
                    "metrics": metrics,
                }
            )
            print(name, seed, metrics["frases_exactas"], metrics["exactitud_tokens"], training["best_epoch"], flush=True)

    summaries = []
    for name in finalist_names:
        rows = [row for row in multiseed if row["config_name"] == name]
        summaries.append(
            {
                "config_name": name,
                "mean_sentence_exact": sum(row["metrics"]["frases_exactas"] for row in rows) / len(rows),
                "mean_token_accuracy": sum(row["metrics"]["exactitud_tokens"] for row in rows) / len(rows),
                "best_epochs": [row["training"]["best_epoch"] for row in rows],
                "parameters": rows[0]["training"]["parameters"],
                "beam_size": rows[0]["beam_size"],
                "length_penalty": rows[0]["length_penalty"],
            }
        )
    winner_summary = max(
        summaries,
        key=lambda row: (row["mean_sentence_exact"], row["mean_token_accuracy"], -row["parameters"]),
    )
    winner_config = config_map[winner_summary["config_name"]]
    sorted_epochs = sorted(winner_summary["best_epochs"])
    final_epochs = max(22, sorted_epochs[len(sorted_epochs) // 2])
    print("\nGanador:", winner_summary, "épocas finales:", final_epochs, flush=True)

    final_model, final_training = train_model(
        winner_config, TRAIN_ALL, None, seed=42, use_position=True, fixed_epochs=final_epochs
    )
    final_metrics = evaluate(
        final_model,
        VALIDATION_OFFICIAL,
        beam_size=winner_summary["beam_size"],
        length_penalty=winner_summary["length_penalty"],
        include_predictions=True,
    )
    final_subgroups = subgroup_metrics(final_metrics["predicciones"])

    ablation_model, ablation_training = train_model(
        winner_config, TRAIN_ALL, None, seed=42, use_position=False, fixed_epochs=final_epochs
    )
    ablation_metrics = evaluate(
        ablation_model,
        VALIDATION_OFFICIAL,
        beam_size=winner_summary["beam_size"],
        length_penalty=winner_summary["length_penalty"],
    )

    secret_translations = [
        {
            "secreto": phrase,
            "traduccion": translate_beam(
                final_model,
                phrase,
                beam_size=winner_summary["beam_size"],
                length_penalty=winner_summary["length_penalty"],
            ),
        }
        for phrase in SECRET_PHRASES
    ]

    artifact = {
        "model_state_dict": final_model.state_dict(),
        "config": compact_config(winner_config),
        "use_position": True,
        "final_epochs": final_epochs,
        "beam_size": winner_summary["beam_size"],
        "length_penalty": winner_summary["length_penalty"],
        "src_token_to_id": SRC_VOCAB.token_to_id,
        "tgt_token_to_id": TGT_VOCAB.token_to_id,
    }
    torch.save(artifact, ARTIFACTS / "modelo_transformer_optimizado.pt")

    report = {
        "protocol": {
            "inner_train": len(TRAIN_INNER),
            "inner_validation": len(DEV_INNER),
            "official_validation": len(VALIDATION_OFFICIAL),
            "selection_metric_order": ["frases_exactas", "exactitud_tokens", "menos_parametros"],
            "secret_used_for_selection": False,
            "device": str(DEVICE),
        },
        "baseline_historical": {
            "exactitud_tokens": 0.9815668202764977,
            "frases_exactas": 0.9416666666666667,
            "frases_correctas": 226,
            "parametros": 98810,
        },
        "stage_search": stage_results,
        "beam_search": beam_results,
        "multiseed": multiseed,
        "multiseed_summary": summaries,
        "winner": {
            "config": compact_config(winner_config),
            "selection_summary": winner_summary,
            "final_epochs": final_epochs,
            "training": final_training,
            "official_metrics": {k: v for k, v in final_metrics.items() if k != "predicciones"},
            "subgroups": final_subgroups,
        },
        "ablation_without_position": {
            "training": ablation_training,
            "metrics": ablation_metrics,
        },
        "official_predictions": final_metrics["predicciones"],
        "secret_translations": secret_translations,
    }
    (OUT / "resultados_optimizacion_transformer.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print("\nMÉTRICAS OFICIALES", report["winner"]["official_metrics"])
    print("ABLACIÓN", ablation_metrics)
    print("SECRETOS")
    for row in secret_translations:
        print(row["secreto"], "->", row["traduccion"])
    print("Resultados:", OUT / "resultados_optimizacion_transformer.json")


if __name__ == "__main__":
    main()
