from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_NAME = "nateraw/vit-base-patch16-224-cifar10"
DATASET_NAME = "uoft-cs/cifar10"
SPLIT = "test"
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class ExampleRecord:
    dataset_index: int
    true_label: int
    true_label_name: str
    pred_label: int
    pred_label_name: str
    confidence: float
    margin: float
    correct: bool
    confidence_group: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_processor(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def infer_dataset(
    dataset,
    processor,
    model,
    device,
    batch_size: int = 64,
) -> pd.DataFrame:
    id2label = model.config.id2label
    rows: List[ExampleRecord] = []
    batch_starts = range(0, len(dataset), batch_size)

    for start in tqdm(
        batch_starts,
        desc="Inferindo CIFAR-10",
        total=(len(dataset) + batch_size - 1) // batch_size,
    ):
        batch = dataset[start : start + batch_size]
        images = batch["img"]
        true_labels = batch["label"]

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        pred_labels = probs.argmax(dim=-1).cpu().numpy()
        probs_np = probs.cpu().numpy()

        for i, (y_true, y_pred) in enumerate(zip(true_labels, pred_labels)):
            global_idx = start + i

            # confiança da classe prevista/correta
            confidence = float(probs_np[i, y_pred])

            # margem entre top1 e top2
            sorted_probs = np.sort(probs_np[i])[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])

            rows.append(
                ExampleRecord(
                    dataset_index=global_idx,
                    true_label=int(y_true),
                    true_label_name=id2label[int(y_true)],
                    pred_label=int(y_pred),
                    pred_label_name=id2label[int(y_pred)],
                    confidence=confidence,
                    margin=margin,
                    correct=bool(y_true == y_pred),
                )
            )

    return pd.DataFrame([asdict(r) for r in rows])


def assign_confidence_groups_within_class(
    df_correct: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    for cls in tqdm(
        sorted(df_correct["true_label"].unique()),
        desc="Agrupando por confiança",
    ):
        cls_df = df_correct[df_correct["true_label"] == cls].copy()

        # tercis por ranking dentro da classe
        cls_df = cls_df.sort_values("confidence", ascending=True).reset_index(drop=True)

        n = len(cls_df)
        if n < 3:
            raise ValueError(f"Classe {cls} tem poucos exemplos corretos: {n}")

        low_end = n // 3
        mid_end = 2 * n // 3

        groups = []
        for idx in range(n):
            if idx < low_end:
                groups.append("low")
            elif idx < mid_end:
                groups.append("medium")
            else:
                groups.append("high")

        cls_df["confidence_group"] = groups

        # informações extras úteis para análise posterior
        cls_df["confidence_rank_within_class"] = np.arange(len(cls_df))
        cls_df["confidence_percentile_within_class"] = (
            cls_df["confidence_rank_within_class"] / max(len(cls_df) - 1, 1)
        )

        parts.append(cls_df)

    return pd.concat(parts, ignore_index=True)


def stratified_sample(
    df_grouped: pd.DataFrame,
    n_per_group_per_class: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    sampled_parts = []
    group_pairs = [
        (cls, group)
        for cls in sorted(df_grouped["true_label"].unique())
        for group in ["low", "medium", "high"]
    ]

    for cls, group in tqdm(group_pairs, desc="Amostrando estratificado (aleatório)"):
        subset = df_grouped[
            (df_grouped["true_label"] == cls)
            & (df_grouped["confidence_group"] == group)
        ].copy()

        if len(subset) < n_per_group_per_class:
            raise ValueError(
                f"Classe {cls}, grupo {group}: "
                f"há só {len(subset)} exemplos, "
                f"mas você pediu {n_per_group_per_class}."
            )

        sampled = subset.sample(
            n=n_per_group_per_class,
            random_state=seed,
            replace=False,
        )
        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sort_values(
        ["true_label", "confidence_group", "dataset_index"]
    ).reset_index(drop=True)
    return sampled_df


def representative_sample_one_per_group(
    df_grouped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona 1 instância por classe x grupo de confiança.
    Critério:
    1) menor distância até a mediana da confiança do estrato
    2) maior margem
    3) menor dataset_index
    """
    selected_parts = []
    group_pairs = [
        (cls, group)
        for cls in sorted(df_grouped["true_label"].unique())
        for group in ["low", "medium", "high"]
    ]

    for cls, group in tqdm(group_pairs, desc="Selecionando 1 representativa por estrato"):
        subset = df_grouped[
            (df_grouped["true_label"] == cls)
            & (df_grouped["confidence_group"] == group)
        ].copy()

        if len(subset) == 0:
            raise ValueError(f"Nenhuma instância encontrada para classe {cls}, grupo {group}.")

        median_conf = subset["confidence"].median()
        subset["group_confidence_median"] = median_conf
        subset["distance_to_group_median"] = (subset["confidence"] - median_conf).abs()

        subset = subset.sort_values(
            by=["distance_to_group_median", "margin", "dataset_index"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

        chosen = subset.iloc[[0]].copy()
        selected_parts.append(chosen)

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.sort_values(
        ["true_label", "confidence_group", "dataset_index"]
    ).reset_index(drop=True)
    return selected_df


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    processor, model, device = load_model_and_processor(MODEL_NAME)

    # 1) inferência no split inteiro
    df = infer_dataset(dataset, processor, model, device, batch_size=64)

    # 2) manter apenas corretamente classificados
    df_correct = df[df["correct"]].copy().reset_index(drop=True)

    # 3) atribuir baixa/média/alta confiança dentro de cada classe
    df_grouped = assign_confidence_groups_within_class(df_correct)

    # 4) amostragem balanceada aleatória: 5 por classe x confiança
    selected = stratified_sample(
        df_grouped,
        n_per_group_per_class=5,  # 5 x 3 x 10 = 150
        seed=SEED,
    )

    # 5) amostragem representativa: 1 por classe x confiança
    selected_representative_1 = representative_sample_one_per_group(df_grouped)

    # salvar
    all_predictions_path = OUTPUT_DIR / "cifar10_all_predictions.csv"
    correct_predictions_path = OUTPUT_DIR / "cifar10_correct_predictions.csv"
    grouped_path = OUTPUT_DIR / "cifar10_correct_grouped_by_confidence.csv"
    selected_path = OUTPUT_DIR / "cifar10_selected_instances.csv"
    representative_path = OUTPUT_DIR / "cifar10_selected_instances_representative_1.csv"

    df.to_csv(all_predictions_path, index=False)
    df_correct.to_csv(correct_predictions_path, index=False)
    df_grouped.to_csv(grouped_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_representative_1.to_csv(
        representative_path,
        index=False,
    )

    print("Total test:", len(df))
    print("Correct only:", len(df_correct))

    print("\nSelecionadas aleatoriamente por classe x confiança (5 por estrato):")
    print(
        selected.groupby(["true_label_name", "confidence_group"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nSelecionadas representativas por classe x confiança (1 por estrato):")
    print(
        selected_representative_1.groupby(["true_label_name", "confidence_group"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nArquivos salvos:")
    print(f"- {all_predictions_path}")
    print(f"- {correct_predictions_path}")
    print(f"- {grouped_path}")
    print(f"- {selected_path}")
    print(f"- {representative_path}")


if __name__ == "__main__":
    main()
