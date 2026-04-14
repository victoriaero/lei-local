#!/usr/bin/env python3
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import importlib.util
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- Optional parquet support (preferred for large individual logs) ---
try:
    import pandas as pd
except Exception:
    pd = None

# ---------------------------
# Hugging Face cache/network
# ---------------------------
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

PREFER_LOCAL_HF_CACHE = True


def load_vae_with_cache_fallback(model_name: str) -> AutoencoderKL:
    if PREFER_LOCAL_HF_CACHE:
        try:
            return AutoencoderKL.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            print(f"[HF] Cache local do VAE indisponível, tentando internet: {e}")
    return AutoencoderKL.from_pretrained(model_name)


def load_processor_with_cache_fallback(model_name: str, size: int = 224, use_fast: bool = True) -> AutoImageProcessor:
    if PREFER_LOCAL_HF_CACHE:
        try:
            return AutoImageProcessor.from_pretrained(
                model_name,
                size=size,
                use_fast=use_fast,
                local_files_only=True,
            )
        except Exception as e:
            print(f"[HF] Cache local do processor indisponível, tentando internet: {e}")
    return AutoImageProcessor.from_pretrained(model_name, size=size, use_fast=use_fast)


def load_classifier_with_cache_fallback(model_name: str) -> AutoModelForImageClassification:
    if PREFER_LOCAL_HF_CACHE:
        try:
            return AutoModelForImageClassification.from_pretrained(model_name, local_files_only=True)
        except Exception as e:
            print(f"[HF] Cache local do classificador indisponível, tentando internet: {e}")
    return AutoModelForImageClassification.from_pretrained(model_name)


# Evita colisão entre este arquivo (random.py) e o módulo padrão `random`.
_stdlib_random_path = os.path.join(sysconfig.get_paths()["stdlib"], "random.py")
_stdlib_random_spec = importlib.util.spec_from_file_location("random", _stdlib_random_path)
if _stdlib_random_spec and _stdlib_random_spec.loader:
    _stdlib_random_module = importlib.util.module_from_spec(_stdlib_random_spec)
    _stdlib_random_spec.loader.exec_module(_stdlib_random_module)
    sys.modules["random"] = _stdlib_random_module


def set_global_seed(seed_value: int) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# --- Seed base (cada run pode sobrescrever com seed própria) ---
BASE_SEED = 420
set_global_seed(BASE_SEED)

# ===========================
# 0. Preparação: AutoencoderKL 32x32 Genérico + Classificador CIFAR-10
# ===========================
CUDA_DEVICE_INDEX = int(os.environ.get("LEI_CUDA_DEVICE", "0"))
DEVICE = torch.device(
    f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
)

# Carrega VAE genérico 32x32 (modelo DCAE 32x32 treinado em ImageNet)
VAE_MODEL_NAME = "stabilityai/sd-vae-ft-ema"
vae = load_vae_with_cache_fallback(VAE_MODEL_NAME).to(DEVICE)
vae.eval()
VAE_SCALING_FACTOR = float(getattr(vae.config, "scaling_factor", 1.0))

sample_size = int(vae.config.sample_size)  # deve ser 32

# Classificador CIFAR-10 pré-treinado
MODEL_NAME = "nateraw/vit-base-patch16-224-cifar10"
feature_extractor = load_processor_with_cache_fallback(MODEL_NAME, size=224, use_fast=True)
clf_model = load_classifier_with_cache_fallback(MODEL_NAME).to(DEVICE)
clf_model.eval()

CIFAR10_CLASSES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

print(f"Usando dispositivo: {DEVICE}")

# ===========================
# 1. Hiperparâmetros (Baseline Aleatório)
# ===========================
NUM_GERACOES = 200
POPULACAO_INICIAL = 100  # candidatos avaliados por iteração (inclui incumbente)

# Escala da amostragem aleatória local ao redor de z0
SIGMA_LOCAL = 0.20
SIGMA_MIN = 1e-6
SIGMA_MAX = 3.0

PIXEL_PERTURB_STD = 0.30

BATCH_EVAL_SIZE = 8

K_GRID = 25
n_cols = int(np.ceil(np.sqrt(K_GRID)))
n_rows = int(np.ceil(K_GRID / n_cols))

N_SNAPSHOTS = 25
OUTPUT_BASE = "outputs_lei_local_sensitivity_random"

SIGMA_INIT = 1.0

# Orçamento de avaliações do classificador (comparável com GA/CMA-ES)
CLASSIFIER_EVAL_BUDGET_IN_LOOP = NUM_GERACOES * POPULACAO_INICIAL  # ~20k por padrão

# ===========================
# 1.1. Hiperparâmetros LEI-Local (Sensibilidade)
# ===========================

# Modo 1 (legado): executar com uma única imagem local
INPUT_IMAGE_PATH = "/scratch/victoria.estanislau/lei-local/src/aviao.jpg"

# Modo 2 (novo): executar em lote a partir de índices do dataset
USE_DATASET_INDEX_BATCH = True
INSTANCES_CSV_PATH = "/scratch/victoria.estanislau/lei-local/src_novo/selecao-instancias/outputs/cifar10_selected_instances_representative_1.csv"
DATASET_NAME = "uoft-cs/cifar10"
DATASET_SPLIT = "test"
RUNS_PER_INSTANCE = 5
RUN_SEED_INCREMENT = 1
INSTANCE_LIMIT: Optional[int] = None  # None = usa todas as instâncias do CSV

TARGET_CLASS = -1  # -1 = usar classe predita automaticamente

# Fase 1: antes de mudar de classe (empurrar margem para cima, mas com penalização de distância)
MARGIN_ALPHA_BEFORE = 1.0
DIST_BETA_BEFORE = 0.3

# Fase 2: já mudou de classe (minimizar dist_norm e não inflar demais a margem)
DIST_GAMMA_AFTER = 2.0
MARGIN_GAMMA_AFTER = 0.2

TRUST_REGION_RADIUS = 0.75
TRUST_REGION_PENALTY = 2.0

# ===========================
# 1.2. Logging / Artefatos
# ===========================

SAVE_RECONSTRUCTION_PATH_IN_LOG = True
SAVE_FULL_Z_VECTORS = False  # Se True, salva vetor completo em string JSON no log (pesado)
Z_VECTOR_HEAD_SIZE = 16       # Alternativa compacta: primeiros N elementos

# ===========================
# 2. Funções de Conversão Latente em Lote
# ===========================

to_tensor_01 = transforms.ToTensor()


def prepare_classifier_inputs(images: Any) -> torch.Tensor:
    inputs = feature_extractor(images=images, return_tensors="pt")
    return inputs["pixel_values"].to(DEVICE)


def latent_batch_to_pil(batch_z: torch.Tensor) -> list[Image.Image]:
    """
    batch_z: [B, 4, 8, 8] (no DEVICE).
    Faz decode no DEVICE, move para CPU e converte para PIL 224x224.
    """
    with torch.no_grad():
        recon = vae.decode(batch_z / VAE_SCALING_FACTOR).sample  # [B,3,32,32], [-1,+1]
    recon = (recon.clamp(-1, 1) + 1.0) / 2.0
    recon_cpu = recon.cpu()

    pil_list: list[Image.Image] = []
    to_pil_local = transforms.ToPILImage()
    for i in range(recon_cpu.shape[0]):
        img_rgb = recon_cpu[i]
        pil_32 = to_pil_local(img_rgb)
        pil = pil_32.resize((224, 224))
        pil_list.append(pil)

    del recon, recon_cpu
    return pil_list


# ===========================
# 2.1. Encode x0 -> z0
# ===========================

def encode_image_to_latent(pil_img: Image.Image) -> torch.Tensor:
    pil_32 = pil_img.resize((sample_size, sample_size))
    x = to_tensor_01(pil_32).unsqueeze(0).to(DEVICE)
    x = 2.0 * x - 1.0

    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.latent_dist.mean * VAE_SCALING_FACTOR

    return z.squeeze(0)


def get_latent_stats(z: torch.Tensor) -> tuple[tuple[int, ...], int, float]:
    latent_shape = tuple(int(dim) for dim in z.shape)
    latent_dim = int(z.numel())
    latent_dim_sqrt = float(np.sqrt(latent_dim))
    return latent_shape, latent_dim, latent_dim_sqrt


def get_classifier_logits_and_class(pil_img: Image.Image) -> tuple[torch.Tensor, float, int]:
    x = prepare_classifier_inputs(pil_img)

    with torch.no_grad():
        outputs = clf_model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)

    logits0 = logits[0].detach().cpu()
    probs_np = probs.cpu().numpy()[0]
    pred_class = int(np.argmax(probs_np))
    p0 = float(probs_np[pred_class])

    del logits, probs, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logits0, p0, pred_class


# ===========================
# 3. Operadores Aleatórios (Baseline)
# ===========================

@dataclass
class Individual:
    individual_id: int
    z: torch.Tensor
    created_by: str
    parent_ids: str
    mutation_sigma: Optional[float]
    birth_generation: int


def next_individual_id(counter: dict[str, int]) -> int:
    counter["value"] += 1
    return counter["value"]


@dataclass
class RandomState:
    center: np.ndarray
    sigma: float
    dim: int


def flatten_latent(z: torch.Tensor) -> np.ndarray:
    return z.detach().cpu().view(-1).numpy().astype(np.float32)


def vector_to_latent(v: np.ndarray, latent_shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(v.reshape(latent_shape)).to(DEVICE).float()


def init_random_state(z0: torch.Tensor) -> RandomState:
    x0 = flatten_latent(z0).astype(np.float64)
    dim = int(x0.size)
    return RandomState(
        center=x0,
        sigma=float(SIGMA_LOCAL),
        dim=dim,
    )


def sample_population_random(
    state: RandomState,
    latent_shape: tuple[int, ...],
    generation: int,
    id_counter: dict[str, int],
    batch_size: int,
) -> tuple[list[Individual], np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size precisa ser >= 1 para baseline aleatório.")

    noise = np.random.randn(batch_size, state.dim)
    x_samples = state.center[None, :] + state.sigma * noise

    population: list[Individual] = []
    for i in range(batch_size):
        z_tensor = vector_to_latent(x_samples[i].astype(np.float32), latent_shape)
        population.append(
            Individual(
                individual_id=next_individual_id(id_counter),
                z=z_tensor,
                created_by="random_sample",
                parent_ids="z0_center",
                mutation_sigma=float(state.sigma),
                birth_generation=int(generation),
            )
        )
    return population, x_samples


# ===========================
# 4. Fitness de Sensibilidade (duas fases)
# ===========================

def evaluate_fitness_sensitivity(
    population: list[Individual],
    z0: torch.Tensor,
    orig_class: int,
    latent_dim_sqrt: float,
) -> dict[str, np.ndarray]:
    n = len(population)

    metrics: dict[str, np.ndarray] = {
        "fitness_total": np.zeros(n, dtype=np.float32),
        "margin_logit": np.zeros(n, dtype=np.float32),
        "dist_norm": np.zeros(n, dtype=np.float32),
        "dist_l2": np.zeros(n, dtype=np.float32),
        "prob_original_class": np.zeros(n, dtype=np.float32),
        "prob_best_alt_class": np.zeros(n, dtype=np.float32),
        "pred_class": np.zeros(n, dtype=np.int32),
        "target_class_if_changed": np.full(n, -1, dtype=np.int32),
        "logit_original": np.zeros(n, dtype=np.float32),
        "logit_best_alt": np.zeros(n, dtype=np.float32),
        "fitness_margin_term": np.zeros(n, dtype=np.float32),
        "fitness_distance_penalty": np.zeros(n, dtype=np.float32),
        "fitness_constraint_penalty": np.zeros(n, dtype=np.float32),
        "constraint_violation": np.zeros(n, dtype=np.float32),
        "within_confidence_region": np.zeros(n, dtype=np.int32),
        "changed_class": np.zeros(n, dtype=np.int32),
    }

    z0 = z0.to(DEVICE)
    z0_flat = z0.view(-1)

    for start in range(0, n, BATCH_EVAL_SIZE):
        batch_individuals = population[start: start + BATCH_EVAL_SIZE]
        batch_z = torch.stack([ind.z for ind in batch_individuals], dim=0)

        pil_imgs = latent_batch_to_pil(batch_z)
        inputs = prepare_classifier_inputs(pil_imgs)

        with torch.no_grad():
            outputs = clf_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)

        logit_orig = logits[:, orig_class]
        logits_others = logits.clone()
        logits_others[:, orig_class] = -1e9
        logit_other_max, idx_other_max = torch.max(logits_others, dim=1)
        margin = logit_other_max - logit_orig

        p_orig = probs[:, orig_class]
        p_other_max = probs.gather(1, idx_other_max.unsqueeze(1)).squeeze(1)

        flat = batch_z.view(batch_z.size(0), -1)
        z0_batch = z0_flat.unsqueeze(0).expand_as(flat)
        diff = flat - z0_batch
        dist_l2 = torch.norm(diff, dim=1)
        dist_norm = dist_l2 / latent_dim_sqrt
        over_radius = torch.clamp(dist_norm - TRUST_REGION_RADIUS, min=0.0)

        pred_classes = torch.argmax(probs, dim=1)
        changed_mask = (pred_classes != orig_class).float()
        same_mask = 1.0 - changed_mask

        margin_term_before = MARGIN_ALPHA_BEFORE * margin
        margin_term_after = -MARGIN_GAMMA_AFTER * torch.relu(margin)
        margin_term = same_mask * margin_term_before + changed_mask * margin_term_after

        distance_penalty_before = DIST_BETA_BEFORE * dist_norm
        distance_penalty_after = DIST_GAMMA_AFTER * dist_norm
        distance_penalty = same_mask * distance_penalty_before + changed_mask * distance_penalty_after

        constraint_penalty = TRUST_REGION_PENALTY * over_radius

        batch_fitness = margin_term - distance_penalty - constraint_penalty

        bsz = batch_z.size(0)
        end = start + bsz

        pred_np = pred_classes.detach().cpu().numpy().astype(np.int32)
        changed_np = (pred_np != orig_class).astype(np.int32)
        alt_np = idx_other_max.detach().cpu().numpy().astype(np.int32)
        alt_np = np.where(changed_np == 1, alt_np, -1)

        metrics["fitness_total"][start:end] = batch_fitness.detach().cpu().numpy()
        metrics["margin_logit"][start:end] = margin.detach().cpu().numpy()
        metrics["dist_norm"][start:end] = dist_norm.detach().cpu().numpy()
        metrics["dist_l2"][start:end] = dist_l2.detach().cpu().numpy()
        metrics["prob_original_class"][start:end] = p_orig.detach().cpu().numpy()
        metrics["prob_best_alt_class"][start:end] = p_other_max.detach().cpu().numpy()
        metrics["pred_class"][start:end] = pred_np
        metrics["target_class_if_changed"][start:end] = alt_np
        metrics["logit_original"][start:end] = logit_orig.detach().cpu().numpy()
        metrics["logit_best_alt"][start:end] = logit_other_max.detach().cpu().numpy()
        metrics["fitness_margin_term"][start:end] = margin_term.detach().cpu().numpy()
        metrics["fitness_distance_penalty"][start:end] = distance_penalty.detach().cpu().numpy()
        metrics["fitness_constraint_penalty"][start:end] = constraint_penalty.detach().cpu().numpy()
        metrics["constraint_violation"][start:end] = over_radius.detach().cpu().numpy()
        metrics["within_confidence_region"][start:end] = (over_radius.detach().cpu().numpy() <= 0).astype(np.int32)
        metrics["changed_class"][start:end] = changed_np

        del (
            batch_z,
            pil_imgs,
            inputs,
            outputs,
            logits,
            probs,
            logit_orig,
            logits_others,
            logit_other_max,
            idx_other_max,
            margin,
            p_orig,
            p_other_max,
            flat,
            z0_batch,
            diff,
            dist_l2,
            dist_norm,
            over_radius,
            pred_classes,
            changed_mask,
            same_mask,
            margin_term_before,
            margin_term_after,
            margin_term,
            distance_penalty_before,
            distance_penalty_after,
            distance_penalty,
            constraint_penalty,
            batch_fitness,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return metrics


# ===========================
# 6. Instrumentação e métricas
# ===========================

def sanitize_for_path(raw: str) -> str:
    keep = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def tensor_fingerprint(z: torch.Tensor) -> str:
    z_bytes = z.detach().cpu().numpy().astype(np.float32).tobytes()
    return hashlib.sha1(z_bytes).hexdigest()[:16]


def compact_z_payload(z: torch.Tensor) -> tuple[float, float, float, str]:
    z_cpu = z.detach().cpu().view(-1)
    z_mean = float(z_cpu.mean().item())
    z_std = float(z_cpu.std(unbiased=False).item())
    z_norm = float(torch.norm(z_cpu, p=2).item())
    head = z_cpu[:Z_VECTOR_HEAD_SIZE].tolist()
    return z_mean, z_std, z_norm, json.dumps(head)


def maybe_full_z_payload(z: torch.Tensor) -> Optional[str]:
    if not SAVE_FULL_Z_VECTORS:
        return None
    return json.dumps(z.detach().cpu().view(-1).tolist())


def compute_population_diversity(population: list[Individual]) -> tuple[float, float]:
    if not population:
        return float("nan"), float("nan")
    z_stack = torch.stack([ind.z.detach().cpu().view(-1) for ind in population], dim=0)
    centroid = z_stack.mean(dim=0, keepdim=True)
    dist_to_centroid = torch.norm(z_stack - centroid, dim=1)
    centroid_mean = float(dist_to_centroid.mean().item())

    n = z_stack.shape[0]
    if n < 2:
        return float("nan"), centroid_mean

    with torch.no_grad():
        # O(n^2) mas viável para população atual (~100)
        dmat = torch.cdist(z_stack, z_stack, p=2)
        tri = torch.triu_indices(n, n, offset=1)
        pairwise = dmat[tri[0], tri[1]]
        pairwise_mean = float(pairwise.mean().item())

    return pairwise_mean, centroid_mean


def write_records_to_parquet_or_csv(records: list[dict[str, Any]], parquet_path: Path) -> tuple[Path, str]:
    """Prefere parquet; fallback para CSV quando engine não disponível."""
    if pd is not None:
        try:
            df = pd.DataFrame(records)
            df.to_parquet(parquet_path, index=False)
            return parquet_path, "parquet"
        except Exception:
            pass

    csv_fallback = parquet_path.with_suffix(".csv")
    if records:
        keys = list(records[0].keys())
        with csv_fallback.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in records:
                writer.writerow(row)
    else:
        with csv_fallback.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
    return csv_fallback, "csv_fallback"


def save_generation_summary(generation_summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not generation_summary_rows:
        return
    keys = list(generation_summary_rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in generation_summary_rows:
            writer.writerow(row)


def save_run_summary(run_summary: dict[str, Any], output_path: Path) -> None:
    keys = list(run_summary.keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(run_summary)


def save_instance_summary(instance_summary: dict[str, Any], output_path: Path) -> None:
    keys = list(instance_summary.keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(instance_summary)


def aggregate_generation_metrics(
    instance_id: str,
    run_id: str,
    generation: int,
    eval_metrics: dict[str, np.ndarray],
    population: list[Individual],
    mutation_sigma_reference: float,
    num_mutation_offspring: int,
    num_crossover_offspring: int,
    elite_count: int,
) -> dict[str, Any]:
    fitness = eval_metrics["fitness_total"]
    margins = eval_metrics["margin_logit"]
    changed = eval_metrics["changed_class"]
    dist_norm = eval_metrics["dist_norm"]

    flips_mask = changed == 1
    flip_dists = dist_norm[flips_mask]

    pairwise_mean, centroid_mean = compute_population_diversity(population)

    return {
        "instance_id": instance_id,
        "run_id": run_id,
        "generation": generation,
        "population_size": int(len(population)),
        "best_fitness": float(np.max(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "median_fitness": float(np.median(fitness)),
        "std_fitness": float(np.std(fitness)),
        "best_margin": float(np.max(margins)),
        "mean_margin": float(np.mean(margins)),
        "fraction_margin_positive": float(np.mean(margins > 0)),
        "num_flips": int(np.sum(flips_mask)),
        "flip_rate": float(np.mean(flips_mask)),
        "best_flip_distance": float(np.min(flip_dists)) if flip_dists.size > 0 else float("nan"),
        "mean_flip_distance": float(np.mean(flip_dists)) if flip_dists.size > 0 else float("nan"),
        "mean_dist_norm": float(np.mean(dist_norm)),
        "median_dist_norm": float(np.median(dist_norm)),
        "std_dist_norm": float(np.std(dist_norm)),
        "fraction_outside_region": float(np.mean(eval_metrics["within_confidence_region"] == 0)),
        "mean_pairwise_latent_distance": pairwise_mean,
        "distance_to_centroid_mean": centroid_mean,
        "mutation_sigma": float(mutation_sigma_reference),
        "num_mutation_offspring": int(num_mutation_offspring),
        "num_crossover_offspring": int(num_crossover_offspring),
        "elite_count": int(elite_count),
    }


def compute_stagnation_length(best_fitness_per_gen: list[float]) -> int:
    if not best_fitness_per_gen:
        return 0
    best_so_far = -float("inf")
    last_improvement_idx = 0
    for idx, val in enumerate(best_fitness_per_gen):
        if val > best_so_far:
            best_so_far = val
            last_improvement_idx = idx
    return len(best_fitness_per_gen) - 1 - last_improvement_idx


def finalize_run_summary(
    instance_id: str,
    run_id: str,
    individual_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_evals = len(individual_rows)
    total_generations = len(generation_rows)

    found_flip = any(int(r["changed_class"]) == 1 for r in individual_rows)

    first_flip_row = None
    if found_flip:
        for r in individual_rows:
            if int(r["changed_class"]) == 1:
                first_flip_row = r
                break

    num_total_flips = sum(int(r["changed_class"]) == 1 for r in individual_rows)
    unique_target_classes = {
        int(r["target_class_if_changed"])
        for r in individual_rows
        if int(r["changed_class"]) == 1 and int(r["target_class_if_changed"]) >= 0
    }

    best_fitness_row = max(individual_rows, key=lambda r: float(r["fitness_total"])) if individual_rows else None
    best_margin_row = max(individual_rows, key=lambda r: float(r["margin_logit"])) if individual_rows else None

    flip_rows = [r for r in individual_rows if int(r["changed_class"]) == 1]

    best_flip_row = None
    if flip_rows:
        best_flip_row = max(flip_rows, key=lambda r: float(r["fitness_total"]))

    max_margin_flip_row = None
    if flip_rows:
        max_margin_flip_row = max(flip_rows, key=lambda r: float(r["margin_logit"]))

    diversity_start = float("nan")
    diversity_mid = float("nan")
    diversity_end = float("nan")
    if generation_rows:
        diversity_start = float(generation_rows[0]["mean_pairwise_latent_distance"])
        diversity_mid = float(generation_rows[len(generation_rows) // 2]["mean_pairwise_latent_distance"])
        diversity_end = float(generation_rows[-1]["mean_pairwise_latent_distance"])

    best_fitness_per_gen = [float(r["best_fitness"]) for r in generation_rows]
    stagnation_length = compute_stagnation_length(best_fitness_per_gen)

    improvement_last_10pct = float("nan")
    if generation_rows:
        split_idx = max(1, int(math.floor(0.9 * len(generation_rows))))
        first_slice = best_fitness_per_gen[:split_idx]
        last_slice = best_fitness_per_gen[split_idx:]
        if first_slice and last_slice:
            improvement_last_10pct = float(max(last_slice) - max(first_slice))

    summary = {
        "instance_id": instance_id,
        "run_id": run_id,
        "total_evals": int(total_evals),
        "total_generations": int(total_generations),
        "found_flip": int(bool(found_flip)),
        "first_flip_eval": int(first_flip_row["eval_id"]) if first_flip_row else None,
        "first_flip_generation": int(first_flip_row["generation"]) if first_flip_row else None,
        "evals_to_first_flip": int(first_flip_row["eval_id"]) if first_flip_row else None,
        "num_total_flips": int(num_total_flips),
        "num_unique_target_classes": int(len(unique_target_classes)),
        "best_fitness_ever": float(best_fitness_row["fitness_total"]) if best_fitness_row else float("nan"),
        "best_margin_ever": float(best_margin_row["margin_logit"]) if best_margin_row else float("nan"),
        "generation_of_best_fitness": int(best_fitness_row["generation"]) if best_fitness_row else None,
        "generation_of_best_margin": int(best_margin_row["generation"]) if best_margin_row else None,
        "best_flip_distance": float(best_flip_row["dist_norm"]) if best_flip_row else float("nan"),
        "best_flip_margin": float(best_flip_row["margin_logit"]) if best_flip_row else float("nan"),
        "best_flip_eval": int(best_flip_row["eval_id"]) if best_flip_row else None,
        "best_flip_generation": int(best_flip_row["generation"]) if best_flip_row else None,
        "best_flip_lpips": float(best_flip_row["lpips"]) if best_flip_row else float("nan"),
        "first_flip_distance": float(first_flip_row["dist_norm"]) if first_flip_row else float("nan"),
        "first_flip_margin": float(first_flip_row["margin_logit"]) if first_flip_row else float("nan"),
        "max_margin_flip_distance": float(max_margin_flip_row["dist_norm"]) if max_margin_flip_row else float("nan"),
        "diversity_start": diversity_start,
        "diversity_mid": diversity_mid,
        "diversity_end": diversity_end,
        "stagnation_length": int(stagnation_length),
        "fitness_improvement_last_10pct_budget": improvement_last_10pct,
    }
    return summary


def append_individual_rows(
    rows: list[dict[str, Any]],
    instance_id: str,
    run_id: str,
    generation: int,
    population: list[Individual],
    eval_metrics: dict[str, np.ndarray],
    eval_counter: dict[str, int],
    eval_stage: str = "in_loop",
    recon_prefix: Optional[str] = None,
) -> None:
    for idx, ind in enumerate(population):
        eval_counter["value"] += 1
        eval_id = eval_counter["value"]

        z_mean, z_std, z_norm, z_head_json = compact_z_payload(ind.z)
        z_full = maybe_full_z_payload(ind.z)

        row = {
            "instance_id": instance_id,
            "run_id": run_id,
            "generation": int(generation),
            "eval_stage": eval_stage,
            "eval_id": int(eval_id),
            "individual_id": int(ind.individual_id),
            "parent_ids": ind.parent_ids,
            "created_by": ind.created_by,
            "pred_class": int(eval_metrics["pred_class"][idx]),
            "changed_class": int(eval_metrics["changed_class"][idx]),
            "target_class_if_changed": int(eval_metrics["target_class_if_changed"][idx]),
            "prob_original_class": float(eval_metrics["prob_original_class"][idx]),
            "prob_best_alt_class": float(eval_metrics["prob_best_alt_class"][idx]),
            "logit_original": float(eval_metrics["logit_original"][idx]),
            "logit_best_alt": float(eval_metrics["logit_best_alt"][idx]),
            "margin_logit": float(eval_metrics["margin_logit"][idx]),
            "fitness_total": float(eval_metrics["fitness_total"][idx]),
            "fitness_margin_term": float(eval_metrics["fitness_margin_term"][idx]),
            "fitness_distance_penalty": float(eval_metrics["fitness_distance_penalty"][idx]),
            "fitness_constraint_penalty": float(eval_metrics["fitness_constraint_penalty"][idx]),
            "dist_l2": float(eval_metrics["dist_l2"][idx]),
            "dist_norm": float(eval_metrics["dist_norm"][idx]),
            "within_confidence_region": int(eval_metrics["within_confidence_region"][idx]),
            "constraint_violation": float(eval_metrics["constraint_violation"][idx]),
            "lpips": float("nan"),  # reservado para cálculo opcional futuro
            "z_hash": tensor_fingerprint(ind.z),
            "z_mean": z_mean,
            "z_std": z_std,
            "z_l2_norm": z_norm,
            "z_head": z_head_json,
            "z_vector": z_full,
            "mutation_sigma": float(ind.mutation_sigma) if ind.mutation_sigma is not None else float("nan"),
            "birth_generation": int(ind.birth_generation),
            "reconstruction_ref": None,
        }
        rows.append(row)


def export_legacy_metrics_csv(individual_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Compatibilidade com saída antiga metrics_per_gen.csv."""
    keys = [
        "generation",
        "index",
        "fitness",
        "margin_logit",
        "dist_norm",
        "dist_raw",
        "p_orig",
        "p_other_max",
        "pred_class",
        "changed_class",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in individual_rows:
            if r.get("eval_stage") != "in_loop":
                continue
            grouped.setdefault(int(r["generation"]), []).append(r)

        for generation in sorted(grouped.keys()):
            rows = grouped[generation]
            for idx, row in enumerate(rows):
                writer.writerow({
                    "generation": generation,
                    "index": idx,
                    "fitness": row["fitness_total"],
                    "margin_logit": row["margin_logit"],
                    "dist_norm": row["dist_norm"],
                    "dist_raw": row["dist_l2"],
                    "p_orig": row["prob_original_class"],
                    "p_other_max": row["prob_best_alt_class"],
                    "pred_class": row["pred_class"],
                    "changed_class": row["changed_class"],
                })


def load_metrics_artifacts(run_dir: str) -> dict[str, Any]:
    """
    Carrega artefatos novos/antigos quando disponíveis.
    Preferência: individual_log.parquet -> individual_log.csv -> metrics_per_gen.csv.
    """
    base = Path(run_dir)
    out: dict[str, Any] = {"individual": None, "generation": None, "run": None, "legacy": None}

    individual_parquet = base / "individual_log.parquet"
    individual_csv = base / "individual_log.csv"
    generation_csv = base / "generation_summary.csv"
    run_csv = base / "run_summary.csv"
    legacy_csv = base / "metrics_per_gen.csv"

    if pd is not None:
        if individual_parquet.exists():
            try:
                out["individual"] = pd.read_parquet(individual_parquet)
            except Exception:
                out["individual"] = None
        elif individual_csv.exists():
            out["individual"] = pd.read_csv(individual_csv)

        if generation_csv.exists():
            out["generation"] = pd.read_csv(generation_csv)
        if run_csv.exists():
            out["run"] = pd.read_csv(run_csv)
        if legacy_csv.exists():
            out["legacy"] = pd.read_csv(legacy_csv)
    else:
        out["individual"] = str(individual_parquet if individual_parquet.exists() else individual_csv)
        out["generation"] = str(generation_csv) if generation_csv.exists() else None
        out["run"] = str(run_csv) if run_csv.exists() else None
        out["legacy"] = str(legacy_csv) if legacy_csv.exists() else None

    return out


def derive_from_legacy_metrics(legacy_metrics_csv: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Deriva o máximo possível a partir de logs antigos (sem classificador/decoder).
    Métricas indisponíveis permanecem NaN/None.
    """
    rows: list[dict[str, Any]] = []
    with open(legacy_metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        g = int(r["generation"])
        grouped.setdefault(g, []).append(r)

    generation_summary: list[dict[str, Any]] = []
    for g in sorted(grouped.keys()):
        rr = grouped[g]
        fitness = np.array([float(x["fitness"]) for x in rr], dtype=np.float32)
        margins = np.array([float(x["margin_logit"]) for x in rr], dtype=np.float32)
        dist_norm = np.array([float(x["dist_norm"]) for x in rr], dtype=np.float32)
        flips = np.array([int(x["changed_class"]) for x in rr], dtype=np.int32)

        generation_summary.append({
            "generation": g,
            "population_size": int(len(rr)),
            "best_fitness": float(np.max(fitness)),
            "mean_fitness": float(np.mean(fitness)),
            "median_fitness": float(np.median(fitness)),
            "std_fitness": float(np.std(fitness)),
            "best_margin": float(np.max(margins)),
            "mean_margin": float(np.mean(margins)),
            "flip_rate": float(np.mean(flips == 1)),
            "num_flips": int(np.sum(flips == 1)),
            "mean_dist_norm": float(np.mean(dist_norm)),
            "fraction_outside_region": float("nan"),
            "mean_pairwise_latent_distance": float("nan"),
            "distance_to_centroid_mean": float("nan"),
        })

    run_summary = {
        "total_evals": int(len(rows)),
        "total_generations": int(len(grouped)),
        "found_flip": int(any(int(r["changed_class"]) == 1 for r in rows)),
        "num_total_flips": int(sum(int(r["changed_class"]) == 1 for r in rows)),
        "best_fitness_ever": float(max(float(r["fitness"]) for r in rows)) if rows else float("nan"),
        "best_margin_ever": float(max(float(r["margin_logit"]) for r in rows)) if rows else float("nan"),
        "note": "Resumo derivado de logs legados; métricas de logits detalhados, parent_ids, diversidade latente e run-level avançado não podem ser reconstruídas sem rerun.",
    }

    return generation_summary, run_summary


# ===========================
# 7. Loop Principal + Salvamento
# ===========================

def class_name(class_idx: int) -> str:
    return CIFAR10_CLASSES.get(int(class_idx), f"class_{class_idx}")


def load_instance_specs_from_csv(csv_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV de instâncias não encontrado em {csv_path}")

    specs: list[dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "dataset_index" not in reader.fieldnames:
            raise ValueError(
                "CSV de instâncias precisa conter a coluna 'dataset_index'. "
                f"Colunas encontradas: {reader.fieldnames}"
            )
        for row in reader:
            idx_raw = row.get("dataset_index")
            if idx_raw is None or str(idx_raw).strip() == "":
                continue
            specs.append({
                "dataset_index": int(idx_raw),
                "row": row,
            })

    if not specs:
        raise ValueError(f"Nenhuma instância válida encontrada em {csv_path}")
    return specs


def dataset_item_to_pil_image(dataset_item: Any) -> Image.Image:
    img = dataset_item["img"]
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if hasattr(img, "convert"):
        return img.convert("RGB")

    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def run_single_experiment(
    x0_pil: Image.Image,
    input_reference: str,
    instance_id: str,
    run_seed: int,
    run_sequence_idx: int,
    dataset_index: Optional[int] = None,
) -> Path:
    set_global_seed(run_seed)

    z0 = encode_image_to_latent(x0_pil)
    latent_shape, latent_dim, latent_dim_sqrt = get_latent_stats(z0)
    _, p0, pred_class = get_classifier_logits_and_class(x0_pil)

    if TARGET_CLASS < 0:
        target_class = pred_class
    else:
        target_class = TARGET_CLASS

    print(f"Imagem de entrada: {input_reference}")
    if dataset_index is not None:
        print(f"Dataset index: {dataset_index}")
    print(f"Classe predita: {class_name(pred_class)} (indice {pred_class}), p0 = {p0:.4f}")
    print(f"Classe original usada em LEI-Local: {class_name(target_class)} (indice {target_class})")

    timestamp = datetime.datetime.now().isoformat(timespec="microseconds").replace(":", "-")
    confidence_group = f"trr_{TRUST_REGION_RADIUS:.2f}"
    run_id = f"{timestamp}_r{run_sequence_idx:03d}"

    run_dir = (
        Path(OUTPUT_BASE)
        / f"instance={instance_id}"
        / f"class={target_class}"
        / f"confidence={confidence_group}"
        / f"seed={run_seed}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"evolution.{instance_id}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log")
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    config = {
        "seed": run_seed,
        "run_sequence_idx": run_sequence_idx,
        "optimizer": "random_baseline",
        "NUM_GERACOES": NUM_GERACOES,
        "POPULACAO_INICIAL": POPULACAO_INICIAL,
        "PIXEL_PERTURB_STD": PIXEL_PERTURB_STD,
        "SIGMA_INIT": SIGMA_INIT,
        "SIGMA_LOCAL": SIGMA_LOCAL,
        "SIGMA_MIN": SIGMA_MIN,
        "SIGMA_MAX": SIGMA_MAX,
        "BATCH_EVAL_SIZE": BATCH_EVAL_SIZE,
        "CLASSIFIER_EVAL_BUDGET_IN_LOOP": CLASSIFIER_EVAL_BUDGET_IN_LOOP,
        "ORIG_CLASS": target_class,
        "ORIG_CLASS_NAME": class_name(target_class),
        "P0": p0,
        "K_GRID": K_GRID,
        "N_SNAPSHOTS": N_SNAPSHOTS,
        "LATENT_SHAPE": latent_shape,
        "LATENT_DIM": latent_dim,
        "VAE_MODEL": VAE_MODEL_NAME,
        "VAE_SCALING_FACTOR": VAE_SCALING_FACTOR,
        "CLASSIFIER": MODEL_NAME,
        "TRUST_REGION_RADIUS": TRUST_REGION_RADIUS,
        "TRUST_REGION_PENALTY": TRUST_REGION_PENALTY,
        "MARGIN_ALPHA_BEFORE": MARGIN_ALPHA_BEFORE,
        "DIST_BETA_BEFORE": DIST_BETA_BEFORE,
        "DIST_GAMMA_AFTER": DIST_GAMMA_AFTER,
        "MARGIN_GAMMA_AFTER": MARGIN_GAMMA_AFTER,
        "INPUT_IMAGE_REFERENCE": input_reference,
        "DATASET_INDEX": dataset_index,
        "DATASET_NAME": DATASET_NAME if dataset_index is not None else None,
        "DATASET_SPLIT": DATASET_SPLIT if dataset_index is not None else None,
        "instance_id": instance_id,
        "run_id": run_id,
        "confidence_group": confidence_group,
    }

    total_iterations = int(np.ceil(CLASSIFIER_EVAL_BUDGET_IN_LOOP / POPULACAO_INICIAL))
    total_iterations = max(total_iterations, 1)
    gens_lin = list(np.linspace(1, total_iterations, N_SNAPSHOTS, dtype=int))
    snapshot_gens = sorted(set(gens_lin + [total_iterations]))
    config["NUM_ITERACOES_RANDOM"] = total_iterations

    id_counter = {"value": 0}
    eval_counter = {"value": 0}

    random_state = init_random_state(z0)
    config["RANDOM_DIM"] = random_state.dim
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    grid_frames: list[Image.Image] = []
    individual_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    last_population: Optional[list[Individual]] = None
    last_eval_metrics: Optional[dict[str, np.ndarray]] = None
    last_generation = 0

    pbar = tqdm(range(1, total_iterations + 1), desc="Iteracoes")

    consumed_evals = 0
    for gen in pbar:
        remaining = CLASSIFIER_EVAL_BUDGET_IN_LOOP - consumed_evals
        if remaining <= 0:
            break
        batch_size = min(POPULACAO_INICIAL, remaining)
        sigma_before = float(random_state.sigma)

        population, x_samples = sample_population_random(
            random_state,
            latent_shape,
            generation=gen,
            id_counter=id_counter,
            batch_size=batch_size,
        )
        eval_metrics = evaluate_fitness_sensitivity(
            population, z0, target_class, latent_dim_sqrt
        )
        fitness = eval_metrics["fitness_total"]

        mean_f = float(np.mean(fitness))
        best_f = float(np.max(fitness))
        std_f = float(np.std(fitness))
        frac_changed = float(np.mean(eval_metrics["changed_class"] == 1))

        logger.info(
            f"Iteracao {gen:03d} - mean={mean_f:.4f}, best={best_f:.4f}, "
            f"std={std_f:.4f}, frac_changed={frac_changed:.3f}, "
            f"sigma={sigma_before:.5f}, batch={batch_size}, evals={consumed_evals + batch_size}/{CLASSIFIER_EVAL_BUDGET_IN_LOOP}"
        )

        append_individual_rows(
            rows=individual_rows,
            instance_id=instance_id,
            run_id=run_id,
            generation=gen,
            population=population,
            eval_metrics=eval_metrics,
            eval_counter=eval_counter,
            eval_stage="in_loop",
            recon_prefix="best_gen" if SAVE_RECONSTRUCTION_PATH_IN_LOG else None,
        )

        if gen % 5 == 0:
            best_idx_5 = int(np.argmax(fitness))
            best_tensor_5 = population[best_idx_5].z
            with torch.no_grad():
                recon_5 = vae.decode(best_tensor_5.unsqueeze(0) / VAE_SCALING_FACTOR).sample
            recon_5 = (recon_5.clamp(-1, 1) + 1.0) / 2.0
            recon_cpu_5 = recon_5.cpu().squeeze(0)
            pil_best_5 = transforms.ToPILImage()(recon_cpu_5).resize((224, 224))
            pil_best_5.save(run_dir / f"best_gen_{gen}.png")
            del recon_5, recon_cpu_5, pil_best_5
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if gen in snapshot_gens:
            sorted_idxs = np.argsort(-fitness)
            indices = np.linspace(0, len(sorted_idxs) - 1, K_GRID, dtype=int)
            W, H = 224, 224
            grid_img = Image.new("RGB", (n_cols * W, n_rows * H))
            selected = [population[idx].z for idx in sorted_idxs[indices]]
            batch_z = torch.stack(selected, dim=0).to(DEVICE)
            pil_imgs = latent_batch_to_pil(batch_z)
            for j, pil in enumerate(pil_imgs):
                pil_resized = pil.resize((W, H))
                row, col = divmod(j, n_cols)
                grid_img.paste(pil_resized, (col * W, row * H))
            grid_frames.append(grid_img)
            del batch_z, pil_imgs, pil_resized, grid_img
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best_idx = int(np.argmax(fitness))
        sigma_after = sigma_before
        gen_row = aggregate_generation_metrics(
            instance_id=instance_id,
            run_id=run_id,
            generation=gen,
            eval_metrics=eval_metrics,
            population=population,
            mutation_sigma_reference=sigma_before,
            num_mutation_offspring=batch_size,
            num_crossover_offspring=0,
            elite_count=0,
        )
        gen_row["best_candidate_idx"] = int(best_idx)
        gen_row["sigma_before"] = sigma_before
        gen_row["sigma_after"] = sigma_after
        gen_row["batch_size"] = int(batch_size)
        gen_row["consumed_evals"] = int(consumed_evals + batch_size)
        generation_rows.append(gen_row)

        consumed_evals += batch_size
        last_population = population
        last_eval_metrics = eval_metrics
        last_generation = gen

        pbar.set_postfix(
            mean=f"{mean_f:.3f}",
            best=f"{best_f:.3f}",
            frac_changed=f"{frac_changed:.2f}",
            sigma=f"{sigma_after:.4f}",
            evals=f"{consumed_evals}/{CLASSIFIER_EVAL_BUDGET_IN_LOOP}",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if last_population is None or last_eval_metrics is None:
        raise RuntimeError("Nenhuma avaliação foi executada no baseline aleatório.")

    population = last_population
    final_metrics = last_eval_metrics
    torch.save(population, run_dir / "final_population.pt")

    # Não faz avaliação extra do classificador; reaproveita última população para respeitar orçamento.
    append_individual_rows(
        rows=individual_rows,
        instance_id=instance_id,
        run_id=run_id,
        generation=last_generation,
        population=population,
        eval_metrics=final_metrics,
        eval_counter=eval_counter,
        eval_stage="final_eval_reuse",
        recon_prefix="best_final_eval" if SAVE_RECONSTRUCTION_PATH_IN_LOG else None,
    )

    best_idx_final = int(np.argmax(final_metrics["fitness_total"]))
    best_tensor_final = population[best_idx_final].z

    with torch.no_grad():
        recon_final = vae.decode(best_tensor_final.unsqueeze(0) / VAE_SCALING_FACTOR).sample
    recon_final = (recon_final.clamp(-1, 1) + 1.0) / 2.0
    recon_cpu_final = recon_final.cpu().squeeze(0)
    pil_best_final = transforms.ToPILImage()(recon_cpu_final).resize((224, 224))
    pil_best_final.save(run_dir / "best_final.png")

    delta_z_best = (best_tensor_final - z0.to(best_tensor_final.device)).detach().cpu().numpy()
    np.save(run_dir / "best_delta_z.npy", delta_z_best)

    if grid_frames:
        grid_frames[0].save(
            run_dir / "gif_evolution.gif",
            save_all=True,
            append_images=grid_frames[1:],
            duration=500,
            loop=0,
        )

    individual_log_path, individual_format = write_records_to_parquet_or_csv(
        individual_rows,
        run_dir / "individual_log.parquet",
    )
    save_generation_summary(generation_rows, run_dir / "generation_summary.csv")

    run_summary = finalize_run_summary(
        instance_id=instance_id,
        run_id=run_id,
        individual_rows=individual_rows,
        generation_rows=generation_rows,
    )
    save_run_summary(run_summary, run_dir / "run_summary.csv")

    instance_summary = {
        "instance_id": instance_id,
        "run_id": run_id,
        "seed": run_seed,
        "run_sequence_idx": run_sequence_idx,
        "orig_class": target_class,
        "orig_class_name": class_name(target_class),
        "input_image_reference": input_reference,
        "dataset_index": dataset_index,
        "individual_log_path": str(individual_log_path),
        "individual_log_format": individual_format,
        "generation_summary_path": str(run_dir / "generation_summary.csv"),
        "run_summary_path": str(run_dir / "run_summary.csv"),
    }
    save_instance_summary(instance_summary, run_dir / "instance_summary.csv")

    # Mantém saída antiga para retrocompatibilidade
    export_legacy_metrics_csv(individual_rows, run_dir / "metrics_per_gen.csv")

    artifacts_doc = {
        "where": str(run_dir),
        "files": {
            "individual_log": str(individual_log_path),
            "generation_summary": str(run_dir / "generation_summary.csv"),
            "run_summary": str(run_dir / "run_summary.csv"),
            "instance_summary": str(run_dir / "instance_summary.csv"),
            "legacy_metrics": str(run_dir / "metrics_per_gen.csv"),
        },
        "notes": [
            "individual_log tem uma linha por individuo avaliado (inclui geração final de avaliação).",
            "generation_summary agrega métricas por geração.",
            "run_summary agrega métricas finais por run.",
            "lpips é reservado (NaN) até ser integrado ao pipeline.",
            "z_vector completo é opcional (SAVE_FULL_Z_VECTORS).",
        ],
    }
    with (run_dir / "artifacts_manifest.json").open("w") as f:
        json.dump(artifacts_doc, f, indent=2)

    return run_dir


def run_experiments() -> None:
    if USE_DATASET_INDEX_BATCH:
        instance_specs = load_instance_specs_from_csv(INSTANCES_CSV_PATH)
        if INSTANCE_LIMIT is not None:
            instance_specs = instance_specs[:INSTANCE_LIMIT]

        print(
            "Carregando dataset para execução em lote:",
            f"{DATASET_NAME} ({DATASET_SPLIT}), instâncias={len(instance_specs)}, "
            f"runs_por_instancia={RUNS_PER_INSTANCE}"
        )
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

        total_jobs = len(instance_specs) * RUNS_PER_INSTANCE
        job_idx = 0
        for instance_pos, spec in enumerate(instance_specs, start=1):
            dataset_index = int(spec["dataset_index"])
            if dataset_index < 0 or dataset_index >= len(dataset):
                raise IndexError(
                    f"dataset_index={dataset_index} fora do range do split "
                    f"{DATASET_SPLIT} (tamanho={len(dataset)})"
                )

            x0_pil = dataset_item_to_pil_image(dataset[dataset_index])
            instance_id = sanitize_for_path(
                f"{DATASET_NAME}_{DATASET_SPLIT}_idx_{dataset_index:05d}"
            )
            input_reference = f"{DATASET_NAME}:{DATASET_SPLIT}:{dataset_index}"

            for run_sequence_idx in range(1, RUNS_PER_INSTANCE + 1):
                run_seed = BASE_SEED + (job_idx * RUN_SEED_INCREMENT)
                job_idx += 1
                print(
                    f"\n[{job_idx}/{total_jobs}] Executando instância "
                    f"{instance_pos}/{len(instance_specs)} "
                    f"(dataset_index={dataset_index}) run={run_sequence_idx} seed={run_seed}"
                )
                run_single_experiment(
                    x0_pil=x0_pil,
                    input_reference=input_reference,
                    instance_id=instance_id,
                    run_seed=run_seed,
                    run_sequence_idx=run_sequence_idx,
                    dataset_index=dataset_index,
                )
        return

    # Modo legado: execução para uma única imagem local
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Imagem de entrada não encontrada em {INPUT_IMAGE_PATH}")

    x0_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    instance_id = sanitize_for_path(Path(INPUT_IMAGE_PATH).stem)
    run_single_experiment(
        x0_pil=x0_pil,
        input_reference=INPUT_IMAGE_PATH,
        instance_id=instance_id,
        run_seed=BASE_SEED,
        run_sequence_idx=1,
        dataset_index=None,
    )


# ===========================
# 8. Exemplo rápido / documentação programática
# ===========================

def print_usage_example() -> None:
    example = {
        "run": "python3 random.py",
        "inspect": [
            "cat <run_dir>/run_summary.csv",
            "head -n 5 <run_dir>/generation_summary.csv",
            "python3 -c \"import pandas as pd; print(pd.read_parquet('<run_dir>/individual_log.parquet').head())\"",
        ],
        "retro_compat": "derive_from_legacy_metrics('<run_dir>/metrics_per_gen.csv')",
    }
    print(json.dumps(example, indent=2))


if __name__ == "__main__":
    print("Executando LEI-Local com baseline aleatorio ->", OUTPUT_BASE)
    run_experiments()
