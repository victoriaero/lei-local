#!/usr/bin/env python3
import os
import random
import datetime
import json
import logging
import csv

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from diffusers import AutoencoderKL

# ——— Seeds de Reprodutibilidade ———
seed = 420
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ===========================
# 0. Preparação: AutoencoderKL 32×32 Genérico + Classificador CIFAR-10
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega VAE genérico 32×32 (modelo DCAE 32×32 treinado em ImageNet)
VAE_MODEL_NAME = "stabilityai/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(VAE_MODEL_NAME).to(DEVICE)
vae.eval()
VAE_SCALING_FACTOR = float(getattr(vae.config, "scaling_factor", 1.0))

sample_size = int(vae.config.sample_size)         # deve ser 32

# Classificador CIFAR-10 pré-treinado
MODEL_NAME = "nateraw/vit-base-patch16-224-cifar10"
feature_extractor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, size=224, use_fast=True
)
clf_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
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
# 1. Hiperparâmetros do GA
# ===========================
NUM_GERACOES      = 350
POPULACAO_INICIAL = 120      # <<< reduzido para caber na GPU

ELITISM        = 1
PROB_MUTACAO   = 0.95
PROB_CROSSOVER = 0.05

PIXEL_PERTURB_STD = 0.30

BATCH_EVAL_SIZE = 8          # <<< bem menor para reduzir pico de memória

K_GRID = 25
n_cols = int(np.ceil(np.sqrt(K_GRID)))
n_rows = int(np.ceil(K_GRID / n_cols))

N_SNAPSHOTS = 25
OUTPUT_BASE = "outputs_lei_local_sensitivity"

SIGMA_INIT = 1.0

# ===========================
# 1.1. Hiperparâmetros LEI-Local (Sensibilidade)
# ===========================

INPUT_IMAGE_PATH = "/scratch/samiramalaquias/ppsn/sapo.jpg"

TARGET_CLASS = -1  # -1 = usar classe predita automaticamente

# Fase 1: antes de mudar de classe (empurrar margem para cima, mas com penalização de distância)
MARGIN_ALPHA_BEFORE   = 1.0
DIST_BETA_BEFORE      = 0.3

# Fase 2: já mudou de classe (minimizar dist_norm e não inflar demais a margem)
DIST_GAMMA_AFTER      = 2.0
MARGIN_GAMMA_AFTER    = 0.2

TRUST_REGION_RADIUS   = 1.0
TRUST_REGION_PENALTY  = 2.0

SIGMA_LOCAL = 0.20

# ===========================
# 2. Funções de Conversão Latente em Lote
# ===========================

to_tensor_01 = transforms.ToTensor()

def prepare_classifier_inputs(images) -> torch.Tensor:
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
    recon_cpu = recon.cpu()   # <<< libera GPU logo depois

    pil_list = []
    to_pil_local = transforms.ToPILImage()
    for i in range(recon_cpu.shape[0]):
        img_rgb = recon_cpu[i]      # [3,32,32] em CPU
        pil_32 = to_pil_local(img_rgb)
        pil = pil_32.resize((224, 224))
        pil_list.append(pil)

    # libera tensores para GC
    del recon, recon_cpu
    return pil_list

# ===========================
# 2.1. Encode x0 → z0
# ===========================

def encode_image_to_latent(pil_img: Image.Image) -> torch.Tensor:
    pil_32 = pil_img.resize((sample_size, sample_size))
    x = to_tensor_01(pil_32).unsqueeze(0).to(DEVICE)
    x = 2.0 * x - 1.0

    with torch.no_grad():
        posterior = vae.encode(x)
        z = posterior.latent_dist.mean * VAE_SCALING_FACTOR

    return z.squeeze(0)

def get_latent_stats(z: torch.Tensor):
    latent_shape = tuple(int(dim) for dim in z.shape)
    latent_dim = int(z.numel())
    latent_dim_sqrt = float(np.sqrt(latent_dim))
    return latent_shape, latent_dim, latent_dim_sqrt

def get_classifier_logits_and_class(pil_img: Image.Image):
    x = prepare_classifier_inputs(pil_img)

    with torch.no_grad():
        outputs = clf_model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = F.softmax(logits, dim=1)

    logits0 = logits[0].detach().cpu()
    probs_np = probs.cpu().numpy()[0]
    pred_class = int(np.argmax(probs_np))
    p0 = float(probs_np[pred_class])

    # libera tensores
    del logits, probs, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logits0, p0, pred_class

# ===========================
# 3. Operadores Genéticos
# ===========================

def init_population_local(z0: torch.Tensor, size: int) -> list[torch.Tensor]:
    pop = []
    for _ in range(size):
        noise = torch.randn_like(z0) * SIGMA_LOCAL
        pop.append((z0 + noise).to(DEVICE))
    return pop

def mutate_latent(z: torch.Tensor, gen: int) -> torch.Tensor:
    fator = 0.05 + 0.95 * (1.0 - (gen - 1) / (NUM_GERACOES - 1))
    max_std = PIXEL_PERTURB_STD * fator
    std_rand = torch.rand(1, device=DEVICE).item() * max_std
    noise = torch.randn_like(z) * std_rand
    return z + noise

def crossover_latent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    α = torch.rand(1, device=DEVICE)
    return α * z1 + (1.0 - α) * z2

def tournament_selection(pop: list[torch.Tensor], fitness: np.ndarray, k: int = 2) -> torch.Tensor:
    idxs = random.sample(range(len(pop)), k)
    return pop[idxs[0]] if fitness[idxs[0]] >= fitness[idxs[1]] else pop[idxs[1]]

# ===========================
# 4. Fitness de Sensibilidade (duas fases)
# ===========================

def evaluate_fitness_sensitivity(
    pop_z: list[torch.Tensor],
    z0: torch.Tensor,
    logits0: torch.Tensor,
    orig_class: int,
    latent_dim_sqrt: float
):
    n = len(pop_z)
    fitness          = np.zeros(n, dtype=np.float32)
    margin_all       = np.zeros(n, dtype=np.float32)
    dist_norm_all    = np.zeros(n, dtype=np.float32)
    dist_raw_all     = np.zeros(n, dtype=np.float32)
    p_orig_all       = np.zeros(n, dtype=np.float32)
    p_other_max_all  = np.zeros(n, dtype=np.float32)
    pred_class_all   = np.zeros(n, dtype=np.int32)

    z0 = z0.to(DEVICE)
    z0_flat = z0.view(-1)

    for start in range(0, n, BATCH_EVAL_SIZE):
        batch = pop_z[start: start + BATCH_EVAL_SIZE]
        batch_z = torch.stack(batch, dim=0)  # [B, C, 8, 8]

        # --- decode & classificador ---
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
        dist_raw = torch.norm(diff, dim=1)
        dist_norm = dist_raw / latent_dim_sqrt
        over_radius = torch.clamp(dist_norm - TRUST_REGION_RADIUS, min=0.0)

        pred_classes = torch.argmax(probs, dim=1)
        changed_mask = (pred_classes != orig_class).float()
        same_mask    = 1.0 - changed_mask

        fitness_before = (
            MARGIN_ALPHA_BEFORE * margin
            - DIST_BETA_BEFORE   * dist_norm
            - TRUST_REGION_PENALTY * over_radius
        )

        fitness_after = (
            - DIST_GAMMA_AFTER * dist_norm
            - MARGIN_GAMMA_AFTER * torch.relu(margin)
            - TRUST_REGION_PENALTY * over_radius
        )

        batch_fitness = same_mask * fitness_before + changed_mask * fitness_after

        bsz = batch_z.size(0)
        end = start + bsz

        fitness[start:end]          = batch_fitness.cpu().numpy()
        margin_all[start:end]       = margin.cpu().numpy()
        dist_norm_all[start:end]    = dist_norm.cpu().numpy()
        dist_raw_all[start:end]     = dist_raw.cpu().numpy()
        p_orig_all[start:end]       = p_orig.cpu().numpy()
        p_other_max_all[start:end]  = p_other_max.cpu().numpy()
        pred_class_all[start:end]   = pred_classes.cpu().numpy().astype(np.int32)

        # libera tudo desse batch
        del (batch_z, pil_imgs, inputs,
             outputs, logits, probs, logit_orig, logits_others,
             logit_other_max, idx_other_max, p_orig, p_other_max,
             flat, z0_batch, diff, dist_raw, dist_norm,
             over_radius, pred_classes, changed_mask, same_mask,
             fitness_before, fitness_after, batch_fitness)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (fitness,
            margin_all,
            dist_norm_all,
            dist_raw_all,
            p_orig_all,
            p_other_max_all,
            pred_class_all)

# ===========================
# 5. Loop Principal + Salvamento
# ===========================

def run_experiments():
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Imagem de entrada não encontrada em {INPUT_IMAGE_PATH}")
    x0_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    z0 = encode_image_to_latent(x0_pil)
    latent_shape, latent_dim, latent_dim_sqrt = get_latent_stats(z0)
    logits0, p0, pred_class = get_classifier_logits_and_class(x0_pil)

    if TARGET_CLASS < 0:
        target_class = pred_class
    else:
        target_class = TARGET_CLASS

    print(f"Imagem de entrada: {INPUT_IMAGE_PATH}")
    print(f"Classe predita: {CIFAR10_CLASSES[pred_class]} (índice {pred_class}), p0 = {p0:.4f}")
    print(f"Classe original usada em LEI-Local: {CIFAR10_CLASSES[target_class]} (índice {target_class})")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    run_dir = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("evolution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    config = {
        "seed": seed,
        "NUM_GERACOES": NUM_GERACOES,
        "POPULACAO_INICIAL": POPULACAO_INICIAL,
        "ELITISM": ELITISM,
        "PROB_MUTACAO": PROB_MUTACAO,
        "PROB_CROSSOVER": PROB_CROSSOVER,
        "PIXEL_PERTURB_STD": PIXEL_PERTURB_STD,
        "SIGMA_INIT": SIGMA_INIT,
        "SIGMA_LOCAL": SIGMA_LOCAL,
        "BATCH_EVAL_SIZE": BATCH_EVAL_SIZE,
        "ORIG_CLASS": target_class,
        "ORIG_CLASS_NAME": CIFAR10_CLASSES[target_class],
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
        "INPUT_IMAGE_PATH": INPUT_IMAGE_PATH,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(run_dir, "metrics_per_gen.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "index",
            "fitness",
            "margin_logit",
            "dist_norm",
            "dist_raw",
            "p_orig",
            "p_other_max",
            "pred_class",
            "changed_class"
        ])

    gens_lin = list(np.linspace(1, NUM_GERACOES, N_SNAPSHOTS, dtype=int))
    snapshot_gens = sorted(set(gens_lin + [NUM_GERACOES]))

    pop_z = init_population_local(z0, POPULACAO_INICIAL)

    grid_frames = []

    pbar = tqdm(range(1, NUM_GERACOES + 1), desc="Gerações")
    for gen in pbar:
        (fitness,
         margin_arr,
         dist_norm_arr,
         dist_raw_arr,
         p_orig_arr,
         p_other_arr,
         pred_class_arr) = evaluate_fitness_sensitivity(
            pop_z, z0, logits0, target_class, latent_dim_sqrt
        )

        mean_f = float(fitness.mean())
        best_f = float(fitness.max())
        std_f = float(fitness.std())
        frac_changed = float((pred_class_arr != target_class).mean())

        logger.info(
            f"Geração {gen:03d} — mean={mean_f:.4f}, best={best_f:.4f}, "
            f"std={std_f:.4f}, frac_changed={frac_changed:.3f}"
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            for idx in range(len(pop_z)):
                changed = int(pred_class_arr[idx] != target_class)
                writer.writerow([
                    gen,
                    idx,
                    float(fitness[idx]),
                    float(margin_arr[idx]),
                    float(dist_norm_arr[idx]),
                    float(dist_raw_arr[idx]),
                    float(p_orig_arr[idx]),
                    float(p_other_arr[idx]),
                    int(pred_class_arr[idx]),
                    changed,
                ])

        if gen % 5 == 0:
            best_idx_5 = int(np.argmax(fitness))
            best_tensor_5 = pop_z[best_idx_5]
            with torch.no_grad():
                recon_5 = vae.decode(best_tensor_5.unsqueeze(0) / VAE_SCALING_FACTOR).sample
            recon_5 = (recon_5.clamp(-1, 1) + 1.0) / 2.0
            recon_cpu_5 = recon_5.cpu().squeeze(0)
            pil_best_5 = transforms.ToPILImage()(recon_cpu_5).resize((224, 224))
            pil_best_5.save(os.path.join(run_dir, f"best_gen_{gen}.png"))
            del recon_5, recon_cpu_5, pil_best_5
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if gen in snapshot_gens:
            sorted_idxs = np.argsort(-fitness)
            indices = np.linspace(0, len(sorted_idxs) - 1, K_GRID, dtype=int)
            W, H = 224, 224
            grid_img = Image.new("RGB", (n_cols * W, n_rows * H))
            selected = [pop_z[idx] for idx in sorted_idxs[indices]]
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

        elite_idxs = np.argsort(fitness)[-ELITISM:][::-1]
        new_pop = [pop_z[i] for i in elite_idxs]

        while len(new_pop) < POPULACAO_INICIAL:
            p1 = tournament_selection(pop_z, fitness)
            p2 = tournament_selection(pop_z, fitness)
            child = crossover_latent(p1, p2) if random.random() < PROB_CROSSOVER else p1.clone()
            if random.random() < PROB_MUTACAO:
                child = mutate_latent(child, gen)
            new_pop.append(child)

        pop_z = new_pop
        pbar.set_postfix(mean=f"{mean_f:.3f}", best=f"{best_f:.3f}",
                         frac_changed=f"{frac_changed:.2f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(pop_z, os.path.join(run_dir, "final_population.pt"))

    (final_fitness,
     final_margin,
     final_dist_norm,
     final_dist_raw,
     final_p_orig,
     final_p_other,
     final_pred_class) = evaluate_fitness_sensitivity(
        pop_z, z0, logits0, target_class, latent_dim_sqrt
    )
    best_idx_final = int(np.argmax(final_fitness))
    best_tensor_final = pop_z[best_idx_final]

    with torch.no_grad():
        recon_final = vae.decode(best_tensor_final.unsqueeze(0) / VAE_SCALING_FACTOR).sample
    recon_final = (recon_final.clamp(-1, 1) + 1.0) / 2.0
    recon_cpu_final = recon_final.cpu().squeeze(0)
    pil_best_final = transforms.ToPILImage()(recon_cpu_final).resize((224, 224))
    pil_best_final.save(os.path.join(run_dir, "best_final.png"))

    delta_z_best = (best_tensor_final - z0.to(best_tensor_final.device)).detach().cpu().numpy()
    np.save(os.path.join(run_dir, "best_delta_z.npy"), delta_z_best)

    if grid_frames:
        grid_frames[0].save(
            os.path.join(run_dir, "gif_evolution.gif"),
            save_all=True,
            append_images=grid_frames[1:],
            duration=500,
            loop=0
        )

if __name__ == "__main__":
    print("Executando LEI-Local (Sensibilidade: perturbação mínima) →", OUTPUT_BASE)
    run_experiments()
