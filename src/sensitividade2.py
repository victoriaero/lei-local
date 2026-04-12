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

# Esse VAE opera com imagem 32×32 → latente 32×32 → reconstrução 32×32
sample_size = int(vae.config.sample_size)         # deve ser 32
LATENT_CHANNELS = int(vae.config.latent_channels)  # normalmente 4
LATENT_HEIGHT = sample_size // 4  # 8
LATENT_WIDTH = sample_size // 4   # 8
LATENT_SHAPE = (LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)  # (4, 8, 8)
LATENT_DIM = LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH    # 256
LATENT_DIM_SQRT = float(np.sqrt(LATENT_DIM))

# Carrega classificador CIFAR-10 pré-treinado (Vision Transformer)
MODEL_NAME = "nateraw/vit-base-patch16-224-cifar10"
feature_extractor = AutoImageProcessor.from_pretrained(
    MODEL_NAME, size=224, use_fast=True
)
clf_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
clf_model.eval()

# Dicionário de classes CIFAR-10
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
# 1. Hiperparâmetros do GA (estrutura básica)
# ===========================
NUM_GERACOES      = 350
POPULACAO_INICIAL = 250

ELITISM        = 1
PROB_MUTACAO   = 0.95
PROB_CROSSOVER = 0.05

# Perturbação latente (reduzida para manter localidade)
PIXEL_PERTURB_STD = 0.30

BATCH_EVAL_SIZE = 64

K_GRID = 25
n_cols = int(np.ceil(np.sqrt(K_GRID)))
n_rows = int(np.ceil(K_GRID / n_cols))

N_SNAPSHOTS = 25
OUTPUT_BASE = "outputs_lei_local_sensitivity"

SIGMA_INIT = 1.0  # compatibilidade

# Normalização esperada pelo ViT-CIFAR
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]
)

# ===========================
# 1.1. Hiperparâmetros LEI-Local (Sensibilidade)
# ===========================

# Caminho da imagem de entrada que queremos explicar localmente
INPUT_IMAGE_PATH = "/scratch/samiramalaquias/lei_local/src/input_x0.png"  # <--- ajuste aqui

# Se TARGET_CLASS < 0, usamos a classe predita para x0 como classe original c0
TARGET_CLASS = -1  # -1 = usar classe predita automaticamente

# Objetivo agora: maximizar margem de logit
#   margin(z) = max_{k != c0} logit_k(z) - logit_c0(z)
MARGIN_ALPHA       = 1.0   # peso da margem
BETA_LAT_DIST      = 0.5   # penalização da distância normalizada em latent
TRUST_REGION_RADIUS = 1.0  # raio em termos de dist_norm (≈ "1 sigma")
TRUST_REGION_PENALTY = 2.0 # penalidade extra por sair da trust-region

# Desvio da população local em torno de z0 (reduzido)
SIGMA_LOCAL = 0.20

# ===========================
# 2. Funções de Conversão Latente em Lote (VAE 32×32 ↔ CIFAR-10)
# ===========================

to_tensor_01 = transforms.ToTensor()

def latent_batch_to_pil(batch_z: torch.Tensor) -> list[Image.Image]:
    """
    Recebe batch_z: [B, 4, 8, 8], decodifica no VAE e retorna lista de PIL 224×224 RGB.
    """
    with torch.no_grad():
        recon = vae.decode(batch_z).sample  # [B,3,32,32], valores em [-1,+1]
    # Normaliza de [-1,+1] para [0,1]
    recon = (recon.clamp(-1, 1) + 1.0) / 2.0  # [B,3,32,32] em [0,1]
    recon_cpu = recon.cpu()

    pil_list = []
    to_pil_local = transforms.ToPILImage()
    for i in range(recon_cpu.shape[0]):
        img_rgb = recon_cpu[i]  # [3,32,32]
        pil_32 = to_pil_local(img_rgb)  # PIL 32×32 RGB
        # Redimensiona para 224×224 para alimentar o ViT-CIFAR
        pil = pil_32.resize((224, 224))
        pil_list.append(pil)

    return pil_list

# ===========================
# 2.1. Codificar imagem de entrada → latente (z0) e obter logits0/p0
# ===========================

def encode_image_to_latent(pil_img: Image.Image) -> torch.Tensor:
    """
    Recebe uma PIL RGB qualquer, redimensiona para sample_size (32),
    normaliza para [-1, 1] e codifica no VAE.
    Retorna z0 com shape [C, 8, 8].
    """
    pil_32 = pil_img.resize((sample_size, sample_size))
    x = to_tensor_01(pil_32).unsqueeze(0).to(DEVICE)  # [1,3,32,32], [0,1]
    x = 2.0 * x - 1.0  # normaliza para [-1,1] (esperado pelo VAE)

    with torch.no_grad():
        posterior = vae.encode(x)
        # Para LEI-Local faz sentido pegar a média (mais estável)
        z = posterior.latent_dist.mean  # [1,4,8,8]

    return z.squeeze(0)  # [4,8,8]


def get_classifier_logits_and_class(pil_img: Image.Image):
    """
    Aplica o classificador à imagem e retorna:
      logits0: tensor [10] (CPU)
      p0: prob da classe predita
      pred_class: índice da classe predita
    """
    pil_224 = pil_img.resize((224, 224))
    x = to_tensor_01(pil_224)  # [3,224,224], [0,1]
    x = normalize(x).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

    with torch.no_grad():
        outputs = clf_model(x)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [1,10]
        probs = F.softmax(logits, dim=1)  # [1,10]

    logits0 = logits[0].detach().cpu()
    probs_np = probs.cpu().numpy()[0]
    pred_class = int(np.argmax(probs_np))
    p0 = float(probs_np[pred_class])
    return logits0, p0, pred_class

# ===========================
# 3. Operadores Genéticos
# ===========================

def init_population_local(z0: torch.Tensor, size: int) -> list[torch.Tensor]:
    """
    Inicializa população em torno de z0: z ~ N(z0, SIGMA_LOCAL^2 I).
    """
    pop = []
    for _ in range(size):
        noise = torch.randn_like(z0) * SIGMA_LOCAL
        pop.append((z0 + noise).to(DEVICE))
    return pop


def mutate_latent(z: torch.Tensor, gen: int) -> torch.Tensor:
    """
    Adiciona ruído Gaussiano ao latent z. std decresce ao longo das gerações.
    """
    fator = 0.05 + 0.95 * (1.0 - (gen - 1) / (NUM_GERACOES - 1))
    max_std = PIXEL_PERTURB_STD * fator
    std_rand = torch.rand(1, device=DEVICE).item() * max_std
    noise = torch.randn_like(z) * std_rand
    return z + noise


def crossover_latent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Interpolação linear entre z1 e z2: α*z1 + (1-α)*z2.
    """
    α = torch.rand(1, device=DEVICE)
    return α * z1 + (1.0 - α) * z2


def tournament_selection(pop: list[torch.Tensor], fitness: np.ndarray, k: int = 2) -> torch.Tensor:
    """
    Torneio de tamanho k: escolhe k índices aleatórios e devolve o tensor de maior fitness.
    """
    idxs = random.sample(range(len(pop)), k)
    return pop[idxs[0]] if fitness[idxs[0]] >= fitness[idxs[1]] else pop[idxs[1]]

# ===========================
# 4. Avaliação de Fitness de Sensibilidade (margin-based)
# ===========================

def evaluate_fitness_sensitivity(
    pop_z: list[torch.Tensor],
    z0: torch.Tensor,
    logits0: torch.Tensor,
    orig_class: int
):
    """
    LEI-Local (Sensibilidade) – versão por margem de logit:

      margin(z) = max_{k != orig_class} logit_k(z) - logit_orig(z)

    Fitness:
      f(z) = MARGIN_ALPHA * margin(z)
             - BETA_LAT_DIST * dist_norm(z)
             - TRUST_REGION_PENALTY * max(0, dist_norm(z) - TRUST_REGION_RADIUS)

    Onde:
      dist_norm(z) = ||z - z0||_2 / sqrt(LATENT_DIM)

    Retorna:
      fitness          : [N]
      margin_all       : [N]
      dist_norm_all    : [N]
      dist_raw_all     : [N]
      p_orig_all       : [N]
      p_other_max_all  : [N]
      pred_class_all   : [N] (int)
    """
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
    logits0 = logits0.to(DEVICE)
    logit_orig0 = logits0[orig_class].item()  # só para referência, se quiser

    for start in range(0, n, BATCH_EVAL_SIZE):
        batch = pop_z[start: start + BATCH_EVAL_SIZE]
        batch_z = torch.stack(batch, dim=0)  # [B, C, 8, 8]

        # Decodifica para imagens
        pil_imgs = latent_batch_to_pil(batch_z)

        # Prepara tensores para o classificador
        cifar_tensors = []
        for p in pil_imgs:
            t = to_tensor_01(p)           # [3,224,224], [0,1]
            t = normalize(t)              # normalizado CIFAR-10
            cifar_tensors.append(t)
        inputs = torch.stack(cifar_tensors).to(DEVICE)  # [B,3,224,224]

        with torch.no_grad():
            outputs = clf_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [B,10]
            probs = F.softmax(logits, dim=1)  # [B,10]

        # Logit da classe original
        logit_orig = logits[:, orig_class]  # [B]

        # Logits das demais classes
        logits_others = logits.clone()
        logits_others[:, orig_class] = -1e9  # tira a classe original da disputa
        logit_other_max, idx_other_max = torch.max(logits_others, dim=1)  # [B]

        # Margem de logit (se > 0, melhor outra classe do que a original)
        margin = logit_other_max - logit_orig  # [B]

        # Probabilidades para logging
        p_orig = probs[:, orig_class]
        p_other_max = probs.gather(1, idx_other_max.unsqueeze(1)).squeeze(1)  # [B]

        # Distância em latent até z0
        flat = batch_z.view(batch_z.size(0), -1)  # [B, 256]
        z0_batch = z0_flat.unsqueeze(0).expand_as(flat)  # [B, 256]
        diff = flat - z0_batch
        dist_raw = torch.norm(diff, dim=1)           # [B]
        dist_norm = dist_raw / LATENT_DIM_SQRT       # [B]

        # Trust region: penaliza distâncias acima do raio
        over_radius = torch.clamp(dist_norm - TRUST_REGION_RADIUS, min=0.0)  # [B]

        # Fitness (quanto maior, melhor)
        batch_fitness = (
            MARGIN_ALPHA * margin
            - BETA_LAT_DIST * dist_norm
            - TRUST_REGION_PENALTY * over_radius
        )

        # Predicted class
        pred_classes = torch.argmax(probs, dim=1)  # [B]

        bsz = batch_z.size(0)
        end = start + bsz

        fitness[start:end]          = batch_fitness.cpu().numpy()
        margin_all[start:end]       = margin.cpu().numpy()
        dist_norm_all[start:end]    = dist_norm.cpu().numpy()
        dist_raw_all[start:end]     = dist_raw.cpu().numpy()
        p_orig_all[start:end]       = p_orig.cpu().numpy()
        p_other_max_all[start:end]  = p_other_max.cpu().numpy()
        pred_class_all[start:end]   = pred_classes.cpu().numpy().astype(np.int32)

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
    # 5.0: Carregar imagem de entrada
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Imagem de entrada não encontrada em {INPUT_IMAGE_PATH}")
    x0_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    # 5.0.1: Codificar em z0 e obter logits0 / p0 / classe alvo
    z0 = encode_image_to_latent(x0_pil)  # [4,8,8]
    logits0, p0, pred_class = get_classifier_logits_and_class(x0_pil)

    if TARGET_CLASS < 0:
        target_class = pred_class
    else:
        target_class = TARGET_CLASS

    print(f"Imagem de entrada: {INPUT_IMAGE_PATH}")
    print(f"Classe predita: {CIFAR10_CLASSES[pred_class]} (índice {pred_class}), p0 = {p0:.4f}")
    print(f"Classe original usada em LEI-Local: {CIFAR10_CLASSES[target_class]} (índice {target_class})")

    # 5.1: Configura diretórios de saída
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    run_dir = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Logging para arquivo
    logger = logging.getLogger("evolution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"))
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Salvar config.json
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
        "CIFAR10_CLASSES": CIFAR10_CLASSES,
        "K_GRID": K_GRID,
        "N_SNAPSHOTS": N_SNAPSHOTS,
        "LATENT_SHAPE": LATENT_SHAPE,
        "VAE_MODEL": VAE_MODEL_NAME,
        "CLASSIFIER": MODEL_NAME,
        "input_size": {"height": 224, "width": 224},
        "TRUST_REGION_RADIUS": TRUST_REGION_RADIUS,
        "TRUST_REGION_PENALTY": TRUST_REGION_PENALTY,
        "MARGIN_ALPHA": MARGIN_ALPHA,
        "BETA_LAT_DIST": BETA_LAT_DIST,
        "INPUT_IMAGE_PATH": INPUT_IMAGE_PATH,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Arquivo de métricas por indivíduo/geração
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

    # Gera lista de gerações para snapshots
    gens_lin = list(np.linspace(1, NUM_GERACOES, N_SNAPSHOTS, dtype=int))
    snapshot_gens = sorted(set(gens_lin + [NUM_GERACOES]))

    # 5.2: Inicializar população LOCAL em torno de z0
    pop_z = init_population_local(z0, POPULACAO_INICIAL)

    # Histórico de frames para GIF
    grid_frames = []

    pbar = tqdm(range(1, NUM_GERACOES + 1), desc="Gerações")
    for gen in pbar:
        # 5.3: Avaliar fitness de sensibilidade (margin-based)
        (fitness,
         margin_arr,
         dist_norm_arr,
         dist_raw_arr,
         p_orig_arr,
         p_other_arr,
         pred_class_arr) = evaluate_fitness_sensitivity(
            pop_z, z0, logits0, target_class
        )

        mean_f = float(fitness.mean())
        best_f = float(fitness.max())
        std_f = float(fitness.std())

        frac_changed = float((pred_class_arr != target_class).mean())

        logger.info(
            f"Geração {gen:03d} — mean={mean_f:.4f}, best={best_f:.4f}, "
            f"std={std_f:.4f}, frac_changed={frac_changed:.3f}"
        )

        # 5.3.1: Salvar métricas por indivíduo nessa geração
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

        # 5.3.2: Salvar melhor indivíduo a cada 5 gerações
        if gen % 5 == 0:
            best_idx_5 = int(np.argmax(fitness))
            best_tensor_5 = pop_z[best_idx_5]
            with torch.no_grad():
                recon_5 = vae.decode(best_tensor_5.unsqueeze(0)).sample  # [1,3,32,32]
            recon_5 = (recon_5.clamp(-1, 1) + 1.0) / 2.0  # [1,3,32,32] ∈ [0,1]
            recon_cpu_5 = recon_5.cpu().squeeze(0)  # [3,32,32]
            pil_best_5 = transforms.ToPILImage()(recon_cpu_5).resize((224, 224))
            pil_best_5.save(os.path.join(run_dir, f"best_gen_{gen}.png"))

        # 5.4: Capturar snapshots em grid, se for geração marcada
        if gen in snapshot_gens:
            sorted_idxs = np.argsort(-fitness)  # índices ordenados por fitness decrescente
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

        # 5.5: Elitismo + torneio + crossover + mutação → nova população
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
        pbar.set_postfix(mean=f"{mean_f:.3f}", best=f"{best_f:.3f}", frac_changed=f"{frac_changed:.2f}")

    # 5.7: Salvar população final (lista de latentes)
    torch.save(pop_z, os.path.join(run_dir, "final_population.pt"))

    # 5.7.1: Avaliar população final para achar melhor indivíduo e salvar suas métricas
    (final_fitness,
     final_margin,
     final_dist_norm,
     final_dist_raw,
     final_p_orig,
     final_p_other,
     final_pred_class) = evaluate_fitness_sensitivity(
        pop_z, z0, logits0, target_class
    )
    best_idx_final = int(np.argmax(final_fitness))
    best_tensor_final = pop_z[best_idx_final]

    # Salvar melhor indivíduo final como imagem
    with torch.no_grad():
        recon_final = vae.decode(best_tensor_final.unsqueeze(0)).sample  # [1,3,32,32]
    recon_final = (recon_final.clamp(-1, 1) + 1.0) / 2.0  # [1,3,32,32] ∈ [0,1]
    recon_cpu_final = recon_final.cpu().squeeze(0)  # [3,32,32]
    pil_best_final = transforms.ToPILImage()(recon_cpu_final).resize((224, 224))
    pil_best_final.save(os.path.join(run_dir, "best_final.png"))

    # Salvar delta_z do melhor indivíduo final (z* - z0)
    delta_z_best = (best_tensor_final - z0.to(best_tensor_final.device)).detach().cpu().numpy()
    np.save(os.path.join(run_dir, "best_delta_z.npy"), delta_z_best)

    # 5.8: Gerar GIF da evolução (grid de snapshots)
    if grid_frames:
        grid_frames[0].save(
            os.path.join(run_dir, "gif_evolution.gif"),
            save_all=True,
            append_images=grid_frames[1:],
            duration=500,
            loop=0
        )

if __name__ == "__main__":
    print(f"Executando LEI-Local (Sensibilidade: margem de logit) com AutoencoderKL 32×32 → outputs em '{OUTPUT_BASE}'")
    run_experiments()
