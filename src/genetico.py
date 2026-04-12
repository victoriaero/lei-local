#!/usr/bin/env python3
import os
import random
import datetime
import json
import logging

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


# Carrega VAE genérico 32×32 (modelo DCAE 32×32 treinado em ImageNet, disponibilizado como checkpoint diffusers)
VAE_MODEL_NAME = "stabilityai/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(VAE_MODEL_NAME).to(DEVICE)
vae.eval()
# Esse VAE opera com imagem 32×32 → latente 32×32 → reconstrução 32×32
sample_size = int(vae.config.sample_size)         # deve ser 32
# Como o VAE é totalmente convolucional, basta usar resolução 32×32 de entrada:
LATENT_CHANNELS = int(vae.config.latent_channels)  # normalmente 4
LATENT_HEIGHT = 32 // 4  # 8
LATENT_WIDTH = 32 // 4   # 8
LATENT_SHAPE = (LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)  # (4, 8, 8)

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

TARGET_CLASS = 0

print(f"Usando dispositivo: {DEVICE}")

# ===========================
# 1. Hiperparâmetros do GA
# ===========================
NUM_GERACOES      = 350
POPULACAO_INICIAL = 250

ELITISM        = 1
PROB_MUTACAO   = 0.95
PROB_CROSSOVER = 0.05

PIXEL_PERTURB_STD = 0.75

BATCH_EVAL_SIZE = 64

K_GRID = 25
n_cols = int(np.ceil(np.sqrt(K_GRID)))
n_rows = int(np.ceil(K_GRID / n_cols))

N_SNAPSHOTS = 25
OUTPUT_BASE = "outputs_vae32_cifar10"

SIGMA_INIT = 1

# Normalização esperada pelo ViT-CIFAR
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616]
)

# ===========================
# 2. Funções de Conversão Latente em Lote (VAE 32×32 ↔ CIFAR-10)
# ===========================
def latent_batch_to_pil(batch_z: torch.Tensor) -> list[Image.Image]:
    """
    Recebe batch_z: [B, 4, 8, 8], decodifica no VAE e retorna lista de PIL 224×224 RGB
    sem jitter de cor.
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
# 3. Operadores Genéticos
# ===========================
def init_population(size: int) -> list[torch.Tensor]:
    """
    Gera `size` tensores z ∼ N(0, SIGMA_INIT²), shape [C×8×8].
    """
    return [
        torch.randn(LATENT_SHAPE, device=DEVICE) * SIGMA_INIT
        for _ in range(size)
    ]


def mutate_latent(z: torch.Tensor, gen: int) -> torch.Tensor:
    """
    Adiciona ruído Gaussiano ao latent z. Escolhe std aleatório em [0, max_std],
    onde max_std = PIXEL_PERTURB_STD * fator, e fator vai de 1.0 → 0.1 ao longo das gerações.
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
# 4. Avaliação de Fitness (vetorizado por batch)
# ===========================
def evaluate_fitness(pop_z: list[torch.Tensor]) -> np.ndarray:
    """
    Avalia fitness de cada latente em `pop_z`. Retorna array numpy de formato [len(pop_z)].
    """
    fitness = np.zeros(len(pop_z), dtype=np.float32)

    for start in range(0, len(pop_z), BATCH_EVAL_SIZE):
        batch = pop_z[start: start + BATCH_EVAL_SIZE]
        batch_z = torch.stack(batch, dim=0)  # [B, C, 8, 8]

        # Decodifica em lote para PIL 224×224 RGB
        pil_imgs = latent_batch_to_pil(batch_z)

        # Converte PIL → tensor normalizado conforme CIFAR-10
        cifar_tensors = []
        for p in pil_imgs:
            t = transforms.ToTensor()(p)  # [3,224,224] ∈ [0,1]
            t = normalize(t)  # normaliza p/ ViT-CIFAR
            cifar_tensors.append(t)
        inputs = torch.stack(cifar_tensors).to(DEVICE)  # [B,3,224,224]

        # Classificador ViT-CIFAR → logits [B,10]
        with torch.no_grad():
            outputs = clf_model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = F.softmax(logits, dim=1)  # [B,10]

        # Extrai probabilidade de “bird” (índice 2)
        p_target = probs[:, TARGET_CLASS].cpu().numpy()  # [B]

        # Penalização de magnitude (vetorizado)
        flat = batch_z.view(batch_z.size(0), -1)  # [B, C*8*8]

        batch_fitness = p_target
        fitness[start: start + batch_z.size(0)] = batch_fitness

    return fitness  # numpy array de tamanho len(pop_z)


# ===========================
# 5. Loop Principal + Salvamento
# ===========================
def run_experiments():
    # 5.1: Configura diretórios de saída
    timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")
    run_dir = os.path.join(OUTPUT_BASE, timestamp)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Logging para arquivo
    logger = logging.getLogger("evolution")
    logger.setLevel(logging.INFO)
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
        "BATCH_EVAL_SIZE": BATCH_EVAL_SIZE,
        "TARGET_CLASS": TARGET_CLASS,
        "TARGET_CLASS_NAME": CIFAR10_CLASSES[TARGET_CLASS],
        "CIFAR10_CLASSES": CIFAR10_CLASSES,
        "K_GRID": K_GRID,
        "N_SNAPSHOTS": N_SNAPSHOTS,
        "LATENT_SHAPE": LATENT_SHAPE,
        "VAE_MODEL": VAE_MODEL_NAME,
        "CLASSIFIER": MODEL_NAME,
        "input_size": {"height": 224, "width": 224}
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Gera lista de gerações para snapshots
    gens_lin = list(np.linspace(1, NUM_GERACOES, N_SNAPSHOTS, dtype=int))
    snapshot_gens = sorted(set(gens_lin + [NUM_GERACOES]))

    # 5.2: Inicializar população (lista de tensores)
    pop_z = init_population(POPULACAO_INICIAL)

    # Histórico de frames para GIF
    grid_frames = []

    pbar = tqdm(range(1, NUM_GERACOES + 1), desc="Gerações")
    for gen in pbar:
        # 5.3: Avaliar fitness (vetorizado)
        fitness = evaluate_fitness(pop_z)
        mean_f = float(fitness.mean())
        best_f = float(fitness.max())
        std_f = float(fitness.std())

        # Log com precisão completa (mesma do CSV)
        logger.info(f"Geração {gen:03d} — mean={mean_f}, best={best_f}, std={std_f}")

        # 5.3.1: Salvar melhor indivíduo a cada 5 gerações
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
        pbar.set_postfix(mean=f"{mean_f:.3f}", best=f"{best_f:.3f}")

    # 5.7: Salvar população final (lista de latentes)
    torch.save(pop_z, os.path.join(run_dir, "final_population.pt"))

    # 5.7.1: Salvar melhor indivíduo da última geração como PNG
    final_fitness = evaluate_fitness(pop_z)
    best_idx_final = int(np.argmax(final_fitness))
    best_tensor_final = pop_z[best_idx_final]
    with torch.no_grad():
        recon_final = vae.decode(best_tensor_final.unsqueeze(0)).sample  # [1,3,32,32]
    recon_final = (recon_final.clamp(-1, 1) + 1.0) / 2.0  # [1,3,32,32] ∈ [0,1]
    recon_cpu_final = recon_final.cpu().squeeze(0)  # [3,32,32]
    # pil_best_final = transforms.ToPILImage()(recon_cpu_final).resize((224, 224))
    # pil_best_final.save(os.path.join(run_dir, "best_final.png"))

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
    print(f"Executando GA com AutoencoderKL 32×32 (genérico) → outputs em '{OUTPUT_BASE}'")
    run_experiments()