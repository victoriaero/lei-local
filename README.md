# Instrumentação de Métricas do `genetico.py`

Este projeto agora salva métricas completas em 3 níveis para evitar rerun de decoder/classificador em análises posteriores:

1. `individual_log.parquet` (ou `individual_log.csv` fallback)
2. `generation_summary.csv`
3. `run_summary.csv`

Também mantém compatibilidade com saída antiga via `metrics_per_gen.csv`.
No arquivo legado `metrics_per_gen.csv`, apenas linhas `eval_stage=in_loop` são exportadas (sem a avaliação final), para manter séries por geração consistentes.

## Organização dos artefatos

Cada execução é salva em:

`outputs_lei_local_sensitivity/instance=<instance_id>/class=<target_class>/confidence=<confidence_group>/seed=<seed>/run=<run_id>/`

Arquivos gerados por run:

- `config.json`
- `run.log`
- `individual_log.parquet` (preferencial) ou `individual_log.csv` (fallback)
- `generation_summary.csv`
- `run_summary.csv`
- `instance_summary.csv`
- `metrics_per_gen.csv` (legado)
- `artifacts_manifest.json`
- `best_gen_*.png`, `best_final.png`, `gif_evolution.gif`, `final_population.pt`, `best_delta_z.npy`

## Significado das métricas principais

### Individual log

Uma linha por indivíduo avaliado (incluindo avaliação final da população pós-loop), com:

- Identificação: `instance_id`, `run_id`, `generation`, `eval_id`, `individual_id`
- Estágio da avaliação: `eval_stage` (`in_loop` para gerações evolutivas, `final_eval` para avaliação final da população)
- Linhagem: `parent_ids`, `created_by`, `birth_generation`, `mutation_sigma`
- Classe/probabilidade/logits: `pred_class`, `changed_class`, `target_class_if_changed`, `prob_original_class`, `prob_best_alt_class`, `logit_original`, `logit_best_alt`, `margin_logit`
- Fitness decomposta: `fitness_total`, `fitness_margin_term`, `fitness_distance_penalty`, `fitness_constraint_penalty`
- Localidade/restrição: `dist_l2`, `dist_norm`, `within_confidence_region`, `constraint_violation`
- Similaridade perceptual: `lpips` (reservado, `NaN` por enquanto)
- Latente compactado: `z_hash`, `z_mean`, `z_std`, `z_l2_norm`, `z_head` (primeiros elementos)

Observação: `z_vector` completo é opcional e controlado por `SAVE_FULL_Z_VECTORS`.

### Generation summary

Resumo por geração com fitness, flips, distâncias e diversidade latente:

- Fitness: `best_fitness`, `mean_fitness`, `median_fitness`, `std_fitness`
- Margem: `best_margin`, `mean_margin`, `fraction_margin_positive`
- Flips: `num_flips`, `flip_rate`, `best_flip_distance`, `mean_flip_distance`
- Distância/localidade: `mean_dist_norm`, `median_dist_norm`, `std_dist_norm`, `fraction_outside_region`
- Diversidade: `mean_pairwise_latent_distance`, `distance_to_centroid_mean`
- Operadores genéticos: `mutation_sigma`, `num_mutation_offspring`, `num_crossover_offspring`, `elite_count`

### Run summary

Resumo final com indicadores de sucesso e dinâmica da busca:

- Volume: `total_evals`, `total_generations`
- Primeiro flip: `found_flip`, `first_flip_eval`, `first_flip_generation`, `evals_to_first_flip`
- Estatística de flips: `num_total_flips`, `num_unique_target_classes`
- Extremos globais: `best_fitness_ever`, `best_margin_ever`, `generation_of_best_fitness`, `generation_of_best_margin`
- Melhor flip: `best_flip_distance`, `best_flip_margin`, `best_flip_eval`, `best_flip_generation`, `best_flip_lpips`
- Robustez temporal: `diversity_start`, `diversity_mid`, `diversity_end`, `stagnation_length`, `fitness_improvement_last_10pct_budget`

## Análises possíveis sem rerodar

Com os arquivos acima, você pode reconstruir:

- taxa de flip por geração
- margem vs distância (`margin_logit` vs `dist_norm`)
- distribuição de distâncias e violações da região de confiança
- dinâmica de diversidade populacional
- trajetória de melhores indivíduos e estagnação
- análise de classes alvo após flip

Sem rerun ainda não é possível obter:

- LPIPS real (campo existe, mas está `NaN` até integrar cálculo)
- reconstrução de imagem por cada avaliação individual (somente snapshots salvos)

## Compatibilidade com logs antigos

No `genetico.py`, utilitários:

- `load_metrics_artifacts(run_dir)`
- `derive_from_legacy_metrics(legacy_metrics_csv)`

Esses utilitários extraem o máximo possível de `metrics_per_gen.csv` legado. Métricas não recuperáveis sem rerun ficam `NaN`/`None`.

## Exemplo simples de uso

```bash
python3 genetico.py
```

Após rodar, inspecione:

```bash
cat <run_dir>/run_summary.csv
head -n 5 <run_dir>/generation_summary.csv
python3 -c "import pandas as pd; print(pd.read_parquet('<run_dir>/individual_log.parquet').head())"
```

Se parquet não estiver disponível no ambiente, o código salva `individual_log.csv` automaticamente.
