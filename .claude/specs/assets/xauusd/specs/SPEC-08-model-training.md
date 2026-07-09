# SPEC-08 — Entrenamiento del Modelo (LSTM → PPO)

## Propósito
Entrenar el agente con arquitectura **LSTM encoder → PPO policy**, protocolo **multi-seed** (≥5), tracking en **MLflow** (que además provee el conteo de trials para el Deflated Sharpe), y `CostModel` realista compartido con el entorno.

> **Versionado (SPEC-12):** cada corrida de entrenamiento parte de una **config congelada** (`ssot-versioning.md`) que ES la `model_version`. Cambiar hiperparámetros/features = versión nueva → al validar (SPEC-09) produce un **bundle inmutable** distinto en el registro, replayable y comparable contra las versiones previas. Nunca sobreescribir artefactos de una versión ya publicada.

## Arquitectura
Patrón CLSTM-PPO desacoplado: LSTM extrae features temporales de la ventana H1 → embedding → concatena con contexto Daily/macro/cuenta → cabeza MLP de PPO (política + valor).

**Implementación:**
- **Opción A:** `RecurrentPPO` de `sb3-contrib` (LSTM en la policy, maneja el estado recurrente).
- **Opción B:** PPO estándar con `features_extractor_class` custom que envuelve un `nn.LSTM` sobre la parte secuencial y concatena el contexto no-secuencial.

Backend PyTorch. Documentar la elección A/B en ADR si difiere del default (A).

## CostModel (compartido con SPEC-05)
```python
class CostModel:
    def spread(self, ts) -> float:      # variable por hora: ancho en Asia y rollover
    def slippage(self, ts, vol) -> float:  # peor caso en eventos / vol-spike
    def swap(self, position, held_overnight, ts) -> float:  # ¡NO omitir!
```
**Swap overnight:** en XAU/USD, especialmente longs con tasas altas, puede costar **2–4% anualizado** de la exposición sostenida. Obtener del broker o modelar como `SOFR + markup − lease_rate` del oro. Con el flat de fin de semana se ahorra el triple swap (miércoles o viernes según convención — verificar broker). Incluido en la recompensa (SPEC-05) y en el backtest (SPEC-09).

## Hiperparámetros de arranque (`config/train.yaml`)
```yaml
algo: recurrent_ppo
policy: MlpLstmPolicy
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
ent_coef: 0.01
clip_range: 0.2
n_envs: 8
n_steps: 2048
batch_size: 64
total_timesteps: 1_000_000     # escalar si el train no satura
lstm_hidden_size: 128
seeds: [0, 1, 2, 3, 4]          # ≥5, no negociable
```
Tunear por walk-forward (SPEC-09), **jamás** in-sample.

## Multi-seed (NO negociable)
PPO tiene varianza enorme entre semillas. Reportar el mejor seed es multiple testing disfrazado.
- Entrenar **≥5 seeds por configuración**.
- Reportar **mediana e IQR** de las métricas OOS.
- **Decisión de graduación:** la *mediana* gana al baseline, no el mejor run.
- En Airflow: **dynamic task mapping** para paralelizar seeds (SPEC-10).

## VecNormalize (anti-leakage)
- `VecNormalize` sobre observaciones; **stats ajustadas solo con el train del fold** y guardadas por fold.
- En inferencia/val/test: `training=False`, `norm_reward=False`, cargar stats del fold.

## MLflow tracking (fuente del conteo de trials para el DSR)
Loggear por corrida:
- Params: todo `config/train.yaml` + `env.yaml` + `risk.yaml` (hash de config).
- Tags: `fold_id`, `seed`, `config_hash`, `git_sha`, `data_version` (DVC).
- Métricas OOS del fold (SPEC-09).
- Artefactos: modelo (`.zip` SB3), stats de `VecNormalize`, curvas de aprendizaje.
- **El registro de CADA configuración probada es lo que alimenta el número real de trials del Deflated Sharpe (SPEC-09).** Sin esto, el DSR es teatro.

## Interfaz
```python
def train_agent(fold: Fold, cfg: TrainConfig, seed: int) -> TrainedModel: ...
def train_multiseed(fold: Fold, cfg: TrainConfig) -> list[TrainedModel]: ...
```

## Criterios de aceptación
- [ ] Arquitectura LSTM→PPO entrena y converge en un fold de humo (smoke test corto).
- [ ] `CostModel.swap` implementado e incluido en el retorno (test: swap reduce PnL de posiciones overnight).
- [ ] `VecNormalize` con stats por fold, sin leakage (test).
- [ ] Entrena ≥5 seeds; agregación mediana/IQR implementada.
- [ ] MLflow loggea params/tags/métricas/artefactos + `config_hash` único por configuración.
- [ ] Reproducibilidad: mismo seed + misma config + mismos datos ⇒ mismas métricas (test).

## Dependencias
SPEC-05 (entorno), SPEC-03 (features), SPEC-09 (folds y métricas).
