# Projekt 2 – Sports Image Classification

Tento adresár obsahuje druhý semestrálny projekt z predmetu *Neuronové siete*.

## Cieľ

Vybudovať **CNN klasifikátor športových obrázkov** v **PyTorch** s podporou
**CUDA/GPU**. Tréning prebieha cez `torch.utils.data.DataLoader`, model používa
konvolučné vrstvy, augmentácie sú riešené cez `torchvision.transforms`.

## Štruktúra

- `sports_image_eda.ipynb` – EDA notebook:
  - indexuje dataset (Kaggle alebo lokálne súbory),
  - analyzuje triedy, rozlíšenia, poškodené obrázky, duplicity,
  - počíta základné štatistiky (jas, RGB kanály) a odporúča preprocessing.
- `sports_src/` – Python balík s kódom pre Projekt 2:
  - `data.py` – indexing datasetu, filtrovane obrázkov, split, `DataLoader` a augmentácie.
  - `model.py` – CNN architektúra pre obrazovú klasifikáciu.
  - `trainer.py` – PyTorch tréningový loop s CUDA podporou, metriky a early stopping.
  - `experiments.py` – helper `run_experiment(...)` pre spúšťanie experimentov.
  - `visualization.py` – grafy tréningovej histórie a confusion matrix.
- `sports_image_experiments.ipynb` – notebook na tréning, experimenty a evaluáciu.

## Požiadavky

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `Pillow`

Ak chceš používať GPU/CUDA, nainštaluj **CUDA build PyTorch** podľa oficiálneho
inštalačného príkazu z `pytorch.org` pre tvoju verziu CUDA a ovládačov.
Bežné `pip install torch torchvision` môže v niektorých prostrediach nainštalovať
iba CPU build.

## Ako spustiť

1. Nainštaluj závislosti (z koreňa repozitára):

   ```bash
   pip install -r requirements.txt
   ```

2. Spusť **EDA** notebook a over, že dataset je správne stiahnutý a načítaný.

3. Spusť **experiments** notebook:

   - nastav `dataset_root`,
   - spusti bunky v poradí,
   - ak je dostupná NVIDIA GPU a CUDA build PyTorch, tréning sa automaticky presunie na GPU.

Výsledky (história tréningu, metriky, grafy) sú viditeľné priamo v notebooku;
voliteľne ich možno ukladať do `results/logs` a `results/plots`.
