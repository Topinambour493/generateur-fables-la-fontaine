# generateur-fables-la-fontaine

Depuis generateur-fables


# Python 3.10 OBLIGATOIRE
```
python -m venv venv
venv\Scripts\activate
```

# 1. Installer PyTorch avec CUDA
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# 2. Installer les autres dépendances
```
pip install -r requirements.txt
```

Pour utiliser le modèle brut:
```
python test_modele_brut.py
```

Pour utiliser le modèle avec LoRA:
```
python usage_final.py
```