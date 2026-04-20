# 🎙️ Reconnaissance d'Émotions Vocales — RAVDESS

**Auteurs : Eric Mantot & Mario Le Tinevez**  
*Module Machine Listening — ESEO*

## Description

Pipeline complet de reconnaissance d'émotions vocales sur le dataset RAVDESS 
(1 440 clips audio, 24 acteurs, 8 émotions).  
L'objectif : détecter automatiquement la colère, la joie, la tristesse, 
la peur et 4 autres émotions dans la voix — cas d'usage : centre d'appel intelligent.

## Émotions détectées

`neutral` `calm` `happy` `sad` `angry` `fearful` `disgust` `surprised`

## Modèles comparés

| Modèle | Accuracy |
|--------|----------|
| Random Forest (MFCC + RMS + ZCR) | ~54% |
| CNN from scratch (PyTorch) | ~62% |
| YAMNet + Dense (TensorFlow) | ~68% |
| CNN + SpecAugment | ~66% |

## Pipeline

1. Chargement RAVDESS depuis Zenodo
2. Preprocessing audio (resample 16kHz, normalisation, padding 3s)
3. Extraction features (Mel-spectrogramme, MFCC, RMS, ZCR)
4. Baseline ML (Random Forest, SVM, Gradient Boosting)
5. Split par acteur (anti speaker-leakage)
6. CNN PyTorch sur mel-spectrogrammes
7. Transfer Learning YAMNet (TensorFlow Hub)
8. SpecAugment (data augmentation audio)
9. Interface Gradio (micro + upload, lien public)

## Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Gradio](https://img.shields.io/badge/Gradio-4.x-green)

- `librosa` — analyse audio
- `YAMNet` — transfer learning (AudioSet)
- `Gradio` — interface web interactive
- `scikit-learn` — modèles baseline

## Lancer le notebook

```bash
# Sur Google Colab — ouvrir directement
# File → Open notebook → GitHub → coller l'URL du repo
```

## Résultats clés

- YAMNet est le meilleur modèle avec **~68% d'accuracy**
- Les confusions les plus fréquentes : `calm` ↔ `neutral`, `fearful` ↔ `surprised`
- Performance humaine sur ce dataset : ~60% — notre modèle fait mieux
