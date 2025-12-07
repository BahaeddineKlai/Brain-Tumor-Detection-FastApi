# Classification de Tumeurs Cérébrales – Explication du Code

Ce projet entraîne un modèle de deep learning pour classifier des images IRM du cerveau en plusieurs classes (types de tumeurs) en utilisant **PyTorch** et des modèles pré-entraînés.

Ce script est conçu pour fonctionner dans **Google Colab** et utilise les données stockées sur **Google Drive**.

---

## Structure du Dataset

Le code attend une structure de dossiers comme celle-ci :

```
brain_tumor_dataset/
  Training/
    Classe_A/
      img1.jpg
    Classe_B/
      img2.jpg
  Testing/
    Classe_A/
    Classe_B/
```

Chaque nom de dossier devient automatiquement une **classe**.

---

## Montage de Google Drive

Cette ligne connecte Colab à votre Google Drive :

```python
drive.mount('/content/drive')
```

Cela permet d’accéder aux images et d’y sauvegarder le modèle.

---

## Configuration du Script

Extrait du code :

```python
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Signification :

* `BATCH_SIZE` : nombre d’images traitées en même temps
* `NUM_EPOCHS` : nombre de passages complets sur le dataset
* `LR` : vitesse d’apprentissage du modèle
* `DEVICE` : GPU si disponible, sinon CPU

---

## Chargement des Images

Classe utilisée : `ImageFolderDataset`

Ce morceau de code détecte les classes :

```python
classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
self.class_to_idx[cls] = idx
```

Et charge une image ainsi :

```python
image = Image.open(path).convert("RGB")
return image, label
```

Le modèle reçoit donc `(image_tensor, label)`.

---

## Prétraitement des Images

### Transformations pour l’entraînement

```python
T.RandomResizedCrop(224)
T.RandomHorizontalFlip()
T.RandomRotation(15)
T.ToTensor()
T.Normalize(...)
```

Objectif : augmenter artificiellement la diversité des données pour éviter l’overfitting.

### Transformations pour la validation/test

```python
T.Resize((224, 224))
T.ToTensor()
T.Normalize(...)
```

Aucune transformation aléatoire pour avoir une évaluation stable.

---

## Création du Modèle

Le script essaie d’utiliser **EfficientNet-B0** :

```python
model = models.efficientnet_b0(weights=...)
model.classifier[1] = nn.Linear(in_features, num_classes)
```

Sinon, il bascule vers **ResNet50** :

```python
model = models.resnet50(weights=...)
model.fc = nn.Linear(in_features, num_classes)
```

C’est du **transfer learning**.

---

## Entraînement – Comment ça marche

Extrait clé :

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)

loss.backward()
optimizer.step()
```

Explication :

1. Reset des gradients
2. Passage avant dans le réseau
3. Calcul de l’erreur
4. Mise à jour des poids

Calcul de la précision :

```python
_, preds = outputs.max(1)
correct += (preds == labels).sum().item()
```

---

## Validation du Modèle

Pendant la validation, le modèle n’apprend pas :

```python
with torch.no_grad():
    outputs = model(images)
```

Les prédictions sont sauvegardées :

```python
all_preds.extend(preds.cpu().numpy().tolist())
```

---

## Sauvegarde des Labels

Les correspondances entre numéros et noms de classes :

```python
idx_to_class = {str(v): k for k, v in class_to_idx.items()}
json.dump(idx_to_class, f, indent=2)
```

---

## Sauvegarde du Meilleur Modèle

Seulement si la validation s’améliore :

```python
if val_acc > best_val_acc:
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "input_size": INPUT_SIZE
    }, MODEL_SAVE_PATH)
```

---

## Phase de Test

Si un dossier `Testing` existe :

```python
test_dataset = ImageFolderDataset(TEST_FOLDER, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

Puis évaluation complète.

---

## Résumé du Pipeline

Flux du programme :

```
Chargement des données → Augmentation → Création du modèle  
→ Entraînement → Validation → Sauvegarde → Test
```
---