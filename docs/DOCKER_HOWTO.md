# 🐳 Guide d'installation Docker — RAG-DPO

## Pourquoi Docker ?

Docker va permettre de packager **tout le stack** (Streamlit + Ollama + ChromaDB) dans des conteneurs isolés, reproductibles et déployables partout. Plus besoin de configurer Python, CUDA, Ollama manuellement.

---

## 📋 Prérequis à installer

### 1. Docker Desktop (Windows)

1. **Télécharger** : https://www.docker.com/products/docker-desktop/
2. **Lancer l'installateur** et cocher :
   - ✅ "Use WSL 2 instead of Hyper-V" (recommandé)
   - ✅ "Add shortcut to desktop"
3. **Redémarrer** le PC si demandé
4. **Lancer Docker Desktop** depuis le menu Démarrer
5. **Vérifier** dans un terminal PowerShell :

```powershell
docker --version
# Docker version 27.x.x, build xxxxx

docker compose version
# Docker Compose version v2.x.x
```

> **⚠️ WSL 2 requis** : Si tu n'as pas WSL 2, Docker Desktop te proposera de l'installer. Accepte — c'est nécessaire pour les conteneurs Linux sur Windows.

### 2. WSL 2 (si pas déjà installé)

Si Docker Desktop signale que WSL 2 n'est pas disponible :

```powershell
# Dans PowerShell en Administrateur
wsl --install
# Redémarrer le PC
# Puis vérifier :
wsl --version
```

### 3. NVIDIA Container Toolkit (pour GPU dans Docker)

C'est **le point critique** : pour que Ollama utilise ta RTX 4070 Ti dans Docker, il faut le support GPU.

#### a) Vérifier le driver NVIDIA

```powershell
nvidia-smi
# Doit afficher ta RTX 4070 Ti avec CUDA 12.x
```

#### b) Installer le support GPU Docker (via WSL)

Le support GPU dans Docker Desktop Windows passe par WSL 2. Avec les drivers NVIDIA récents (≥ 528.xx), **c'est automatique** — Docker Desktop détecte le GPU via WSL.

**Vérifier que ça marche :**

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Tu dois voir ta RTX 4070 Ti dans la sortie. Si ça marche → tu es prêt ! 🎉

**Si ça échoue :**

1. **Mettre à jour le driver NVIDIA** : https://www.nvidia.com/Download/index.aspx
   - Prendre le dernier Game Ready ou Studio Driver pour ta RTX 4070 Ti
2. **Redémarrer** le PC
3. **Relancer Docker Desktop**
4. **Réessayer** la commande `docker run --gpus all`

> **Note** : Tu n'as PAS besoin d'installer CUDA Toolkit dans WSL. Le driver Windows suffit — Docker accède au GPU via le driver hôte.

---

## ✅ Checklist de vérification

Lance ces commandes dans PowerShell et vérifie que tout est ✅ :

```powershell
# 1. Docker fonctionne
docker --version
# ✅ Docker version 27.x.x

# 2. Docker Compose fonctionne
docker compose version
# ✅ Docker Compose version v2.x.x

# 3. Docker peut lancer des conteneurs
docker run --rm hello-world
# ✅ "Hello from Docker!"

# 4. GPU accessible dans Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# ✅ Affiche ta RTX 4070 Ti

# 5. Test rapide Ollama dans Docker
docker run --rm --gpus all ollama/ollama ollama --version
# ✅ ollama version x.x.x
```

Si les 5 tests passent → on est prêt pour la phase Docker du projet ! 🚀

---

## 🔧 Troubleshooting

### "docker: command not found"
→ Docker Desktop n'est pas lancé ou pas dans le PATH. Relancer Docker Desktop.

### "error during connect: ... Is the docker daemon running?"
→ Docker Desktop n'est pas démarré. Le lancer depuis le menu Démarrer.

### "docker: Error response from daemon: could not select device driver"
→ Le support GPU n'est pas configuré. Mettre à jour le driver NVIDIA et redémarrer.

### "WSL 2 is not installed"
→ Exécuter `wsl --install` dans PowerShell Administrateur, puis redémarrer.

### Docker Desktop prend beaucoup de RAM
→ Créer/éditer `%UserProfile%\.wslconfig` :
```ini
[wsl2]
memory=8GB
processors=4
```
Puis `wsl --shutdown` et relancer Docker Desktop.

### Le pull du modèle Ollama est très lent
→ C'est normal pour Mistral-Nemo (~7 GB). La première fois seulement. Le modèle sera persisté dans un volume Docker.

---

## 📐 Architecture Docker prévue

```
docker-compose.yml
├── ollama          # LLM + modèles (GPU, volume persistant)
├── app             # Streamlit multipage (CPU, monte le code source)
└── (chromadb)      # Optionnel : service séparé ou intégré dans app
```

- **Ollama** : conteneur GPU avec volume pour les modèles (~8 GB)
- **App Streamlit** : conteneur CPU avec le code Python + dépendances
- **ChromaDB** : intégré dans le conteneur app (PersistentClient, volume pour les données)
- **Volumes** : `ollama_data`, `chromadb_data`, `logs` persistés entre redémarrages

---

## ⏭️ Prochaines étapes (une fois Docker installé)

1. Je crée le `Dockerfile` + `docker-compose.yml`
2. `docker compose build` — construit les images
3. `docker compose up` — lance le stack
4. Ouvrir `http://localhost:8501` — Streamlit tourne !
5. Test end-to-end dans Docker
6. Push final sur GitHub

**Dis-moi quand les 5 tests de la checklist passent et on attaque !** 🐳
