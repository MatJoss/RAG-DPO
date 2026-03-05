# 🐳 Guide Docker — RAG-DPO

## Table des matières

1. [Concepts clés (pour comprendre)](#-concepts-clés)
2. [Prérequis](#-prérequis)
3. [Architecture Docker](#-architecture-docker)
4. [Persistence des données](#-persistence-des-données)
5. [Lancement pas à pas](#-lancement-pas-à-pas)
6. [Commandes utiles](#-commandes-utiles)
7. [Troubleshooting](#-troubleshooting)

---

## 🧠 Concepts clés

### Qu'est-ce que Docker fait ici ?

Docker crée des **conteneurs** — des mini-machines Linux isolées qui tournent sur ton PC Windows. Au lieu d'installer Python, Ollama, CUDA manuellement, Docker empaquette tout et lance le stack en une commande.

```
Sans Docker (aujourd'hui)          Avec Docker
─────────────────────────          ──────────────
Windows                            Windows
├── Python 3.11 (installé)         └── Docker Desktop
├── pip install ... (installé)         ├── Conteneur "ollama" (Linux)
├── Ollama (installé)                  │   └── Ollama + mistral-nemo
├── Streamlit (installé)               └── Conteneur "app" (Linux)
└── Ton code                               └── Python + Streamlit + ton code
```

### Vocabulaire en 30 secondes

| Terme | Signification |
|-------|---------------|
| **Image** | Recette de cuisine (Dockerfile) → produit un plat prêt à servir |
| **Conteneur** | Un plat servi = une image qui tourne (tu peux en avoir plusieurs) |
| **Volume** | Un dossier de ton PC monté dans le conteneur (données persistantes) |
| **docker-compose** | Chef d'orchestre qui lance plusieurs conteneurs ensemble |
| **Bind mount** | `./logs:/app/logs` = "le dossier `logs` de mon PC = le dossier `/app/logs` du conteneur" |
| **Volume nommé** | Dossier géré par Docker (pas visible directement dans l'explorateur Windows) |

### Comment ça communique ?

```
Ton navigateur                Docker (réseau interne)
──────────────                ────────────────────────
http://localhost:8501  ──────→  app (Streamlit, port 8501)
                                  │
                                  │ http://ollama:11434
                                  │ (nom du service = hostname)
                                  ▼
                               ollama (LLM, GPU)
```

**Point crucial** : dans Docker, `localhost` veut dire "moi-même" pour chaque conteneur. L'app ne peut pas dire `http://localhost:11434` pour joindre Ollama car ce serait chercher dans son propre conteneur. Elle dit `http://ollama:11434` — Docker résout le nom `ollama` vers le bon conteneur automatiquement.

C'est pour ça qu'on a la variable d'environnement `OLLAMA_BASE_URL`.

### Et mes données, elles sont où ?

**Tout persiste.** Quand tu fais `docker compose down` puis `docker compose up`, rien n'est perdu :

| Donnée | Où c'est stocké | Type de montage |
|--------|-----------------|-----------------|
| Modèles Ollama (~8 GB) | Volume Docker `rag-dpo-ollama-data` | Volume nommé |
| ChromaDB (index vectoriel) | `./data/vectordb/` (ton disque) | Bind mount |
| Modèles HuggingFace (~2 GB) | `./models/` (ton disque) | Bind mount |
| Logs / queries / feedback | `./logs/` (ton disque) | Bind mount |
| Config | `./configs/` (ton disque) | Bind mount |
| Documents entreprise | `./data/keep/` (ton disque) | Bind mount |

**Bind mount** = tu vois les fichiers dans l'explorateur Windows, tu peux les éditer.
**Volume nommé** = géré par Docker, pas directement visible (mais persiste).

---

## 📋 Prérequis

### 1. Docker Desktop (Windows)

1. **Télécharger** : https://www.docker.com/products/docker-desktop/
2. **Installer** en cochant ✅ "Use WSL 2 instead of Hyper-V"
3. **Redémarrer** le PC si demandé
4. **Lancer Docker Desktop**

```powershell
docker --version          # Docker version 27.x.x
docker compose version    # Docker Compose version v2.x.x
```

### 2. WSL 2

Si Docker Desktop le demande :

```powershell
# PowerShell Administrateur
wsl --install
# Redémarrer, puis :
wsl --version
```

### 3. GPU dans Docker (NVIDIA)

```powershell
# Vérifier que le GPU est visible dans Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Tu dois voir ta RTX 4070 Ti. ✅ Si ça échoue → mettre à jour le driver NVIDIA et redémarrer.

> **Note** : Pas besoin d'installer CUDA dans WSL. Le driver Windows suffit.

### 4. Checklist rapide

```powershell
# 1. Docker fonctionne
docker --version

# 2. Docker Compose fonctionne
docker compose version

# 3. Docker peut lancer des conteneurs
docker run --rm hello-world

# 4. GPU accessible dans Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 5. Ollama démarre dans Docker
docker run --rm --gpus all ollama/ollama --version
```

> ⚠️ Attention au test 5 : c'est `ollama/ollama --version` (pas `ollama/ollama ollama --version` — l'image a déjà `ollama` comme entrypoint).

---

## 📐 Architecture Docker

### Fichiers créés

```
RAG-DPO/
├── Dockerfile              # Image CPU-only (embeddings sur CPU)
├── Dockerfile.cuda         # Image GPU + flash-attn (embeddings + reranker sur CUDA)
├── docker-compose.yml      # Orchestre Ollama + App (CPU)
├── docker-compose.gpu.yml  # Override pour app GPU (flash-attn, CUDA embeddings)
├── .env.docker             # Variables d'env pour Docker
├── .env.example            # Template de configuration
├── .dockerignore           # Fichiers exclus du build
└── src/utils/paths.py      # Chemins centralisés (env vars → défauts locaux)
```

### Variante GPU pour l'app (flash-attn)

Par défaut, le conteneur `app` tourne en **CPU-only** (suffisant, ~2s pour embeddings).
Si tu as un GPU dédié aux embeddings/reranker, utilise la variante GPU :

```bash
# Lancer avec GPU + flash-attn pour l'app
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Rebuild si c'est la première fois
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

> **⚠️ Le build de `flash-attn` prend ~10 min** (compilation C++/CUDA). L'image est ensuite cachée.
>
> **Note** : `flash-attn` ne fonctionne que sur Linux avec GPU NVIDIA (CUDA). C'est le cas dans Docker même sur un host Windows.

### Les 2 conteneurs

```
docker-compose.yml
│
├── ollama (GPU)
│   ├── Image officielle : ollama/ollama:latest
│   ├── Ta RTX 4070 Ti (GPU passthrough)
│   ├── Volume : ollama_data → modèles persistants
│   └── Port : 11434 (interne + exposé pour debug)
│
└── app (CPU)
    ├── Image custom : construite depuis ton Dockerfile
    ├── Python 3.11 + toutes les dépendances pip
    ├── Streamlit (port 8501 → ton navigateur)
    ├── Volumes montés :
    │   ├── data/vectordb → ChromaDB
    │   ├── models → HuggingFace cache
    │   ├── logs → logs + queries + feedback
    │   ├── configs → config.yaml
    │   ├── data/keep → documents entreprise
    │   └── data/metadata → métadonnées
    └── Env : OLLAMA_BASE_URL=http://ollama:11434
```

### Comment le code gère local vs Docker

Le module `src/utils/paths.py` centralise tout :

```python
# paths.py lit les variables d'environnement, avec défauts "locaux"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
#                                 ▲ Docker injecte ça              ▲ défaut local
```

| Situation | `OLLAMA_BASE_URL` | Source |
|-----------|-------------------|--------|
| `streamlit run app.py` (Windows) | `http://localhost:11434` | Défaut (pas de var d'env) |
| `docker compose up` | `http://ollama:11434` | Injecté par docker-compose.yml |

**Même code, zéro branche `if docker`. Pas besoin de deux `app.py`.**

---

## 💾 Persistence des données

### Ce qui persiste (volumes)

```
Ton PC Windows                         Conteneur Docker
─────────────────                      ──────────────────
E:\Projets\RAG-DPO\data\vectordb\ ──→ /app/data/vectordb/    ← ChromaDB
E:\Projets\RAG-DPO\models\        ──→ /app/models/           ← BGE-M3, reranker
E:\Projets\RAG-DPO\logs\          ──→ /app/logs/             ← queries, feedback
E:\Projets\RAG-DPO\configs\       ──→ /app/configs/          ← config.yaml
E:\Projets\RAG-DPO\data\keep\     ──→ /app/data/keep/        ← docs entreprise
E:\Projets\RAG-DPO\data\metadata\ ──→ /app/data/metadata/    ← résumés, classif
Volume Docker "rag-dpo-ollama-data" → /root/.ollama           ← mistral-nemo (8 GB)
```

### Ce qui est reconstruit (dans l'image)

- Dépendances pip (`requirements.txt`) → reconstruites au `docker build`
- Code source Python → copié dans l'image

### Cycle de vie

```
docker compose up     → Crée les conteneurs, monte les volumes
                        → Ollama démarre, charge les modèles
                        → App démarre, Streamlit tourne
                        → Tu utilises http://localhost:8501

docker compose down   → Arrête les conteneurs
                        → Volumes toujours là ✅
                        → Rien n'est perdu

docker compose up     → Redémarre — tout est instantané (modèles déjà là)
```

---

## 🚀 Lancement pas à pas

### Étape 1 : Construire l'image de l'app

```powershell
cd E:\Projets\RAG-DPO

# Construire l'image (première fois : ~5-10 min pour les dépendances pip)
docker compose build
```

### Étape 2 : Lancer le stack

```powershell
# Lancer Ollama + App en arrière-plan
docker compose up -d
```

La première fois, Docker va :
1. Télécharger l'image `ollama/ollama` si pas déjà fait
2. Démarrer Ollama (GPU)
3. Attendre que Ollama soit healthy (healthcheck)
4. Démarrer l'app Streamlit

### Étape 3 : Télécharger le modèle Ollama (première fois uniquement)

```powershell
# Télécharger mistral-nemo dans le conteneur Ollama (~7 GB, une seule fois)
docker exec rag-dpo-ollama ollama pull mistral-nemo

# Vérifier que le modèle est là
docker exec rag-dpo-ollama ollama list
```

> **Ce modèle persiste** dans le volume `rag-dpo-ollama-data`. Tu ne le retélécharges jamais sauf si tu supprimes explicitement le volume.

### Étape 4 : Ouvrir l'app

Ouvre ton navigateur :

```
http://localhost:8501
```

🎉 C'est tout !

---

## 📝 Commandes utiles

### Gestion du stack

```powershell
# Lancer (arrière-plan)
docker compose up -d

# Voir les logs en temps réel
docker compose logs -f

# Logs de l'app seulement
docker compose logs -f app

# Logs d'Ollama seulement
docker compose logs -f ollama

# Arrêter tout (données persistent)
docker compose down

# Arrêter + supprimer les volumes (⚠️ PERTE DE DONNÉES !)
docker compose down -v

# Reconstruire l'image après modif de requirements.txt
docker compose build app
docker compose up -d

# Reconstruire sans cache (si problème de dépendances)
docker compose build --no-cache app
```

### Debug

```powershell
# Ouvrir un shell dans le conteneur app
docker exec -it rag-dpo-app bash

# Tester la connexion Ollama depuis l'app
docker exec rag-dpo-app curl http://ollama:11434/api/tags

# Vérifier l'état des conteneurs
docker compose ps

# Voir l'utilisation mémoire/CPU
docker stats
```

### Modèles Ollama

```powershell
# Lister les modèles installés
docker exec rag-dpo-ollama ollama list

# Télécharger un autre modèle
docker exec rag-dpo-ollama ollama pull llama3.1:8b

# Supprimer un modèle
docker exec rag-dpo-ollama ollama rm llama3.1:8b
```

---

## 🔧 Troubleshooting

### "VectorDB introuvable"

L'app cherche `data/vectordb/chromadb/` qui doit contenir la base ChromaDB construite au préalable. Ce dossier est monté en volume depuis ton PC.

```powershell
# Vérifier que le dossier existe et n'est pas vide
dir E:\Projets\RAG-DPO\data\vectordb\chromadb\
```

Si vide → tu dois d'abord exécuter le pipeline de construction en local (`python rebuild_pipeline.py`).

### "Connection refused" vers Ollama

```powershell
# Vérifier que le conteneur Ollama tourne
docker compose ps

# Vérifier les logs Ollama
docker compose logs ollama

# Tester depuis le conteneur app
docker exec rag-dpo-app curl http://ollama:11434/api/tags
```

### L'app est lente au premier démarrage

Normal : BGE-M3 (~1.5 GB) et le reranker Jina (~1 GB) sont téléchargés dans `./models/` au premier lancement. Les suivants sont instantanés car le cache est monté en volume.

### Docker Desktop prend trop de RAM

Créer/éditer `%UserProfile%\.wslconfig` :

```ini
[wsl2]
memory=8GB
processors=4
```

Puis :

```powershell
wsl --shutdown
# Relancer Docker Desktop
```

### Le build est très long

- **Premier build** : ~5-10 min (pip install). Normal.
- **Rebuilds suivants** : ~30 sec (cache Docker des couches pip).
- Si seul `requirements.txt` change → seules les couches pip sont refaites.

### Modifier le code sans reconstruire

Le code est copié dans l'image au build. Pour du **développement actif**, monte le code en volume. Ajoute temporairement dans `docker-compose.yml` sous `app.volumes` :

```yaml
- .:/app  # Monte tout le code source (dev only)
```

Puis un simple `docker compose restart app` suffit après chaque modification Python.

### Port 8501 déjà utilisé

Si Streamlit tourne déjà en local, change le port :

```powershell
# Éditer .env.docker
# STREAMLIT_PORT=8502
# Puis relancer
docker compose up -d
```

### Erreur `ollama ollama --version`

L'image `ollama/ollama` a déjà `ollama` comme entrypoint. Donc :

```powershell
# ❌ Mauvais (exécute "ollama ollama --version")
docker run --rm ollama/ollama ollama --version

# ✅ Correct
docker run --rm ollama/ollama --version
```

---

## 📐 Résumé : local vs Docker

| | Local (Windows) | Docker |
|--|-----------------|--------|
| **Lancer** | `streamlit run app.py` | `docker compose up -d` |
| **Ollama** | Installé sur Windows | Conteneur GPU dédié |
| **ChromaDB** | `data/vectordb/chromadb/` | Même dossier (bind mount) |
| **Modèles HF** | `models/huggingface/hub/` | Même dossier (bind mount) |
| **Logs** | `logs/` | Même dossier (bind mount) |
| **Config** | Aucune var d'env nécessaire | `OLLAMA_BASE_URL=http://ollama:11434` |
| **Code Python** | Identique | Identique |
| **Port** | `localhost:8501` | `localhost:8501` |
| **GPU** | Ollama direct | Ollama via NVIDIA Container Toolkit |
