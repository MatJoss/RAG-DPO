"""
Script de vérification de l'installation RAG-DPO
Vérifie Python, CUDA, Ollama, variables d'environnement, etc.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional

os.environ['PROJECT_ROOT']=r"E:\Projets\RAG-DPO"

class InstallationChecker:
    """Vérifie l'installation complète du système"""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def print_header(self, text: str):
        """Affiche un header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def check_item(self, name: str, status: bool, details: str = ""):
        """Affiche le résultat d'une vérification"""
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}")
        if details:
            print(f"   {details}")
        
        if status:
            self.checks.append(name)
        else:
            self.errors.append(name)
    
    def warn_item(self, name: str, message: str):
        """Affiche un avertissement"""
        print(f"⚠️  {name}")
        print(f"   {message}")
        self.warnings.append(name)
    
    def check_python_version(self) -> bool:
        """Vérifie la version Python"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor == 11:
            self.check_item("Python Version", True, f"Python {version_str} (recommandé)")
            return True
        elif version.major == 3 and version.minor >= 10:
            self.check_item("Python Version", True, f"Python {version_str} (compatible)")
            return True
        else:
            self.check_item("Python Version", False, f"Python {version_str} - Version 3.11 recommandée")
            return False
    
    def check_cuda(self) -> Tuple[bool, Optional[str]]:
        """Vérifie CUDA"""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            
            if result.returncode == 0:
                # Extraire version
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        version = line.split('release')[1].split(',')[0].strip()
                        self.check_item("CUDA Toolkit", True, f"Version {version}")
                        return True, version
            
            self.check_item("CUDA Toolkit", False, "nvcc introuvable")
            return False, None
        
        except FileNotFoundError:
            self.check_item("CUDA Toolkit", False, "CUDA non installé")
            return False, None
        except Exception as e:
            self.check_item("CUDA Toolkit", False, str(e))
            return False, None
    
    def check_cuda_path(self) -> bool:
        """Vérifie la variable CUDA_PATH"""
        cuda_path = os.getenv('CUDA_PATH')
        
        if cuda_path and Path(cuda_path).exists():
            self.check_item("CUDA_PATH", True, cuda_path)
            return True
        elif cuda_path:
            self.check_item("CUDA_PATH", False, f"{cuda_path} n'existe pas")
            return False
        else:
            self.warn_item("CUDA_PATH", "Variable non définie")
            return False
    
    def check_pytorch_cuda(self) -> bool:
        """Vérifie PyTorch et CUDA"""
        try:
            import torch
            
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.check_item("PyTorch CUDA", True, 
                              f"PyTorch {version} | CUDA {cuda_version} | GPU: {gpu_name}")
                return True
            else:
                self.warn_item("PyTorch CUDA", 
                             f"PyTorch {version} installé mais CUDA non disponible")
                return False
        
        except ImportError:
            self.check_item("PyTorch", False, "PyTorch non installé")
            return False
        except Exception as e:
            self.check_item("PyTorch CUDA", False, str(e))
            return False
    
    def check_ollama(self) -> bool:
        """Vérifie Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode == 0:
                models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] 
                         if line.strip()]
                
                if models:
                    self.check_item("Ollama", True, f"{len(models)} modèle(s) installé(s)")
                    
                    # Vérifier les modèles recommandés
                    if 'llama3.1:8b' in models:
                        self.check_item("  llama3.1:8b", True, "Modèle LLM présent")
                    else:
                        self.warn_item("  llama3.1:8b", "Modèle LLM non téléchargé")
                    
                    if 'nomic-embed-text:latest' in models:
                        self.check_item("  nomic-embed-text:latest", True, "Modèle embeddings présent")
                    else:
                        self.warn_item("  nomic-embed-text:latest", "Modèle embeddings non téléchargé")
                    
                    return True
                else:
                    self.warn_item("Ollama", "Installé mais aucun modèle téléchargé")
                    return False
            else:
                self.check_item("Ollama", False, "Service non démarré")
                return False
        
        except FileNotFoundError:
            self.check_item("Ollama", False, "Non installé")
            return False
        except Exception as e:
            self.check_item("Ollama", False, str(e))
            return False
    
    def check_env_vars(self) -> bool:
        """Vérifie les variables d'environnement"""
        required_vars = {
            'PROJECT_ROOT': 'Racine du projet',
            'OLLAMA_MODELS': 'Emplacement modèles Ollama',
            'TORCH_HOME': 'Cache PyTorch',
        }
        
        all_ok = True
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            
            if value:
                path = Path(value)
                if path.exists():
                    self.check_item(f"{var}", True, value)
                else:
                    self.warn_item(f"{var}", f"{value} (dossier n'existe pas encore)")
            else:
                self.warn_item(f"{var}", f"Variable non définie - {description}")
                all_ok = False
        
        return all_ok
    
    def check_project_structure(self) -> bool:
        """Vérifie la structure du projet"""
        project_root = os.getenv('PROJECT_ROOT', r"E:\Projets\RAG-DPO")
        
        if not project_root:
            self.warn_item("Structure Projet", "PROJECT_ROOT non défini")
            return False
        
        project_path = Path(project_root)
        
        if not project_path.exists():
            self.check_item("Structure Projet", False, f"{project_root} n'existe pas")
            return False
        
        required_dirs = ['data', 'models', 'vectordb', 'src', 'venv']
        missing = []
        
        for dir_name in required_dirs:
            if not (project_path / dir_name).exists():
                missing.append(dir_name)
        
        if missing:
            self.warn_item("Structure Projet", 
                         f"Dossiers manquants: {', '.join(missing)}")
            return False
        else:
            self.check_item("Structure Projet", True, "Tous les dossiers présents")
            return True
    
    def check_dependencies(self) -> bool:
        """Vérifie les dépendances Python critiques"""
        critical_packages = [
            ('langchain', 'LangChain'),
            ('chromadb', 'ChromaDB'),
            ('sentence_transformers', 'Sentence Transformers'),
            ('streamlit', 'Streamlit'),
            ('requests', 'Requests'),
            ('bs4', 'BeautifulSoup'),
        ]
        
        all_ok = True
        
        for package, name in critical_packages:
            try:
                __import__(package)
                self.check_item(f"  {name}", True, "Installé")
            except ImportError:
                self.check_item(f"  {name}", False, "Non installé")
                all_ok = False
        
        return all_ok
    
    def check_disk_space(self) -> bool:
        """Vérifie l'espace disque"""
        project_root = os.getenv('PROJECT_ROOT', '.')
        
        try:
            import shutil
            total, used, free = shutil.disk_usage(project_root)
            
            free_gb = free / (1024**3)
            
            if free_gb > 50:
                self.check_item("Espace Disque", True, f"{free_gb:.1f} GB libres")
                return True
            elif free_gb > 20:
                self.warn_item("Espace Disque", 
                             f"{free_gb:.1f} GB libres (50 GB recommandés)")
                return True
            else:
                self.check_item("Espace Disque", False, 
                              f"{free_gb:.1f} GB libres (insuffisant)")
                return False
        
        except Exception as e:
            self.warn_item("Espace Disque", f"Impossible de vérifier: {e}")
            return True
    
    def run_all_checks(self):
        """Lance toutes les vérifications"""
        self.print_header("🔍 VÉRIFICATION DE L'INSTALLATION RAG-DPO")
        
        print("\n📋 Vérifications système...")
        self.check_python_version()
        self.check_cuda()
        self.check_cuda_path()
        self.check_disk_space()
        
        print("\n📦 Vérifications packages Python...")
        self.check_pytorch_cuda()
        self.check_dependencies()
        
        print("\n🦙 Vérifications Ollama...")
        self.check_ollama()
        
        print("\n⚙️  Vérifications configuration...")
        self.check_env_vars()
        self.check_project_structure()
        
        # Résumé
        self.print_header("📊 RÉSUMÉ")
        
        print(f"\n✅ Vérifications réussies : {len(self.checks)}")
        print(f"⚠️  Avertissements : {len(self.warnings)}")
        print(f"❌ Erreurs : {len(self.errors)}")
        
        if self.errors:
            print("\n❌ Problèmes critiques détectés :")
            for error in self.errors:
                print(f"   - {error}")
            print("\n➡️  Consultez le guide d'installation pour corriger ces problèmes")
        
        if self.warnings:
            print("\n⚠️  Avertissements (non-bloquants) :")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n🎉 Installation parfaite ! Vous êtes prêt à démarrer.")
            print("\n📝 Prochaines étapes :")
            print("   1. Lancer le scraping : python src/scraping/cnil_scraper.py")
            print("   2. Traiter les données : python src/processing/process_data.py")
            print("   3. Lancer l'interface : streamlit run src/app.py")
        elif not self.errors:
            print("\n✅ Installation fonctionnelle avec quelques avertissements.")
            print("   Vous pouvez commencer le scraping.")


def main():
    checker = InstallationChecker()
    checker.run_all_checks()


if __name__ == "__main__":
    main()
