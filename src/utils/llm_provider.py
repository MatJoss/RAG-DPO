"""
Abstraction LLM Provider - Switch facile entre Local (Ollama) et Hybrid (Mistral)
Permet de changer de mode sans modifier le code métier
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Interface abstraite pour les providers LLM"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Génère une réponse"""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le provider est disponible"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Provider local utilisant Ollama"""
    
    def __init__(self, model: str = None, base_url: str = "http://localhost:11434", vision_model: str = "llava:7b"):
        # Modèle depuis .env ou paramètre ou défaut
        self.model = model or os.getenv('OLLAMA_MODEL', 'mistral-nemo')
        self.base_url = base_url
        self.vision_model = vision_model
        self.embedding_model = "nomic-embed-text"  # Modèle d'embeddings Ollama
        
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            logger.info(f"✓ Ollama initialisé avec {self.model}")
            
            # Vérifier si modèle vision disponible
            self.has_vision = self._check_vision_model()
            if self.has_vision:
                logger.info(f"✓ Modèle vision disponible : {self.vision_model}")
        except ImportError:
            logger.error("❌ Package 'ollama' non installé. Installez avec: pip install ollama")
            self.client = None
            self.has_vision = False
        except Exception as e:
            logger.error(f"❌ Erreur connexion Ollama: {e}")
            self.client = None
            self.has_vision = False
    
    def _check_vision_model(self) -> bool:
        """Vérifie si le modèle vision est disponible"""
        try:
            models = self.client.list()
            
            # Récupérer les noms de modèles (support différents formats de réponse)
            # if isinstance(models, dict) and 'models' in models:
                # model_list = models['models']
            # elif isinstance(models, list):
                # model_list = models
            # else:
                # logger.debug(f"Format de réponse inattendu : {type(models)}")
                # return False
            
            # Extraire noms (support 'name' ou 'model')
            model_names = []
            # for m in models:
                # if isinstance(m, dict):
                    # name = m.get('name') or m.get('model') or ''
                    # model_names.append(name)
                # elif isinstance(m, str):
                    # model_names.append(m)
            model_names = [m['model'] for m in models.get('models', [])]
            
            logger.debug(f"Modèles détectés : {model_names}")
            
            # Vérification flexible
            # 1. Match exact
            if self.vision_model in model_names:
                logger.debug(f"✓ Match exact : {self.vision_model}")
                return True
            
            # 2. Match partiel (ex: 'llava:7b' dans 'llava:7b-latest')
            vision_base = self.vision_model.split(':')[0]  # 'llava' depuis 'llava:7b'
            for name in model_names:
                if vision_base in name.lower():
                    logger.debug(f"✓ Match partiel : {name} contient {vision_base}")
                    return True
            
            logger.debug(f"✗ Aucun match pour {self.vision_model} dans {model_names}")
            return False
            
        except Exception as e:
            logger.debug(f"Erreur vérification modèle vision : {e}")
            return False
    
    def is_available(self) -> bool:
        """Vérifie si Ollama est disponible"""
        if self.client is None:
            return False
        
        try:
            # Test de connexion
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f"⚠️  Ollama non disponible: {e}")
            return False
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048, **kwargs) -> str:
        """Génère une réponse avec Ollama (mode completion simple)."""
        if not self.is_available():
            raise RuntimeError("Ollama n'est pas disponible. Lancez 'ollama serve'")
        
        model = kwargs.get('model', self.model)
        response_format = kwargs.get('format', None)
        
        try:
            gen_kwargs = dict(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'num_ctx': 16384,
                }
            )
            if response_format:
                gen_kwargs['format'] = response_format
            
            response = self.client.generate(**gen_kwargs)
            return response['response']
        
        except Exception as e:
            logger.error(f"❌ Erreur génération Ollama: {e}")
            raise
    
    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 2048, **kwargs) -> str:
        """Génère via l'API chat (optimisé pour instruction-tuned models).
        
        Args:
            messages: Liste de dicts [{'role': 'system'|'user'|'assistant', 'content': str}]
            temperature: Température de génération
            max_tokens: Nombre max de tokens générés
        
        Returns:
            Texte de la réponse
        """
        if not self.is_available():
            raise RuntimeError("Ollama n'est pas disponible. Lancez 'ollama serve'")
        
        model = kwargs.get('model', self.model)
        
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'num_ctx': 16384,  # 16K tokens — suffisant pour nos contextes, rapide
                }
            )
            return response['message']['content']
        
        except Exception as e:
            logger.error(f"❌ Erreur chat Ollama: {e}")
            raise
    
    def generate_with_image(self, prompt: str, image_path: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """
        Génère une réponse avec analyse d'image (vision)
        
        Args:
            prompt: Question/instruction
            image_path: Chemin vers l'image
            temperature: Température de génération
            max_tokens: Nombre max de tokens
        
        Returns:
            Réponse du modèle vision
        """
        if not self.is_available():
            raise RuntimeError("Ollama n'est pas disponible")
        
        if not self.has_vision:
            raise RuntimeError(f"Modèle vision {self.vision_model} non disponible. Installez avec: ollama pull {self.vision_model}")
        
        try:
            response = self.client.chat(
                model=self.vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }],
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            return response['message']['content']
        
        except Exception as e:
            logger.error(f"❌ Erreur génération vision Ollama: {e}")
            raise
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings avec Ollama"""
        if not self.is_available():
            raise RuntimeError("Ollama n'est pas disponible")
        
        try:
            embeddings = []
            for text in texts:
                # Tronquer si trop long (contexte Ollama limité)
                text_truncated = text[:2000]
                
                response = self.client.embeddings(
                    model=self.embedding_model,
                    prompt=text_truncated
                )
                embeddings.append(response['embedding'])
            
            return embeddings
        
        except Exception as e:
            logger.error(f"❌ Erreur embeddings Ollama: {e}")
            raise

class MistralProvider(BaseLLMProvider):
    """Provider hybride utilisant Mistral AI (hébergé EU)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-small-latest"):
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        self.model = model
        
        if not self.api_key:
            logger.warning("⚠️  MISTRAL_API_KEY non défini dans .env")
            self.client = None
        else:
            try:
                from mistralai import Mistral
                self.client = Mistral(api_key=self.api_key)
                logger.info(f"✓ Mistral AI initialisé avec {model}")
            except ImportError:
                logger.error("❌ Package 'mistralai' non installé. Installez avec: pip install mistralai")
                self.client = None
    
    def is_available(self) -> bool:
        """Vérifie si Mistral est disponible"""
        return self.client is not None and self.api_key is not None
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096, **kwargs) -> str:
        """Génère une réponse avec Mistral"""
        if not self.is_available():
            raise RuntimeError("Mistral API non disponible. Vérifiez MISTRAL_API_KEY dans .env")
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Erreur génération Mistral: {e}")
            raise
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings avec Mistral"""
        if not self.is_available():
            raise RuntimeError("Mistral API non disponible")
        
        try:
            embeddings_batch = self.client.embeddings.create(
                model="mistral-embed",
                inputs=texts
            )
            
            return [item.embedding for item in embeddings_batch.data]
        
        except Exception as e:
            logger.error(f"❌ Erreur embeddings Mistral: {e}")
            raise


class LLMFactory:
    """Factory pour créer le bon provider selon la config"""
    
    @staticmethod
    def create(mode: str = None) -> BaseLLMProvider:
        """
        Crée un provider LLM selon le mode
        
        Args:
            mode: 'local' ou 'hybrid'. Si None, lit depuis .env
        
        Returns:
            Instance de BaseLLMProvider
        """
        if mode is None:
            load_dotenv()
            mode = os.getenv('MODE', 'local').lower()
        
        logger.info(f"🔧 Initialisation du provider en mode: {mode}")
        
        if mode == 'local':
            # Lire modèle et vision depuis .env
            model = os.getenv('OLLAMA_MODEL', 'mistral-nemo')
            vision_model = os.getenv('OLLAMA_VISION_MODEL', 'llava:7b')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            provider = OllamaProvider(model=model, base_url=base_url, vision_model=vision_model)
            if not provider.is_available():
                logger.warning("⚠️  Ollama non disponible. Installez avec: curl -fsSL https://ollama.com/install.sh | sh")
                logger.warning("⚠️  Puis lancez: ollama serve")
                logger.warning(f"⚠️  Et téléchargez le modèle: ollama pull {model}")
            return provider
        
        elif mode == 'hybrid':
            model = os.getenv('MISTRAL_MODEL', 'mistral-small-latest')
            provider = MistralProvider(model=model)
            if not provider.is_available():
                logger.warning("⚠️  Mistral API non disponible. Configurez MISTRAL_API_KEY dans .env")
                logger.warning("⚠️  Obtenez une clé sur: https://console.mistral.ai/")
            return provider
        
        else:
            raise ValueError(f"Mode invalide: {mode}. Utilisez 'local' ou 'hybrid'")


class RAGConfig:
    """Configuration centralisée du système RAG"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()
        
        # Mode
        self.mode = os.getenv('MODE', 'local')
        
        # Paths
        self.project_root = os.getenv('PROJECT_ROOT', '.')
        self.data_path = os.getenv('DATA_PATH', './data')
        self.vectordb_path = os.getenv('VECTORDB_PATH', './vectordb')
        self.models_path = os.getenv('MODELS_PATH', './models')
        
        # RAG params
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 128))
        self.top_k = int(os.getenv('TOP_K_RESULTS', 5))
        
        # LLM
        self.llm_provider = LLMFactory.create(self.mode)
        
        logger.info(f"✓ Configuration chargée en mode {self.mode}")
    
    def switch_mode(self, new_mode: str):
        """Change le mode (local <-> hybrid) à la volée"""
        logger.info(f"🔄 Changement de mode: {self.mode} → {new_mode}")
        self.mode = new_mode
        self.llm_provider = LLMFactory.create(new_mode)
        
        # Mettre à jour le .env
        env_file = os.path.join(self.project_root, '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(env_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.startswith('MODE='):
                        f.write(f'MODE={new_mode}\n')
                    else:
                        f.write(line)
        
        logger.info(f"✓ Mode changé vers {new_mode}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Test des providers
    
    print("=" * 60)
    print("🧪 Test des LLM Providers")
    print("=" * 60)
    
    # Test Local
    print("\n1️⃣  Test Ollama (Local)")
    try:
        local_provider = OllamaProvider()
        if local_provider.is_available():
            response = local_provider.generate("Explique en une phrase ce qu'est le RGPD.")
            print(f"✓ Réponse: {response[:100]}...")
        else:
            print("❌ Ollama non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    # Test Hybrid
    print("\n2️⃣  Test Mistral (Hybrid)")
    try:
        hybrid_provider = MistralProvider()
        if hybrid_provider.is_available():
            response = hybrid_provider.generate("Explique en une phrase ce qu'est le RGPD.")
            print(f"✓ Réponse: {response[:100]}...")
        else:
            print("❌ Mistral API non disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    # Test Factory
    print("\n3️⃣  Test Factory")
    config = RAGConfig()
    print(f"✓ Mode actuel: {config.mode}")
    print(f"✓ Provider: {type(config.llm_provider).__name__}")
    
    print("\n" + "=" * 60)