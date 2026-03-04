"""
Generator - Génération de réponses via LLM
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Réponse générée avec métadonnées"""
    text: str
    model: str
    tokens_used: Optional[int] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


class Generator:
    """
    Génère des réponses via LLM local (Ollama)
    
    Gère :
    - Appel au LLM avec prompt formaté
    - Gestion des erreurs
    - Paramètres de génération (temperature, max_tokens, etc.)
    """
    
    def __init__(
        self,
        llm_provider,
        model: str = "mistral-nemo",
        temperature: float = 0.1,  # Basse pour réponses factuelles
        max_tokens: int = 3000
    ):
        """
        Args:
            llm_provider: Provider LLM (OllamaProvider)
            model: Nom du modèle
            temperature: Temperature de génération (0.0-1.0)
            max_tokens: Nombre max de tokens générés
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedResponse:
        """
        Génère une réponse
        
        Args:
            system_prompt: Prompt système (instructions)
            user_prompt: Prompt utilisateur (contexte + question)
            temperature: Override de la temperature
            max_tokens: Override du max_tokens
        
        Returns:
            GeneratedResponse avec le texte et métadonnées
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"🤖 Génération réponse avec {self.model} (temp={temp}, max_tokens={max_tok})")
        
        # Construction des messages pour l'API chat
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            import time
            start_time = time.time()
            
            # Appel au LLM via chat() — optimisé pour instruction-tuned models
            response_text = self.llm_provider.chat(
                messages=messages,
                model=self.model,
                temperature=temp,
                max_tokens=max_tok
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Réponse générée en {generation_time:.2f}s ({len(response_text)} chars)")
            
            return GeneratedResponse(
                text=response_text,
                model=self.model,
                generation_time=generation_time
            )
        
        except Exception as e:
            logger.error(f"❌ Erreur génération: {e}")
            return GeneratedResponse(
                text="",
                model=self.model,
                error=str(e)
            )
    
    def generate_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedResponse:
        """
        Génère avec historique de conversation
        
        Args:
            system_prompt: Prompt système
            messages: Liste de messages [{role: 'user'|'assistant', content: str}]
            temperature: Override temperature
            max_tokens: Override max_tokens
        
        Returns:
            GeneratedResponse
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"🤖 Génération avec historique ({len(messages)} messages)")
        
        try:
            import time
            start_time = time.time()
            
            # Construction messages pour l'API chat
            chat_messages = [{'role': 'system', 'content': system_prompt}]
            for msg in messages:
                chat_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            # Appel LLM via chat()
            response_text = self.llm_provider.chat(
                messages=chat_messages,
                model=self.model,
                temperature=temp,
                max_tokens=max_tok
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"✅ Réponse générée en {generation_time:.2f}s")
            
            return GeneratedResponse(
                text=response_text,
                model=self.model,
                generation_time=generation_time
            )
        
        except Exception as e:
            logger.error(f"❌ Erreur génération avec historique: {e}")
            return GeneratedResponse(
                text="",
                model=self.model,
                error=str(e)
            )


def create_generator(
    llm_provider,
    model: str = "mistral-nemo",
    temperature: float = 0.1,
    max_tokens: int = 3000
) -> Generator:
    """Factory function pour créer un generator"""
    return Generator(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
