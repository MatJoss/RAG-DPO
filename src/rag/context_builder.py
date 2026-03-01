"""
Context Builder - Construction du contexte pour le LLM
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .intent_classifier import QuestionIntent

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Construit le prompt pour le LLM à partir des chunks récupérés
    
    Gère :
    - Template de prompt DPO
    - Injection des chunks avec métadonnées
    - Gestion de la limite de tokens avec résumé intelligent
    - Formatage des sources pour citations
    """
    
    # Prompt système optimisé pour précision maximale DPO/RGPD
    SYSTEM_PROMPT = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu réponds UNIQUEMENT à partir des sources fournies dans le contexte.

RÈGLES NON NÉGOCIABLES :
1. CHAQUE affirmation factuelle DOIT être suivie de [Source X]
2. Tu ne DOIS JAMAIS inventer un fait, un chiffre, un délai, une procédure ou une obligation
3. Si l'information n'est PAS dans les sources → dis explicitement "Cette information n'apparaît pas dans les sources consultées."
4. Tu ne DOIS JAMAIS inventer de numéros de source. Utilise UNIQUEMENT les [Source X] listées dans le contexte.

PÉRIMÈTRE :
- Tu es EXCLUSIVEMENT un assistant RGPD/CNIL. Tu ne réponds qu'aux questions relatives à la protection des données personnelles.
- Si la question est HORS de ce périmètre (marketing, technique pure, opinion, etc.), réponds en UNE PHRASE : "Cette question ne relève pas du périmètre RGPD/CNIL couvert par mes sources." et STOP. Ne développe pas.

STRUCTURE DE RÉPONSE :
- Commence par la réponse directe en 1-2 phrases (le principe clé)
- Ajoute les détails nécessaires : critères, conditions, étapes — issus des sources
- Si des listes ou étapes existent dans les sources, reproduis-les sous forme de listes Markdown
- NE RÉPÈTE PAS la même information avec des mots différents. Une seule formulation suffit.
- Vise la CONCISION : 50-200 mots pour une question simple, 200-400 mots max pour une question complexe.

STYLE :
- Formate ta réponse en Markdown : **gras** pour les termes clés, listes à puces, listes numérotées
- Vocabulaire juridique précis : responsable de traitement, sous-traitant, base légale, AIPD, etc.
- Distingue clairement OBLIGATION LÉGALE vs RECOMMANDATION/BONNE PRATIQUE
- Cite les articles de loi tels que mentionnés dans les sources
- Sois concret et opérationnel

INTERDICTIONS :
- Jamais de phrases vagues type "il est recommandé de se rapprocher de la CNIL" si l'info est dans les sources
- Jamais de généralités non sourcées
- Jamais d'invention de source [Source X] qui n'existe pas dans le contexte
- Jamais de paraphrase redondante : ne reformule pas ce que tu viens de dire
"""

    # ── Prompts adaptatifs selon l'intent ──

    SYSTEM_PROMPT_METHODOLOGIQUE = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu construis des **méthodologies opérationnelles complètes**.

APPROCHE :
1. Appuie-toi en PRIORITÉ sur les sources fournies pour les fondements juridiques et les obligations
2. Complète avec tes connaissances RGPD générales pour la structuration métier (étapes, acteurs, livrables)
3. DISTINGUE toujours ce qui vient des sources vs ta connaissance générale

RÈGLES :
1. Chaque obligation juridique citée DOIT être suivie de [Source X]
2. Les éléments de structuration métier (ordre des étapes, acteurs à mobiliser, livrables) peuvent venir de ta connaissance RGPD — signale-le avec [Pratique RGPD]
3. Ne JAMAIS inventer de fait juridique, chiffre, délai ou article de loi
4. Si une information juridique n'est PAS dans les sources → dis-le explicitement

STRUCTURE DE RÉPONSE OBLIGATOIRE :
1. **Principe clé** — en 1-2 phrases, le fondement juridique [Source X]
2. **Méthodologie** — étapes chronologiques numérotées avec :
   - Qui (acteur interne : DPO, RSSI, DSI, Juridique, Métier...)
   - Quoi (action concrète)
   - Livrable attendu
3. **Points de vigilance** — risques, erreurs courantes
4. **Références** — articles RGPD, guides CNIL cités

STYLE :
- Formate en Markdown : **gras**, listes numérotées, listes à puces
- Vocabulaire juridique précis
- Concret et opérationnel — pas de théorie abstraite
- 300-500 mots

INTERDICTIONS :
- Jamais de "contactez la CNIL" si l'info est disponible
- Jamais d'invention de source [Source X]
- Jamais mélanger OBLIGATION LÉGALE et BONNE PRATIQUE sans le signaler
"""

    SYSTEM_PROMPT_ORGANISATIONNEL = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu structures les **rôles, responsabilités et processus internes**.

APPROCHE :
1. Fondements juridiques depuis les sources [Source X]
2. Organisation interne depuis ta connaissance RGPD [Pratique RGPD]
3. Distingue toujours les deux

STRUCTURE DE RÉPONSE :
1. **Cadre juridique** — obligations légales avec [Source X]
2. **Acteurs et responsabilités** :
   - DPO : rôle, positionnement
   - Responsable de traitement : obligations
   - Sous-traitant : obligations contractuelles
   - Autres acteurs internes (RSSI, DSI, Juridique, Métiers)
3. **Processus recommandé** — workflow, circuits de validation
4. **Points de vigilance**

RÈGLES :
- Chaque obligation → [Source X]
- Structuration organisationnelle → [Pratique RGPD] si pas dans les sources
- Markdown, **gras**, listes
- 200-400 mots
- Jamais inventer de source, jamais de paraphrase redondante
"""

    SYSTEM_PROMPT_CAS_PRATIQUE = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu analyses des **cas pratiques** de manière structurée.

APPROCHE :
1. Identifie les enjeux juridiques du cas depuis les sources [Source X]
2. Applique les principes au cas concret
3. Donne une recommandation opérationnelle

STRUCTURE DE RÉPONSE :
1. **Analyse du cas** — enjeux identifiés, cadre applicable
2. **Règles applicables** — obligations et principes [Source X]
3. **Application au cas** — comment les règles s'appliquent concrètement
4. **Recommandation** — actions à mener, dans quel ordre

RÈGLES :
- Chaque règle citée → [Source X]
- Analyse personnelle du cas → [Pratique RGPD]
- Markdown, **gras**, listes
- 200-400 mots
- Jamais inventer de fait juridique
"""

    SYSTEM_PROMPT_COMPARAISON = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu compares des **concepts, régimes ou options** de manière structurée.

STRUCTURE DE RÉPONSE :
1. **Définition** de chaque concept/option [Source X]
2. **Tableau comparatif** ou liste structurée :
   - Critères de distinction
   - Conditions d'application
   - Avantages / limites
3. **Conclusion** — dans quel cas utiliser chaque option

RÈGLES :
- Chaque définition et critère → [Source X]
- Synthèse comparative → [Pratique RGPD] si pas explicite dans les sources
- Utilise des tableaux Markdown si pertinent
- 200-400 mots
- Jamais inventer de source
"""

    SYSTEM_PROMPT_LISTE = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu fournis des **listes exhaustives et détaillées**.

RÈGLES NON NÉGOCIABLES :
1. CHAQUE élément de la liste DOIT être suivi de [Source X]
2. Tu ne DOIS JAMAIS inventer un fait, un chiffre, un délai, une procédure ou une obligation
3. Si l'information n'est PAS dans les sources → dis explicitement "Cette information n'apparaît pas dans les sources consultées."
4. Tu ne DOIS JAMAIS inventer de numéros de source

STRUCTURE DE RÉPONSE :
1. **Introduction** — cadre de la liste (1-2 phrases) [Source X]
2. **Liste complète** — numérotée, avec détail pour chaque élément
3. **Note** — si la liste semble incomplète, le signaler

STYLE :
- Listes numérotées Markdown
- **Gras** pour chaque terme clé
- Détail suffisant pour chaque élément (pas juste le nom)
- Exhaustivité > concision pour ce type de question

INTERDICTIONS :
- Jamais de "contactez la CNIL" si l'info est dans les sources
- Jamais d'invention de source
- Jamais de liste tronquée sans le signaler
"""

    # ── Mapping intent → system prompt ──
    INTENT_PROMPTS = {
        "factuel": "SYSTEM_PROMPT",
        "methodologique": "SYSTEM_PROMPT_METHODOLOGIQUE",
        "organisationnel": "SYSTEM_PROMPT_ORGANISATIONNEL",
        "cas_pratique": "SYSTEM_PROMPT_CAS_PRATIQUE",
        "comparaison": "SYSTEM_PROMPT_COMPARAISON",
        "liste_exhaustive": "SYSTEM_PROMPT_LISTE",
    }

    # ── User prompt adaptatif ──

    USER_PROMPT_TEMPLATE = """DOCUMENTS DE RÉFÉRENCE :
{context}

SOURCES DISPONIBLES :
{sources_list}

QUESTION DU DPO :
{question}

{intent_instruction}
"""

    # Instructions spécifiques par intent (injectées dans le user prompt)
    INTENT_INSTRUCTIONS = {
        "factuel": (
            "Consigne : Réponds de façon CONCISE en utilisant EXCLUSIVEMENT les informations ci-dessus. "
            "Cite [Source X] après chaque fait. Utilise des listes Markdown et **gras** pour les termes clés. "
            "Si l'information est absente des sources, indique-le en une phrase. "
            "Ne dépasse pas 300 mots sauf si la question demande explicitement une liste exhaustive."
        ),
        "methodologique": (
            "Consigne : Construis une **méthodologie opérationnelle complète** avec étapes chronologiques, "
            "acteurs internes à mobiliser et livrables attendus. Cite [Source X] pour chaque obligation juridique. "
            "Complète avec ta connaissance RGPD pour la structuration (signale [Pratique RGPD]). "
            "{negative_instruction}"
            "Vise 300-500 mots."
        ),
        "organisationnel": (
            "Consigne : Structure ta réponse autour des **rôles, responsabilités et processus internes**. "
            "Cite [Source X] pour le cadre juridique. Complète avec [Pratique RGPD] pour l'organisation. "
            "{negative_instruction}"
            "Vise 200-400 mots."
        ),
        "cas_pratique": (
            "Consigne : Analyse ce cas pratique de manière structurée : enjeux → règles → application → recommandation. "
            "Cite [Source X] pour chaque règle. Donne une recommandation concrète. "
            "{negative_instruction}"
            "Vise 200-400 mots."
        ),
        "comparaison": (
            "Consigne : Compare de manière structurée (tableau ou liste) avec critères de distinction. "
            "Cite [Source X] pour chaque définition et critère. Conclus sur les cas d'usage. "
            "Vise 200-400 mots."
        ),
        "liste_exhaustive": (
            "Consigne : Fournis une liste EXHAUSTIVE et DÉTAILLÉE. Chaque élément doit avoir [Source X]. "
            "Numérote les éléments. Si la liste semble incomplète, signale-le. "
            "L'exhaustivité prime sur la concision."
        ),
    }

    def __init__(
        self,
        max_context_length: int = 32000,  # Chars (~8000 tokens, Nemo 128K mais on garde de la marge)
        include_metadata: bool = True,
        llm_provider = None  # Pour Map-Reduce si nécessaire
    ):
        """
        Args:
            max_context_length: Longueur max du contexte en caractères
            include_metadata: Inclure métadonnées dans le contexte
            llm_provider: Provider LLM pour résumés intelligents
        """
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self.llm_provider = llm_provider
    
    def build_context(
        self,
        documents: List,  # List[RetrievedDocument] from retriever
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        reverse_packing_override: Optional[bool] = None,
        intent: Optional[QuestionIntent] = None,
    ) -> Dict[str, str]:
        """
        Construit le contexte complet pour le LLM
        
        Stratégie adaptative :
        - Si contexte <= max_length : contexte direct
        - Si contexte > max_length : Map-Reduce
          1. Découpe chunks en batches
          2. Génère réponse partielle par batch
          3. Fusionne les réponses partielles
        
        Args:
            documents: Documents récupérés par le retriever
            question: Question de l'utilisateur
            conversation_history: Historique optionnel [{role: str, content: str}]
            reverse_packing_override: Si True, force le reverse repacking (moins pertinent en premier).
                Si False, force l'ordre naturel (plus pertinent en premier).
                Si None, utilise le reverse repacking par défaut.
            intent: QuestionIntent classifié (si None, utilise le prompt factuel par défaut)
        
        Returns:
            Dict avec 'system', 'user', 'sources_metadata'
        """
        # 1. Construction du contexte documentaire
        context_str = self._format_documents(documents, reverse_packing_override=reverse_packing_override)
        
        # 2. Gestion historique (si fourni)
        if conversation_history:
            history_str = self._format_history(conversation_history)
            context_str = f"{history_str}\n\n{context_str}"
        
        # 3. Gestion intelligente de la longueur
        if len(context_str) > self.max_context_length:
            logger.warning(
                f"⚠️  Contexte trop long ({len(context_str)} chars > {self.max_context_length})"
            )
            
            # Map-Reduce : réponses partielles puis fusion
            if self.llm_provider:
                logger.info("🔄 Map-Reduce : génération par batches puis fusion...")
                context_str = self._map_reduce_context(documents, question)
            else:
                # Fallback: troncature simple
                logger.warning("   ⚠️  Pas de LLM pour Map-Reduce, troncature simple")
                context_str = context_str[:self.max_context_length] + "\n\n[...contexte tronqué...]"
        
        # 4. Sélection du system prompt selon l'intent
        intent_type = intent.intent if intent else "factuel"
        prompt_attr = self.INTENT_PROMPTS.get(intent_type, "SYSTEM_PROMPT")
        system_prompt = getattr(self, prompt_attr, self.SYSTEM_PROMPT)
        
        # 5. Construction instruction utilisateur adaptative
        intent_instruction = self._build_intent_instruction(intent)
        
        # 6. Construction prompt utilisateur
        sources_list = []
        for i, doc in enumerate(documents, 1):
            source_url = doc.chunks[0].metadata.get('source_url', '') if doc.chunks else ''
            parent_url = doc.chunks[0].metadata.get('parent_url', '') if doc.chunks else ''
            display = source_url or parent_url or doc.document_path
            file_type = doc.chunks[0].metadata.get('file_type', '') if doc.chunks else ''
            type_label = f" ({file_type.upper()})" if file_type else ''
            # Tag origine : [CNIL] ou [Interne]
            source_origin = doc.chunks[0].metadata.get('source', 'CNIL') if doc.chunks else 'CNIL'
            origin_tag = '[Interne]' if source_origin == 'ENTREPRISE' else '[CNIL]'
            sources_list.append(f"[Source {i}] {origin_tag} {doc.primary_nature}{type_label} - {display[:100]}")
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_str,
            sources_list="\n".join(sources_list),
            question=question,
            intent_instruction=intent_instruction,
        )
        
        # 7. Extraction métadonnées sources pour post-traitement
        sources_metadata = self._extract_sources_metadata(documents)
        
        if intent and intent.intent != "factuel":
            logger.info(f"📝 System prompt adapté: {intent.intent} (structure={intent.expected_structure})")
        
        return {
            "system": system_prompt,
            "user": user_prompt,
            "sources_metadata": sources_metadata
        }
    
    def _build_intent_instruction(self, intent: Optional[QuestionIntent]) -> str:
        """Construit l'instruction utilisateur adaptée à l'intent."""
        if intent is None:
            return self.INTENT_INSTRUCTIONS["factuel"]
        
        template = self.INTENT_INSTRUCTIONS.get(intent.intent, self.INTENT_INSTRUCTIONS["factuel"])
        
        # Construire l'instruction anti-dérive (negative_topics)
        negative_instruction = ""
        if intent.negative_topics:
            topics_str = ", ".join(intent.negative_topics)
            negative_instruction = (
                f"IMPORTANT : Ne parle PAS de {topics_str} sauf si la question le demande explicitement. "
            )
        
        # Substitution sécurisée (certains templates n'ont pas {negative_instruction})
        if "{negative_instruction}" in template:
            return template.format(negative_instruction=negative_instruction)
        return template
    
    def _map_reduce_context(self, documents: List, question: str) -> str:
        """
        Map-Reduce : génère réponses partielles par batch puis fusionne
        
        Stratégie :
        1. MAP : Découpe chunks en batches de ~5000 chars
        2. Pour chaque batch : génère réponse partielle avec le LLM
        3. REDUCE : Fusionne les réponses partielles en une réponse cohérente
        
        Avantage : CONSERVE toute l'information (listes, critères, étapes)
        
        Args:
            documents: Documents récupérés
            question: Question utilisateur
        
        Returns:
            Contexte consolidé des réponses partielles
        """
        # 1. MAP : Découper en batches
        batches = []
        current_batch = []
        current_size = 0
        batch_max_size = 5000  # ~1000 tokens
        
        for doc in documents:
            for chunk in doc.chunks:
                chunk_size = len(chunk.text)
                if current_size + chunk_size > batch_max_size and current_batch:
                    # Batch plein, commencer nouveau
                    batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                
                current_batch.append({
                    "source_id": documents.index(doc) + 1,
                    "nature": doc.primary_nature,
                    "text": chunk.text,
                    "url": chunk.metadata.get('source_url', doc.document_path)
                })
                current_size += chunk_size
        
        # Dernier batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"   📦 {len(batches)} batches créés")
        
        # 2. Générer réponse partielle par batch
        partial_responses = []
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"   🤖 Génération réponse partielle {i}/{len(batches)}...")
            
            # Construire contexte du batch
            batch_context = "\n\n".join([
                f"[Source {item['source_id']}] {item['nature']}\n{item['text']}"
                for item in batch
            ])
            
            # Prompt pour réponse partielle
            partial_prompt = f"""Réponds à la question en utilisant UNIQUEMENT les informations ci-dessous.
Extrais TOUTES les informations pertinentes : listes, critères, étapes, chiffres, délais.
Ne résume PAS, donne l'information COMPLÈTE et STRUCTURÉE.

CONTEXTE :
{batch_context}

QUESTION :
{question}

Réponse (conserve listes, étapes, critères INTÉGRALEMENT) :"""
            
            try:
                response = self.llm_provider.generate(
                    partial_prompt,
                    temperature=0.0,  # Factuel pur
                    max_tokens=800  # Assez pour listes
                )
                partial_responses.append(response.strip())
            except Exception as e:
                logger.warning(f"   ⚠️  Erreur génération batch {i}: {e}")
        
        # 3. REDUCE : Fusionner les réponses partielles
        if len(partial_responses) == 1:
            result = partial_responses[0]
        else:
            logger.info(f"   🔄 Fusion de {len(partial_responses)} réponses partielles...")
            
            # Concaténer les réponses partielles
            combined = "\n\n---\n\n".join([
                f"Réponse partielle {i}:\n{resp}"
                for i, resp in enumerate(partial_responses, 1)
            ])
            
            # Prompt de fusion
            fusion_prompt = f"""Fusionne ces réponses partielles en UNE réponse cohérente et complète.
CONSERVE TOUTES les informations : listes, critères, étapes, chiffres.
Élimine les redondances mais GARDE les détails.
Structure de manière claire avec numérotation/bullets.

RÉPONSES PARTIELLES :
{combined}

QUESTION ORIGINALE :
{question}

Réponse fusionnée (complète et structurée) :"""
            
            try:
                result = self.llm_provider.generate(
                    fusion_prompt,
                    temperature=0.0,
                    max_tokens=1000
                ).strip()
            except Exception as e:
                logger.error(f"   ❌ Erreur fusion: {e}")
                # Fallback : concaténation simple
                result = "\n\n".join(partial_responses)
        
        logger.info(f"   ✅ Map-Reduce terminé: {len(result)} chars")
        return result
        """
        Résume intelligemment chaque chunk en fonction de la question
        pour tenir dans la fenêtre de contexte
        
        Stratégie :
        1. Pour chaque chunk, extraire l'essence pertinente vis-à-vis de la question
        2. Conserver métadonnées (source, nature)
        3. Viser ~300 chars par chunk au lieu de 400-450
        
        Args:
            documents: Documents récupérés
            question: Question utilisateur
        
        Returns:
            Contexte résumé
        """
        summarize_prompt = """Extrait du chunk UNIQUEMENT les informations pertinentes pour la question.
CONSERVE INTÉGRALEMENT : listes à puces, étapes numérotées, critères, chiffres, délais, procédures.
Format : phrases courtes et directes. Max 400 caractères.

Question: {question}

Chunk:
{chunk_text}

Extraction (max 400 chars, conserve listes/étapes):"""
        
        summarized_parts = []
        total_chunks = sum(len(doc.chunks) for doc in documents)
        logger.info(f"   Résumé de {total_chunks} chunks...")
        
        for i, doc in enumerate(documents, 1):
            parts = [f"[Source {i}] {doc.primary_nature}"]
            
            if self.include_metadata:
                display_source = doc.chunks[0].metadata.get('source_url', doc.document_path) if doc.chunks else doc.document_path
                parts.append(f"Source: {display_source}")
            
            parts.append("")
            
            # Résumer chaque chunk
            for j, chunk in enumerate(doc.chunks, 1):
                chunk_header = f"Chunk {j}/{len(doc.chunks)}"
                if self.include_metadata:
                    chunk_header += f" ({chunk.confidence})"
                
                parts.append(chunk_header)
                
                # Résumé LLM
                try:
                    summary_prompt_text = summarize_prompt.format(
                        question=question,
                        chunk_text=chunk.text[:1500]  # Limiter pour vitesse
                    )
                    summary = self.llm_provider.generate(
                        summary_prompt_text,
                        temperature=0.1,  # Très déterministe
                        max_tokens=150  # Plus long pour listes
                    )
                    parts.append(summary.strip())
                except Exception as e:
                    logger.warning(f"   ⚠️  Erreur résumé chunk {i}.{j}: {e}")
                    # Fallback: troncature
                    parts.append(chunk.text[:300] + "...")
                
                parts.append("")
            
            parts.append("---")
            summarized_parts.append("\n".join(parts))
        
        result = "\n\n".join(summarized_parts)
        logger.info(f"   ✅ Contexte résumé: {len(result)} chars (vs {self.max_context_length} max)")
        return result
    
    def _format_documents(self, documents: List, reverse_packing_override: Optional[bool] = None) -> str:
        """
        Formate les documents pour le contexte.
        
        Reverse repacking : les documents les MOINS pertinents sont placés
        en premier, les PLUS pertinents en dernier (proches de la question).
        Cela exploite le biais de récence des LLMs pour maximiser la précision.
        
        Args:
            documents: Documents triés par pertinence décroissante
            reverse_packing_override: True = reverse (défaut), False = ordre naturel
        
        Les numéros de source [Source N] sont attribués par rang de pertinence
        (Source 1 = le plus pertinent) mais l'ORDRE d'apparition dépend du mode.
        """
        formatted_parts = []
        
        # Déterminer l'ordre de présentation
        use_reverse = reverse_packing_override if reverse_packing_override is not None else True
        
        n_docs = len(documents)
        
        # Créer la liste (source_num, doc)
        indexed_docs = list(enumerate(documents, 1))  # [(1, doc1), (2, doc2), ...]
        if use_reverse:
            indexed_docs = list(reversed(indexed_docs))  # [(N, docN), ..., (1, doc1)]
        
        for source_num, doc in indexed_docs:
            # Tag origine : [CNIL] ou [Interne]
            source_origin = doc.chunks[0].metadata.get('source', 'CNIL') if doc.chunks else 'CNIL'
            origin_tag = '[Interne]' if source_origin == 'ENTREPRISE' else '[CNIL]'
            parts = [f"[Source {source_num}] {origin_tag} {doc.primary_nature}"]
            
            if self.include_metadata:
                # URL source : priorité source_url > parent_url > document_path
                source_url = ''
                parent_url = ''
                if doc.chunks:
                    source_url = doc.chunks[0].metadata.get('source_url', '')
                    parent_url = doc.chunks[0].metadata.get('parent_url', '')
                display_source = source_url or parent_url or doc.document_path
                parts.append(f"Source: {display_source}")
                
                # Titre/description du document si disponible
                title = doc.chunks[0].metadata.get('title', '') if doc.chunks else ''
                if title:
                    parts.append(f"Description: {title[:150]}")
                
                parts.append(f"Score pertinence: {doc.avg_similarity:.2f}")
            
            parts.append("")  # Ligne vide
            
            # Chunks du document
            for j, chunk in enumerate(doc.chunks, 1):
                chunk_header = f"Chunk {j}/{len(doc.chunks)}"
                if self.include_metadata:
                    # Localisation précise dans le document
                    location_parts = []
                    heading = chunk.metadata.get('heading', '')
                    page_info = chunk.metadata.get('page_info', '')
                    if heading:
                        location_parts.append(f"Section: {heading[:100]}")
                    if page_info:
                        location_parts.append(page_info)
                    
                    location_str = ' | '.join(location_parts) if location_parts else ''
                    if location_str:
                        chunk_header += f" [{location_str}]"
                    chunk_header += f" (Confidence: {chunk.confidence})"
                
                parts.append(chunk_header + ":")
                parts.append(chunk.text.strip())
                parts.append("")  # Ligne vide entre chunks
            
            parts.append("---")  # Séparateur entre documents
            parts.append("")
            
            formatted_parts.append("\n".join(parts))
        
        return "\n".join(formatted_parts)
    
    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formate l'historique de conversation"""
        if not history:
            return ""
        
        parts = ["HISTORIQUE DE LA CONVERSATION :"]
        for msg in history[-5:]:  # Garde seulement les 5 derniers échanges
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            parts.append(f"\n{role}: {msg['content']}")
        
        parts.append("\n---\n")
        return "\n".join(parts)
    
    def _extract_sources_metadata(self, documents: List) -> List[Dict]:
        """Extrait les métadonnées des sources pour post-traitement"""
        sources = []
        
        for i, doc in enumerate(documents, 1):
            # URLs : priorité source_url > parent_url > document_path
            source_url = doc.chunks[0].metadata.get('source_url', '') if doc.chunks else ''
            parent_url = doc.chunks[0].metadata.get('parent_url', '') if doc.chunks else ''
            display_url = source_url or parent_url or doc.document_path
            file_type = doc.chunks[0].metadata.get('file_type', '') if doc.chunks else ''
            title = doc.chunks[0].metadata.get('title', '') if doc.chunks else ''
            
            source_origin = doc.chunks[0].metadata.get('source', 'CNIL') if doc.chunks else 'CNIL'
            source_info = {
                "id": i,
                "document_path": doc.document_path,
                "source_url": display_url,  # URL CNIL ou path Entreprise
                "parent_url": parent_url,  # Page CNIL référençant ce doc
                "nature": doc.primary_nature,
                "origin": source_origin,  # CNIL ou ENTREPRISE
                "file_type": file_type,
                "title": title,
                "avg_similarity": doc.avg_similarity,
                "chunks": [
                    {
                        "index": chunk.chunk_index,
                        "nature": chunk.chunk_nature,
                        "confidence": chunk.confidence,
                        "similarity": chunk.similarity_score,
                        "heading": chunk.metadata.get('heading', ''),
                        "page_info": chunk.metadata.get('page_info', ''),
                        "text": chunk.text
                    }
                    for chunk in doc.chunks
                ]
            }
            sources.append(source_info)
        
        return sources
    
    def _shorten_path(self, path: str, max_length: int = 60) -> str:
        """Raccourcit un chemin de fichier pour lisibilité"""
        if len(path) <= max_length:
            return path
        
        # Garde le début et la fin
        half = (max_length - 3) // 2
        return f"{path[:half]}...{path[-half:]}"
    
    def format_response_with_sources(
        self,
        response: str,
        sources_metadata: List[Dict]
    ) -> Dict[str, any]:
        """
        Formate la réponse finale avec les sources.
        
        Renumérotation : si le LLM cite les sources 1, 3, 7 sur 8 fournies,
        on renuméote en 1, 2, 3 dans le texte ET dans les cartes pour cohérence.
        """
        import re
        
        # 1. Identifier les sources effectivement citées (dans l'ordre d'apparition)
        cited_ids_ordered = []
        for match in re.finditer(r'\[Source\s+(\d+)\]', response):
            src_id = int(match.group(1))
            if src_id not in cited_ids_ordered:
                cited_ids_ordered.append(src_id)
        
        # 2. Construire la table de renumérotation : ancien_id → nouveau_id
        renumber_map = {}
        for new_id, old_id in enumerate(cited_ids_ordered, 1):
            renumber_map[old_id] = new_id
        
        # 3. Renuméroter dans le texte de la réponse
        renumbered_response = response
        if renumber_map:
            # Remplacer en commençant par les plus grands IDs pour éviter les collisions
            # Ex: Source 3→2 avant Source 2→1, sinon Source 3→2→1
            # Stratégie : passer par des placeholders temporaires
            for old_id in sorted(renumber_map.keys(), reverse=True):
                renumbered_response = renumbered_response.replace(
                    f'[Source {old_id}]',
                    f'[__SRC_{renumber_map[old_id]}__]'
                )
            # Puis remplacer les placeholders par les vrais noms
            for new_id in range(1, len(renumber_map) + 1):
                renumbered_response = renumbered_response.replace(
                    f'[__SRC_{new_id}__]',
                    f'[Source {new_id}]'
                )
        
        # 4. Construire les cartes sources renumérotées
        sources_display = []
        for source in sources_metadata:
            old_id = source["id"]
            if old_id in renumber_map:
                new_id = renumber_map[old_id]
                
                locations = []
                for chunk in source.get('chunks', []):
                    loc_parts = []
                    if chunk.get('heading'):
                        loc_parts.append(chunk['heading'][:80])
                    if chunk.get('page_info'):
                        loc_parts.append(chunk['page_info'])
                    if loc_parts:
                        locations.append(' | '.join(loc_parts))
                
                sources_display.append({
                    "id": new_id,
                    "path": source["document_path"],
                    "url": source.get("source_url", source["document_path"]),
                    "parent_url": source.get("parent_url", ""),
                    "nature": source["nature"],
                    "origin": source.get("origin", "CNIL"),
                    "file_type": source.get("file_type", ""),
                    "title": source.get("title", ""),
                    "score": source["avg_similarity"],
                    "locations": locations,
                    "cited": True
                })
        
        # Trier par nouveau numéro
        sources_display.sort(key=lambda s: s['id'])
        
        # 5. Fallback : si aucune citation détectée, garder toutes les sources telles quelles
        if not sources_display:
            for source in sources_metadata:
                locations = []
                for chunk in source.get('chunks', []):
                    loc_parts = []
                    if chunk.get('heading'):
                        loc_parts.append(chunk['heading'][:80])
                    if chunk.get('page_info'):
                        loc_parts.append(chunk['page_info'])
                    if loc_parts:
                        locations.append(' | '.join(loc_parts))
                
                sources_display.append({
                    "id": source["id"],
                    "path": source["document_path"],
                    "url": source.get("source_url", source["document_path"]),
                    "parent_url": source.get("parent_url", ""),
                    "nature": source["nature"],
                    "origin": source.get("origin", "CNIL"),
                    "file_type": source.get("file_type", ""),
                    "title": source.get("title", ""),
                    "score": source["avg_similarity"],
                    "locations": locations,
                    "cited": False
                })
        
        cited_new_ids = list(range(1, len(renumber_map) + 1))
        
        return {
            "response": renumbered_response,
            "sources": sources_display,
            "cited_sources": cited_new_ids
        }
    
    def _extract_cited_sources(self, response: str, sources_metadata: List[Dict]) -> List[int]:
        """Extrait les IDs des sources citées dans la réponse"""
        cited = []
        for source in sources_metadata:
            # Cherche [Source X] dans la réponse
            if f"[Source {source['id']}]" in response:
                cited.append(source["id"])
        return cited


def create_context_builder(
    max_context_length: int = 6500,
    include_metadata: bool = True,
    llm_provider = None
) -> ContextBuilder:
    """Factory function pour créer un context builder"""
    return ContextBuilder(
        max_context_length=max_context_length,
        include_metadata=include_metadata,
        llm_provider=llm_provider
    )
