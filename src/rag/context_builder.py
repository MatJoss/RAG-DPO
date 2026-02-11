"""
Context Builder - Construction du contexte pour le LLM
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime

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
    # v2 — Audit 2026-02-10 : concision, anti-forçage grounding, anti-sur-justification
    SYSTEM_PROMPT = """Tu es un assistant expert RGPD spécialisé dans l'accompagnement des DPO (Délégués à la Protection des Données). Tu réponds UNIQUEMENT à partir des sources fournies dans le contexte.

RÈGLES NON NÉGOCIABLES :
1. CHAQUE affirmation factuelle DOIT être suivie de [Source X]
2. Tu ne DOIS JAMAIS inventer un fait, un chiffre, un délai, une procédure ou une obligation
3. Si l'information n'est PAS dans les sources → dis explicitement "Cette information n'apparaît pas dans les sources consultées."
4. Tu ne DOIS JAMAIS inventer de numéros de source. Utilise UNIQUEMENT les [Source X] listées dans le contexte.
5. Si AUCUNE source ne répond à la question, dis-le directement. Ne cite PAS une source non pertinente juste pour avoir une référence.

CONCISION (CRUCIAL) :
- Réponds d'abord en 2-4 phrases avec l'essentiel : la règle, le principe, la réponse directe
- N'ajoute des détails (listes, étapes, critères) que si la question le demande explicitement ou si c'est nécessaire pour être opérationnel
- Ne reformule PAS la même idée avec des mots différents
- Ne répète PAS une information déjà donnée, même depuis une source différente
- UNE citation [Source X] par fait suffit. Ne pas empiler 3 sources pour le même fait

STRUCTURE DE RÉPONSE :
- Commence par la réponse directe à la question (oui/non si applicable)
- Donne le cadre juridique (article RGPD, recommandation CNIL) si mentionné dans les sources
- Si la question demande des critères, étapes ou listes : reproduis-les depuis les sources
- Ne termine par les lacunes des sources QUE si c'est réellement utile au DPO

NUANCE (IMPORTANT) :
- Lis TOUTES les sources avant de répondre. Ne te base PAS uniquement sur la première source.
- Si une source dit "X n'est pas possible" dans un cas précis et qu'une autre dit "X est possible sous conditions", la réponse correcte est "X est possible sous certaines conditions" avec le détail.
- Attention aux sources qui décrivent un cas particulier (ex: communes, secteur public) : ne généralise PAS leur conclusion à tous les cas.
- Quand les sources présentent des conditions ou restrictions, détaille-les au lieu de répondre "non" catégoriquement.

STYLE :
- Markdown : **gras** pour les termes clés, listes à puces pour les énumérations
- Vocabulaire juridique précis
- Distingue OBLIGATION LÉGALE vs RECOMMANDATION/BONNE PRATIQUE
- Cite les articles de loi tels que mentionnés dans les sources
- Sois concret et opérationnel : chiffres, délais, exemples issus des sources

INTERDICTIONS :
- Jamais de généralités non sourcées
- Jamais d'invention de source [Source X] qui n'existe pas dans le contexte
- Jamais de conclusion qui reformule toute la réponse
- Jamais de phrase "En conclusion" ou "En résumé" qui répète ce qui a déjà été dit
- Jamais de citation d'une source dont le contenu n'a AUCUN rapport avec la question
"""

    USER_PROMPT_TEMPLATE = """DOCUMENTS DE RÉFÉRENCE :
{context}

SOURCES DISPONIBLES :
{sources_list}

QUESTION DU DPO :
{question}

Consigne : Réponds de manière concise et directe en utilisant EXCLUSIVEMENT les sources ci-dessus.
- Lis TOUTES les sources avant de répondre. Si certaines sources se contredisent, c'est qu'elles parlent de cas différents : explique les conditions.
- Commence par la réponse en 2-4 phrases maximum
- Cite [Source X] après chaque fait (une seule source par fait suffit)
- Ajoute des détails (listes, étapes) uniquement si la question le demande
- Si les sources ne contiennent PAS l'information demandée, dis-le clairement sans forcer une citation non pertinente
- Utilise **gras** pour les termes clés
"""

    def __init__(
        self,
        max_context_length: int = 32000,  # Chars (~8000 tokens, Nemo 128K mais on garde de la marge)
        include_metadata: bool = True,
        llm_provider = None,  # Pour Map-Reduce si nécessaire
        reverse_packing: bool = True,  # Ordre inversé (Lost in the Middle)
    ):
        """
        Args:
            max_context_length: Longueur max du contexte en caractères
            include_metadata: Inclure métadonnées dans le contexte
            llm_provider: Provider LLM pour résumés intelligents
            reverse_packing: Si True, place les docs les plus pertinents en dernier
                (exploite le recency bias). Si False, ordre naturel de pertinence.
                Désactiver quand le reranker est actif — le reranker produit un ordre
                optimal mais le reverse packing peut placer un faux positif en position
                dominante (dernière = plus proche de la question).
        """
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self.llm_provider = llm_provider
        self.reverse_packing = reverse_packing
    
    def build_context(
        self,
        documents: List,  # List[RetrievedDocument] from retriever
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        reverse_packing_override: Optional[bool] = None,
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
            reverse_packing_override: Si fourni, force l'ordre (True=inversé, False=naturel)
                                     Sinon utilise self.reverse_packing
        
        Returns:
            Dict avec 'system', 'user', 'sources_metadata'
        """
        # 1. Construction du contexte documentaire
        use_reverse = reverse_packing_override if reverse_packing_override is not None else self.reverse_packing
        context_str = self._format_documents(documents, reverse_packing=use_reverse)
        
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
                try:
                    context_str = self._map_reduce_context(documents, question)
                except Exception as e:
                    logger.error(f"❌ Map-Reduce échoué: {e}. Fallback troncature.")
                    context_str = context_str[:self.max_context_length] + "\n\n[...contexte tronqué...]"
            else:
                # Fallback: troncature simple
                logger.warning("   ⚠️  Pas de LLM pour Map-Reduce, troncature simple")
                context_str = context_str[:self.max_context_length] + "\n\n[...contexte tronqué...]"
        
        # 4. Construction prompt utilisateur
        sources_list = []
        for i, doc in enumerate(documents, 1):
            source_url = doc.chunks[0].metadata.get('source_url', '') if doc.chunks else ''
            parent_url = doc.chunks[0].metadata.get('parent_url', '') if doc.chunks else ''
            display = source_url or parent_url or doc.document_path
            file_type = doc.chunks[0].metadata.get('file_type', '') if doc.chunks else ''
            type_label = f" ({file_type.upper()})" if file_type else ''
            sources_list.append(f"[Source {i}] {doc.primary_nature}{type_label} - {display[:100]}")
        
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=context_str,
            sources_list="\n".join(sources_list),
            question=question
        )
        
        # 5. Extraction métadonnées sources pour post-traitement
        sources_metadata = self._extract_sources_metadata(documents)
        
        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_prompt,
            "sources_metadata": sources_metadata
        }
    
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
    
    def _format_documents(self, documents: List, reverse_packing: Optional[bool] = None) -> str:
        """
        Formate les documents pour le contexte.
        
        Deux modes d'ordre :
        - reverse_packing=True : Lost in the Middle — docs les moins pertinents
          en premier, les plus pertinents en dernier (recency bias). Utile sans reranker.
        - reverse_packing=False : Ordre naturel de pertinence (Source 1 en premier).
          Recommandé avec reranker — évite qu'un faux positif contextuel #1
          domine en position finale.
        
        Les numéros de source [Source N] sont attribués par rang de pertinence
        (Source 1 = le plus pertinent). Les scores de pertinence ne sont PAS
        affichés au LLM pour éviter le biais numérique.
        
        Args:
            documents: Documents à formater
            reverse_packing: Override l'ordre. Si None, utilise self.reverse_packing.
        """
        use_reverse = reverse_packing if reverse_packing is not None else self.reverse_packing
        formatted_parts = []
        
        # Créer la liste (source_num, doc)
        indexed_docs = list(enumerate(documents, 1))  # [(1, doc1), (2, doc2), ...]
        
        if use_reverse:
            # Reverse repacking : les docs les MOINS pertinents en premier,
            # les PLUS pertinents en dernier (proches de la question).
            # Exploite le biais de récence des LLMs.
            ordered_docs = list(reversed(indexed_docs))
        else:
            # Ordre naturel : Source 1 (plus pertinent) en premier.
            # Plus sûr quand le reranker est actif — évite qu'un faux positif
            # contextuel en position #1 domine via le recency bias.
            ordered_docs = indexed_docs
        
        for source_num, doc in ordered_docs:
            parts = [f"[Source {source_num}] {doc.primary_nature}"]
            
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
                
                # NOTE: Score de pertinence retiré du contexte LLM.
                # Le LLM ne doit pas être biaisé par les scores numériques
                # — il doit juger sur le contenu des sources.
            
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
            
            source_info = {
                "id": i,
                "document_path": doc.document_path,
                "source_url": display_url,  # URL CNIL ou path Entreprise
                "parent_url": parent_url,  # Page CNIL référençant ce doc
                "nature": doc.primary_nature,
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
    llm_provider = None,
    reverse_packing: bool = True,
) -> ContextBuilder:
    """Factory function pour créer un context builder"""
    return ContextBuilder(
        max_context_length=max_context_length,
        include_metadata=include_metadata,
        llm_provider=llm_provider,
        reverse_packing=reverse_packing,
    )
