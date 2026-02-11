"""
RAG Pipeline - Orchestration complète du système RAG

Architecture :
1. Query Expansion : LLM reformule en 3 variantes (multi-query retrieval)
2. Hybrid Retrieval : BM25 (sparse) + ChromaDB (dense) + Summary pre-filter
3. RRF Fusion : Reciprocal Rank Fusion des résultats sparse + dense + multi-query
4. Cross-Encoder Reranking : Jina reranker v2 multilingue (40 → 10 chunks)
5. Dual Generation : 2 passes (ordre naturel + inversé) pour self-consistency
6. Stance Comparison : détection concordance/contradiction entre les 2 réponses
7. Synthesis : si contradiction → 3ème appel LLM pour réponse nuancée
8. Grounding Validation : vérification citations sources
"""
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from .retriever import RAGRetriever, RetrievedDocument, RetrievedChunk
from .context_builder import ContextBuilder
from .generator import Generator
from .validators import RelevanceValidator, GroundingValidator
from .reranker import CrossEncoderReranker, RankedChunk

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Réponse complète du système RAG"""
    # Réponse
    answer: str
    
    # Sources
    sources: List[Dict[str, Any]] = field(default_factory=list)
    cited_sources: List[int] = field(default_factory=list)
    
    # Métadonnées
    question: str = ""
    model: str = ""
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    # Contexte (optionnel, pour debug)
    retrieved_documents: Optional[List] = None
    context_used: Optional[str] = None
    
    # Erreur éventuelle
    error: Optional[str] = None


class RAGPipeline:
    """
    Pipeline RAG complet
    
    Flux :
    1. Query → Retriever → Documents pertinents
    2. Documents → Context Builder → Prompt formaté
    3. Prompt → Generator → Réponse LLM
    4. Post-traitement → RAGResponse finale
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        context_builder: ContextBuilder,
        generator: Generator,
        llm_provider = None,  # Pour validators
        reranker: Optional[CrossEncoderReranker] = None,
        debug_mode: bool = False,
        enable_validation: bool = True,  # Validation ON par défaut
        rerank_candidates: int = 40,     # Candidats bruts passés au reranker (semantic+BM25, pas dédupliqués)
        rerank_top_k: int = 10,          # Nombre de chunks à garder après reranking
    ):
        """
        Args:
            retriever: Retriever configuré
            context_builder: Context builder configuré
            generator: Generator configuré
            llm_provider: Provider LLM pour validators
            reranker: Cross-encoder reranker (optionnel)
            debug_mode: Si True, inclut contexte et documents dans la réponse
            enable_validation: Active validation pertinence + grounding
            rerank_candidates: Chunks à envoyer au reranker
            rerank_top_k: Chunks à garder après reranking
        """
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.reranker = reranker
        self.debug_mode = debug_mode
        self.enable_validation = enable_validation
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k
        
        # Validators
        if enable_validation and llm_provider:
            self.relevance_validator = RelevanceValidator(llm_provider, threshold=0.80)
            self.grounding_validator = GroundingValidator(llm_provider)
        else:
            self.relevance_validator = None
            self.grounding_validator = None
    
    def query(
        self,
        question: str,
        where_filter: Optional[Dict] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        n_documents: Optional[int] = None,
        n_chunks_per_doc: Optional[int] = None,
        temperature: Optional[float] = None,
        _retry_count: int = 0  # Compteur interne pour fallback
    ) -> RAGResponse:
        """
        Query principale du système RAG
        
        Args:
            question: Question de l'utilisateur
            where_filter: Filtre ChromaDB optionnel (ex: {"chunk_nature": "GUIDE"})
            conversation_history: Historique optionnel [{role: str, content: str}]
            n_documents: Override nombre de documents
            n_chunks_per_doc: Override chunks par document
            temperature: Override temperature LLM
        
        Returns:
            RAGResponse complète
        """
        import time
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RAG Query: {question[:100]}...")
        logger.info(f"{'='*80}\n")
        
        try:
            # 1. RETRIEVAL
            logger.info("📥 Phase 1 : Retrieval")
            retrieval_start = time.time()
            
            # Si reranker actif, récupérer un large pool de candidats bruts
            # SANS déduplication par document — le reranker triera
            if self.reranker is not None:
                raw_candidates = self.retriever.retrieve_candidates(
                    query=question,
                    n_candidates=self.rerank_candidates,
                    where_filter=where_filter,
                )
                
                retrieval_time = time.time() - retrieval_start
                logger.info(f"✅ Retrieval candidats: {len(raw_candidates)} chunks en {retrieval_time:.2f}s")
                
                if not raw_candidates:
                    logger.warning("⚠️  Aucun chunk candidat trouvé")
                    return RAGResponse(
                        answer="Je n'ai pas trouvé d'information pertinente dans ma base de connaissance pour répondre à cette question.",
                        question=question,
                        retrieval_time=retrieval_time,
                        total_time=time.time() - start_time,
                        error="No candidates found"
                    )
                
                # 1.5 RERANKING (cross-encoder) sur TOUS les candidats bruts
                logger.info(f"🔄 Phase 1.5 : Cross-Encoder Reranking ({len(raw_candidates)} candidats)")
                import time as _time
                rerank_start = _time.time()
                
                ranked_chunks = self.reranker.rerank(
                    query=question,
                    chunks=raw_candidates,
                    top_k=self.rerank_top_k,
                )
                
                # Reconstruire les documents à partir des chunks reranked
                # On passe une liste vide car on n'a pas de documents originaux
                documents = self._rebuild_documents_from_ranked_chunks(
                    ranked_chunks, n_chunks_per_doc
                )
                
                rerank_time = _time.time() - rerank_start
                logger.info(
                    f"✅ Reranking terminé en {rerank_time:.2f}s "
                    f"({len(raw_candidates)} → {sum(len(d.chunks) for d in documents)} chunks, "
                    f"{len(documents)} documents)"
                )
            else:
                # Sans reranker : flow classique avec dédup par document
                documents = self.retriever.retrieve(
                    query=question,
                    where_filter=where_filter,
                    n_documents=n_documents,
                    n_chunks_per_doc=n_chunks_per_doc
                )
                
                retrieval_time = time.time() - retrieval_start
                logger.info(f"✅ Retrieval terminé en {retrieval_time:.2f}s")
            
            if not documents:
                logger.warning("⚠️  Aucun document trouvé")
                return RAGResponse(
                    answer="Je n'ai pas trouvé d'information pertinente dans ma base de connaissance pour répondre à cette question.",
                    question=question,
                    retrieval_time=retrieval_time,
                    total_time=time.time() - start_time,
                    error="No documents found"
                )
            
            # Debug: affiche documents récupérés
            if self.debug_mode:
                logger.debug(self.retriever.format_results_debug(documents))
            
            # 2. VALIDATION PERTINENCE
            # Skip quand le reranker est actif : le cross-encoder EST la validation de pertinence.
            # Le seuil distance <= 0.80 est calibré pour distances cosine brutes (~0.20-0.27),
            # pas pour distances post-reranking (1 - rerank_score). Le validator rejetterait
            # les chunks que le reranker a correctement scorés bas (mais non-nuls).
            if self.enable_validation and self.relevance_validator and self.reranker is None:
                logger.info("🔍 Phase 2 : Validation pertinence des chunks...")
                
                # Extraire tous les chunks pour validation
                all_chunks = []
                for doc in documents:
                    all_chunks.extend(doc.chunks)
                
                total_chunks = len(all_chunks)
                
                # Valider
                valid_chunks = self.relevance_validator.validate_chunks(
                    query=question,
                    chunks=all_chunks,
                    conversation_history=conversation_history
                )
                
                logger.info(f"   ✅ Pertinence: {len(valid_chunks)}/{total_chunks} chunks validés")
                
                # IMPORTANT: Filtrer les chunks non pertinents des documents
                valid_chunk_ids = {chunk.chunk_id for chunk in valid_chunks}
                for doc in documents:
                    doc.chunks = [c for c in doc.chunks if c.chunk_id in valid_chunk_ids]
                
                # Si trop de chunks rejetés ET pas de reranker actif, fallback vers plus de docs
                # Quand le reranker est actif, les chunks bas-score sont légitimement non-pertinents
                # Le fallback classique (retrieve sans reranker) ne ferait qu'ajouter du bruit
                if (len(valid_chunks) < total_chunks * 0.5 
                    and len(valid_chunks) < 3 
                    and self.reranker is None):
                    logger.warning(f"⚠️  Seulement {len(valid_chunks)} chunks pertinents, récupération de documents additionnels...")
                    # Récupérer plus de docs en fallback
                    documents = self.retriever.retrieve(
                        query=question,
                        where_filter=where_filter,
                        n_documents=(n_documents or 3) + 2,  # +2 docs supplémentaires
                        n_chunks_per_doc=n_chunks_per_doc
                    )
                elif len(valid_chunks) < 3 and self.reranker is not None:
                    logger.info(f"ℹ️  {len(valid_chunks)} chunks valides après reranking — normal pour questions spécifiques")
            
            # 3. CONTEXT BUILDING + 4. GENERATION
            # Dual-generation quand le reranker est actif : on génère avec les
            # deux ordres de présentation (normal + inversé) et on compare.
            # Si les réponses se contredisent, le contexte est ambigu.
            logger.info("🏗️  Phase 3 : Context Building")
            
            if self.reranker is not None:
                # ── DUAL-GENERATION (self-consistency via context order) ──
                # Pass A : ordre naturel (Source 1 = plus pertinent en premier)
                context_a = self.context_builder.build_context(
                    documents=documents,
                    question=question,
                    conversation_history=conversation_history,
                    reverse_packing_override=False,
                )
                # Pass B : ordre inversé (Source 1 en dernier, recency bias)
                context_b = self.context_builder.build_context(
                    documents=documents,
                    question=question,
                    conversation_history=conversation_history,
                    reverse_packing_override=True,
                )
                
                logger.info(f"✅ Dual context construit (A={len(context_a['user'])} chars, B={len(context_b['user'])} chars)")
                
                # Génération des deux passes
                logger.info("🤖 Phase 4 : Dual Generation (self-consistency)")
                generation_start = time.time()
                
                generated_a = self.generator.generate(
                    system_prompt=context_a['system'],
                    user_prompt=context_a['user'],
                    temperature=temperature
                )
                generated_b = self.generator.generate(
                    system_prompt=context_b['system'],
                    user_prompt=context_b['user'],
                    temperature=temperature
                )
                
                generation_time = time.time() - generation_start
                
                if generated_a.error and generated_b.error:
                    logger.error(f"❌ Erreur double génération: A={generated_a.error}, B={generated_b.error}")
                    return RAGResponse(
                        answer="Une erreur s'est produite lors de la génération de la réponse.",
                        question=question,
                        retrieval_time=retrieval_time,
                        generation_time=generation_time,
                        total_time=time.time() - start_time,
                        error=generated_a.error
                    )
                
                # Choisir la réponse ou détecter l'incohérence
                generated, context = self._select_dual_response(
                    generated_a, generated_b,
                    context_a, context_b,
                    question, documents,
                )
                
                logger.info(f"✅ Dual generation terminée en {generation_time:.2f}s")
            else:
                # ── SINGLE GENERATION (pas de reranker) ──
                context = self.context_builder.build_context(
                    documents=documents,
                    question=question,
                    conversation_history=conversation_history
                )
                
                logger.info(f"✅ Context construit ({len(context['user'])} chars)")
                
                # Debug: affiche le prompt
                if self.debug_mode:
                    logger.debug(f"\n{'='*80}")
                    logger.debug("SYSTEM PROMPT:")
                    logger.debug(context['system'][:500] + "...")
                    logger.debug(f"\n{'='*80}")
                    logger.debug("USER PROMPT:")
                    logger.debug(context['user'][:1000] + "...")
                    logger.debug(f"{'='*80}\n")
                
                logger.info("🤖 Phase 4 : Generation")
                generation_start = time.time()
                
                generated = self.generator.generate(
                    system_prompt=context['system'],
                    user_prompt=context['user'],
                    temperature=temperature
                )
                
                generation_time = time.time() - generation_start
                
                if generated.error:
                    logger.error(f"❌ Erreur génération: {generated.error}")
                    return RAGResponse(
                        answer="Une erreur s'est produite lors de la génération de la réponse.",
                        question=question,
                        retrieval_time=retrieval_time,
                        generation_time=generation_time,
                        total_time=time.time() - start_time,
                        error=generated.error
                    )
                
                logger.info(f"✅ Génération terminée en {generation_time:.2f}s")
            
            # 4.5 VALIDATION GROUNDING
            if self.enable_validation and self.grounding_validator:
                logger.info("🔍 Validation grounding de la réponse...")
                
                available_sources = [s['id'] for s in context['sources_metadata']]
                validation = self.grounding_validator.validate_response(
                    response=generated.text,
                    available_sources=available_sources,
                    context=context['user']
                )
                
                if validation.is_valid:
                    logger.info(f"   ✅ Grounding validé (score: {validation.score:.2f})")
                else:
                    logger.error(f"   ❌ Grounding échoué: {validation.reason}")
                
                # Grounding issues : traitement gradué
                if not validation.is_valid:
                    if "hallucination" in validation.reason.lower():
                        # Compter les problèmes concrets (montants, articles, dates)
                        n_issues = validation.reason.count(" ; ") + 1
                        if n_issues >= 3:
                            # Trop de faits inventés → rejeter
                            logger.error(f"❌ HALLUCINATION SÉVÈRE ({n_issues} problèmes) - Réponse rejetée")
                            return RAGResponse(
                                answer="Je n'ai pas trouvé suffisamment d'informations fiables dans mes sources pour répondre précisément à cette question. Les documents disponibles ne couvrent peut-être pas ce sujet en détail.",
                                question=question,
                                retrieval_time=retrieval_time,
                                generation_time=generation_time,
                                total_time=time.time() - start_time,
                                error=f"Hallucination detected: {validation.reason}"
                            )
                        else:
                            # Warnings mineurs → garder la réponse mais logger
                            logger.warning(f"⚠️  Grounding partiel ({n_issues} warning(s)): {validation.reason}")
                    
                    # Phrases interdites : WARNING seulement (few-shot prompt devrait gérer)
                    if "phrases interdites" in validation.reason.lower():
                        logger.warning(f"⚠️  Phrases 'consultez' détectées (few-shot prompt pas respecté): {validation.reason}")
                    
                    # Si sources inventées, les supprimer
                    if "inventées" in validation.reason.lower():
                        logger.warning("🧹 Nettoyage des sources inventées...")
                        generated.text = self.grounding_validator.fix_invented_sources(
                            response=generated.text,
                            available_sources=available_sources
                        )
                    
                    # Si réponse évasive, warning seulement
                    if "évasive" in validation.reason.lower():
                        logger.warning("⚠️  Réponse évasive détectée - le LLM ne trouve pas la réponse dans le contexte")
            
            # 5. POST-PROCESSING
            logger.info("📝 Phase 5 : Post-processing")
            
            # Validation qualité : vérifier que réponse contient du contenu substantiel
            response_text = generated.text.strip()
            lines = [l.strip() for l in response_text.split('\n') if l.strip()]
            
            # Réponse trop courte (< 50 chars ou vide) → retry immédiat
            if len(response_text) < 50 and _retry_count == 0:
                logger.warning(f"⚠️  Réponse trop courte ({len(response_text)} chars)")
                logger.info("🔄 FALLBACK : Retry avec plus de documents...")
                return self.query(
                    question=question,
                    where_filter=where_filter,
                    conversation_history=conversation_history,
                    n_documents=(n_documents or 5) + 3,
                    n_chunks_per_doc=(n_chunks_per_doc or 3) + 2,
                    temperature=temperature,
                    _retry_count=1
                )
            
            # Compter lignes avec contenu actionnable (listes, critères, étapes concrètes)
            actionable_keywords = [':', '-', '•', '1.', '2.', '3.', 'selon', 'exemple', 'notamment', 'sont', 'doit']
            actionable_lines = sum(1 for line in lines if any(kw in line.lower() for kw in actionable_keywords))
            
            # Si <30% de contenu actionnable ET réponse longue (>3 lignes) ET pas encore retry → FALLBACK
            # Note: réponses courtes (1-3 lignes) sont souvent concises et correctes, pas insuffisantes
            if len(lines) > 3 and actionable_lines / len(lines) < 0.3 and _retry_count == 0:
                logger.warning(f"⚠️  Réponse insuffisante ({actionable_lines}/{len(lines)} lignes actionnables)")
                logger.info("🔄 FALLBACK : Récupération de plus de documents...")
                
                # Retry avec plus de docs et chunks
                return self.query(
                    question=question,
                    where_filter=where_filter,
                    conversation_history=conversation_history,
                    n_documents=(n_documents or 5) + 3,  # +3 docs
                    n_chunks_per_doc=(n_chunks_per_doc or 3) + 2,  # +2 chunks
                    temperature=temperature,
                    _retry_count=1  # Marquer comme retry
                )
            elif len(lines) > 3 and actionable_lines / len(lines) < 0.3 and _retry_count > 0:
                logger.error("❌ Réponse insuffisante même après fallback")
                # Garder la réponse mais ajouter disclaimer
                generated.text += f"\n\n⚠️ Les sources disponibles ne contiennent pas suffisamment de détails. Pour plus d'informations, consultez directement [Source 1]."
            
            # Post-processing Markdown : améliorer le formatage des listes
            generated.text = self._fix_markdown_formatting(generated.text)
            
            formatted = self.context_builder.format_response_with_sources(
                response=generated.text,
                sources_metadata=context['sources_metadata']
            )
            
            # Construction réponse finale
            total_time = time.time() - start_time
            
            response = RAGResponse(
                answer=formatted['response'],
                sources=formatted['sources'],
                cited_sources=formatted['cited_sources'],
                question=question,
                model=generated.model,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time
            )
            
            # Debug info
            if self.debug_mode:
                response.retrieved_documents = documents
                response.context_used = context['user']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Pipeline terminé en {total_time:.2f}s")
            logger.info(f"   - Retrieval: {retrieval_time:.2f}s")
            logger.info(f"   - Generation: {generation_time:.2f}s")
            logger.info(f"   - Documents: {len(documents)}")
            logger.info(f"   - Sources citées: {len(response.cited_sources)}/{len(response.sources)}")
            logger.info(f"{'='*80}\n")
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Erreur pipeline: {e}", exc_info=True)
            return RAGResponse(
                answer="Une erreur inattendue s'est produite lors du traitement de votre question.",
                question=question,
                total_time=time.time() - start_time,
                error=str(e)
            )
    
    @staticmethod
    def _fix_markdown_formatting(text: str) -> str:
        """Post-processing : convertit les listes inline en listes Markdown à puces.
        
        Détecte les patterns courants où le LLM produit des listes séparées par
        des ';' ou des numéros indentés au lieu de listes Markdown propres.
        """
        import re
        
        lines = text.split('\n')
        result = []
        
        for line in lines:
            stripped = line.strip()
            
            # Pattern 1: lignes avec items séparés par " ; " qui ressemblent à une liste
            # Ex: "critère A ; critère B ; critère C [Source 2]."
            # On détecte si c'est une vraie liste (3+ items séparés par ;)
            if ' ; ' in stripped:
                parts = stripped.split(' ; ')
                if len(parts) >= 3:
                    # Extraire le préfixe (texte avant la liste)
                    # Ex: "Les neuf critères sont : critère A ; critère B..."
                    # Chercher " : " ou ": " comme séparateur intro/liste
                    intro_match = re.match(r'^(.+?(?:sont|suivants?|comprennent|incluent|suivantes?)\s*:\s*)(.*)', stripped, re.IGNORECASE)
                    if intro_match:
                        intro = intro_match.group(1).strip()
                        list_text = intro_match.group(2).strip()
                        parts = list_text.split(' ; ')
                    else:
                        intro = None
                    
                    if len(parts) >= 3:
                        # C'est une liste inline → convertir en puces
                        if intro:
                            result.append(intro)
                            result.append('')  # ligne vide avant la liste
                        for part in parts:
                            part = part.strip().rstrip('.').rstrip(',')
                            if part:
                                # Majuscule au début
                                part = part[0].upper() + part[1:] if part else part
                                result.append(f'- {part}')
                        # Récupérer la source à la fin si présente
                        last_part = parts[-1].strip()
                        source_match = re.search(r'(\[Source\s*\d+\]\.?)$', last_part)
                        if source_match:
                            # La source est déjà sur le dernier item, c'est bon
                            pass
                        result.append('')  # ligne vide après la liste
                        continue
            
            # Pattern 2: lignes indentées avec numéro mais sans format Markdown
            # Ex: "    1. Décrire le traitement" ou "    Décrire le traitement"
            indent_match = re.match(r'^(\s{4,})(\d+[\.\)]\s*)?(.+)$', line)
            if indent_match and not stripped.startswith('-') and not stripped.startswith('*'):
                num = indent_match.group(2)
                content = indent_match.group(3).strip()
                if num:
                    result.append(f'{num.strip()} {content}')
                else:
                    result.append(f'- {content}')
                continue
            
            result.append(line)
        
        return '\n'.join(result)
    
    def _select_dual_response(
        self,
        generated_a,  # GeneratedResponse (ordre naturel)
        generated_b,  # GeneratedResponse (ordre inversé)
        context_a: Dict,
        context_b: Dict,
        question: str,
        documents: List,
    ):
        """
        Compare deux réponses générées avec des ordres de contexte différents.
        
        Self-consistency via context order :
        - Si les deux réponses concordent (même position oui/non, mêmes faits clés)
          → prendre la réponse A (ordre naturel, plus fiable avec reranker)
        - Si elles se contredisent (une dit oui, l'autre non) → le contexte contient
          des sources contradictoires pour des cas différents → synthèse nuancée
        
        Returns:
            (generated, context) — la réponse choisie et son contexte associé
        """
        import re
        
        # Si une seule réponse est valide, la prendre
        if generated_a.error:
            logger.warning("⚠️  Dual-gen: pass A en erreur, utilisation pass B")
            return generated_b, context_b
        if generated_b.error:
            logger.warning("⚠️  Dual-gen: pass B en erreur, utilisation pass A")
            return generated_a, context_a
        
        text_a = generated_a.text.strip()
        text_b = generated_b.text.strip()
        
        # Détecter la position (oui/non) de chaque réponse
        stance_a = self._detect_stance(text_a)
        stance_b = self._detect_stance(text_b)
        
        logger.info(f"🔄 Dual-gen: stance A={stance_a}, stance B={stance_b}")
        logger.info(f"   A ({len(text_a)} chars): {text_a[:120]}...")
        logger.info(f"   B ({len(text_b)} chars): {text_b[:120]}...")
        
        # Cas 1 : les deux réponses concordent → prendre A (ordre naturel)
        # v11b a montré que le reverse packing (pass B) donne 91% vs 93% avec pass A
        # Le recency bias amplifie les chunks toxiques quand ils sont bien classés
        if stance_a == stance_b:
            logger.info(f"✅ Dual-gen: réponses concordantes ({stance_a}), utilisation pass A (ordre naturel)")
            return generated_a, context_a
        
        # Cas 2 : les deux sont neutres/incertaines → prendre A (ordre naturel)
        if stance_a == "neutral" and stance_b == "neutral":
            logger.info(f"✅ Dual-gen: deux réponses neutres, utilisation pass A (ordre naturel)")
            return generated_a, context_a
        
        # Cas 3 : CONTRADICTION (une dit oui, l'autre dit non)
        # → Les sources contiennent des positions différentes pour des cas différents
        # → Générer une réponse de synthèse
        logger.warning(f"⚠️  Dual-gen: CONTRADICTION détectée (A={stance_a}, B={stance_b})")
        logger.warning(f"   → Synthèse des deux réponses")
        
        # Construire un prompt de synthèse qui force la nuance
        synthesis_prompt = f"""Voici deux analyses de la même question, basées sur les mêmes sources RGPD/CNIL mais présentées dans un ordre différent.

ANALYSE 1 (position: {stance_a}) :
{text_a}

ANALYSE 2 (position: {stance_b}) :
{text_b}

QUESTION ORIGINALE : {question}

Les deux analyses utilisent les mêmes sources. Si elles arrivent à des conclusions différentes, c'est que les sources décrivent des CAS DIFFÉRENTS (ex: secteur public vs privé, type d'organisme, finalité du traitement).

Rédige UNE réponse de synthèse qui :
1. Répond "Oui, sous conditions" ou "Cela dépend du cas" si applicable
2. Distingue clairement les cas où c'est possible et ceux où ça ne l'est pas
3. Cite les [Source X] des deux analyses
4. Donne les conditions, critères ou limites mentionnés dans les analyses

Réponse de synthèse :"""

        try:
            synthesis = self.generator.generate(
                system_prompt=context_a['system'],
                user_prompt=synthesis_prompt,
                temperature=0.0
            )
            if not synthesis.error:
                logger.info(f"✅ Dual-gen: synthèse générée ({len(synthesis.text)} chars)")
                # Utiliser context_a pour les métadonnées sources (identiques)
                return synthesis, context_a
            else:
                logger.warning(f"⚠️  Dual-gen: erreur synthèse, fallback sur pass A")
                return generated_a, context_a
        except Exception as e:
            logger.error(f"❌ Dual-gen: erreur synthèse: {e}, fallback sur pass A")
            return generated_a, context_a
    
    @staticmethod
    def _detect_stance(text: str) -> str:
        """
        Détecte la position globale d'une réponse : 'positive', 'negative', ou 'neutral'.
        
        Analyse les premiers ~200 caractères (la réponse directe) pour détecter
        si le LLM dit oui, non, ou nuance.
        """
        # Prendre le début de la réponse (la position est toujours au début)
        start = text[:300].lower()
        
        # Patterns négatifs forts
        neg_patterns = [
            r'\b(?:non|no)\b[,.]',
            r"n'est pas (?:possible|autorisé|permis|mobilisable|valable|applicable)",
            r"ne peut pas être (?:utilisé|invoqué|mobilisé|retenu)",
            r"ne (?:peut|doit|saurait) pas",
            r"n'est pas une base légale",
            r"impossible",
            r"interdit",
        ]
        
        # Patterns positifs forts
        pos_patterns = [
            r'\b(?:oui|yes)\b[,.]',
            r"(?:peut|peuvent) être (?:utilisé|invoqué|mobilisé|retenu)",
            r"est possible",
            r"est autorisé",
            r"sous (?:certaines |)conditions",
            r"sous réserve",
            r"peut (?:constituer|servir|fonder)",
        ]
        
        import re
        neg_score = sum(1 for p in neg_patterns if re.search(p, start))
        pos_score = sum(1 for p in pos_patterns if re.search(p, start))
        
        if neg_score > pos_score and neg_score >= 1:
            return "negative"
        elif pos_score > neg_score and pos_score >= 1:
            return "positive"
        else:
            return "neutral"
    
    def _rebuild_documents_from_ranked_chunks(
        self,
        ranked_chunks: List[RankedChunk],
        n_chunks_per_doc: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Reconstruit les RetrievedDocument à partir de chunks reranked SANS documents originaux.
        
        Utilisé par le nouveau flow où le retriever retourne des chunks bruts (pas
        groupés par document). Le reranker trie tous les candidats, puis on regroupe
        les top chunks par document_path pour construire les documents.
        
        IMPORTANT: Pas de limite n_chunks_per_doc ici — le reranker a DÉJÀ sélectionné
        les top_k meilleurs chunks. Tous doivent être passés au context builder.
        La limitation par document ferait perdre des chunks pertinents que le cross-encoder
        a explicitement choisi de garder.
        
        Args:
            ranked_chunks: Chunks triés par le cross-encoder
            n_chunks_per_doc: IGNORÉ dans ce flow (kept for API compat)
            
        Returns:
            Liste de RetrievedDocument ordonnés par meilleur score de reranking
        """
        from collections import defaultdict
        
        # NOTE: On ne limite PAS par n_chunks_per_doc dans le flow reranker.
        # Le reranker a déjà sélectionné les meilleurs chunks tous documents confondus.
        # Limiter à 3/doc jetterait des chunks bien classés.
        
        # Regrouper les ranked chunks par document
        doc_chunks = defaultdict(list)
        doc_best_score = {}
        for rc in ranked_chunks:
            doc_chunks[rc.document_path].append(rc)
            if rc.document_path not in doc_best_score or rc.rerank_score > doc_best_score[rc.document_path]:
                doc_best_score[rc.document_path] = rc.rerank_score
        
        # Construire les documents, triés par meilleur score de chunk
        sorted_doc_paths = sorted(doc_best_score.keys(), key=lambda p: doc_best_score[p], reverse=True)
        
        new_documents = []
        for doc_path in sorted_doc_paths:
            chunks_ranked = doc_chunks[doc_path]
            chunks_ranked.sort(key=lambda x: x.rerank_score, reverse=True)
            # PAS de limite n_chunks_per_doc : tous les chunks reranked sont gardés
            
            # Reconvertir RankedChunk → RetrievedChunk
            converted_chunks = []
            for rc in chunks_ranked:
                converted_chunks.append(RetrievedChunk(
                    chunk_id=rc.chunk_id,
                    text=rc.text,
                    document_path=rc.document_path,
                    chunk_nature=rc.metadata.get('chunk_nature', 'UNKNOWN'),
                    chunk_index=rc.metadata.get('chunk_index', 0),
                    confidence=rc.metadata.get('confidence', 'medium'),
                    distance=1.0 - rc.rerank_score,  # Convertir score → distance
                    metadata=rc.metadata,
                    hybrid_score=rc.rerank_score,
                ))
            
            # Déduire la nature du document depuis les chunks
            natures = [rc.metadata.get('chunk_nature', 'UNKNOWN') for rc in chunks_ranked]
            primary_nature = max(set(natures), key=natures.count) if natures else 'UNKNOWN'
            
            new_documents.append(RetrievedDocument(
                document_path=doc_path,
                chunks=converted_chunks,
                avg_similarity=doc_best_score[doc_path],
                primary_nature=primary_nature,
            ))
        
        logger.info(f"📦 Reranking → {len(new_documents)} documents reconstruits")
        return new_documents
    
    def _rebuild_documents_from_ranked(
        self,
        ranked_chunks: List[RankedChunk],
        original_documents: List[RetrievedDocument],
        n_chunks_per_doc: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Reconstruit les RetrievedDocument à partir des chunks reranked.
        
        Après reranking, on regroupe les chunks par document_path
        et on reconstruit des RetrievedDocument ordonnés par le meilleur
        score de reranking de leurs chunks.
        """
        from collections import defaultdict
        
        n_chunks = n_chunks_per_doc or self.retriever.n_chunks_per_doc
        
        # Index des documents originaux pour récupérer les métadonnées
        original_doc_map = {doc.document_path: doc for doc in original_documents}
        
        # Regrouper les ranked chunks par document
        doc_chunks = defaultdict(list)
        doc_best_score = {}
        for rc in ranked_chunks:
            doc_chunks[rc.document_path].append(rc)
            if rc.document_path not in doc_best_score or rc.rerank_score > doc_best_score[rc.document_path]:
                doc_best_score[rc.document_path] = rc.rerank_score
        
        # Construire les documents, triés par meilleur score de chunk
        sorted_doc_paths = sorted(doc_best_score.keys(), key=lambda p: doc_best_score[p], reverse=True)
        
        new_documents = []
        for doc_path in sorted_doc_paths:
            chunks_ranked = doc_chunks[doc_path]
            # Trier les chunks de ce document par score de reranking
            chunks_ranked.sort(key=lambda x: x.rerank_score, reverse=True)
            # Limiter au nombre de chunks par document
            chunks_ranked = chunks_ranked[:n_chunks]
            
            # Reconvertir RankedChunk → RetrievedChunk
            original_doc = original_doc_map.get(doc_path)
            converted_chunks = []
            for rc in chunks_ranked:
                converted_chunks.append(RetrievedChunk(
                    chunk_id=rc.chunk_id,
                    text=rc.text,
                    document_path=rc.document_path,
                    chunk_nature=rc.metadata.get('chunk_nature', 'UNKNOWN'),
                    chunk_index=rc.metadata.get('chunk_index', 0),
                    confidence=rc.metadata.get('confidence', 'medium'),
                    distance=1.0 - rc.rerank_score,  # Convertir score → distance
                    metadata=rc.metadata,
                    hybrid_score=rc.rerank_score,
                ))
            
            primary_nature = original_doc.primary_nature if original_doc else 'UNKNOWN'
            
            new_documents.append(RetrievedDocument(
                document_path=doc_path,
                chunks=converted_chunks,
                avg_similarity=doc_best_score[doc_path],
                primary_nature=primary_nature,
            ))
        
        logger.info(f"📦 Reranking → {len(new_documents)} documents reconstruits")
        return new_documents
    
    def format_response(self, response: RAGResponse, show_sources: bool = True) -> str:
        """
        Formate la réponse pour affichage
        
        Args:
            response: RAGResponse à formatter
            show_sources: Inclure les sources
        
        Returns:
            Texte formaté pour CLI
        """
        lines = []
        
        # Réponse
        lines.append("=" * 80)
        lines.append("RÉPONSE")
        lines.append("=" * 80)
        lines.append(response.answer)
        lines.append("")
        
        # Sources
        if show_sources and response.sources:
            lines.append("=" * 80)
            lines.append(f"SOURCES ({len(response.sources)} documents)")
            lines.append("=" * 80)
            
            for source in response.sources:
                cited = "✓" if source['cited'] else " "
                lines.append(f"\n[{cited}] Source {source['id']} - {source['nature']}")
                # Afficher URL (CNIL) ou path (Entreprise)
                url = source.get('url', source['path'])
                lines.append(f"    URL: {url}")
                # Localisation précise
                locations = source.get('locations', [])
                if locations:
                    lines.append(f"    📍 {' | '.join(locations[:3])}")
                lines.append(f"    Score: {source['score']:.3f}")
            
            lines.append("")
        
        # Métadonnées
        lines.append("=" * 80)
        lines.append("MÉTADONNÉES")
        lines.append("=" * 80)
        lines.append(f"Modèle: {response.model}")
        lines.append(f"Temps total: {response.total_time:.2f}s")
        lines.append(f"  - Retrieval: {response.retrieval_time:.2f}s")
        lines.append(f"  - Generation: {response.generation_time:.2f}s")
        lines.append(f"Sources citées: {len(response.cited_sources)}/{len(response.sources)}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def create_pipeline(
    collection,
    llm_provider,
    n_documents: int = 5,
    n_chunks_per_doc: int = 3,
    max_context_length: int = 32000,  # ~8000 tokens, Nemo 128K
    model: str = "mistral-nemo",
    temperature: float = 0.0,  # Factuel strict par défaut
    max_tokens: int = 2000,  # Assez long pour listes complètes
    debug_mode: bool = False,
    enable_validation: bool = True,  # Validation ON par défaut
    enable_hybrid: bool = True,      # Recherche hybride BM25+semantic
    enable_reranker: bool = True,    # Cross-encoder reranking
    enable_summary_prefilter: bool = True,  # Pré-filtre par summaries
    enable_query_expansion: bool = True,    # LLM query expansion (multi-query)
    summaries_path: Optional[str] = None,
    rerank_candidates: int = 40,
    rerank_top_k: int = 10,
) -> RAGPipeline:
    """
    Factory function pour créer un pipeline RAG complet avec toutes les optimisations.
    
    Architecture:
    1. Query → Acronym Expansion → LLM Query Expansion (multi-query)
    2. Summary Pre-Filter (BM25 sur summaries)
    3. Multi-Query Hybrid Retrieval (BM25 sparse + ChromaDB dense × N queries + RRF)
    4. Cross-Encoder Reranking (bge-reranker-v2-m3, multilingue)
    5. Relevance Validation (LLM-based)
    6. Context Building (reverse repacking)
    7. Generation (Mistral-Nemo)
    8. Grounding Validation → Post-process
    
    Args:
        collection: Collection ChromaDB
        llm_provider: Provider LLM
        n_documents: Nombre de documents à récupérer
        n_chunks_per_doc: Chunks par document
        max_context_length: Longueur max du contexte (chars)
        model: Modèle LLM
        temperature: Temperature génération
        debug_mode: Mode debug
        enable_validation: Active validation pertinence + grounding
        enable_hybrid: Active BM25+semantic hybrid search
        enable_reranker: Active cross-encoder reranking
        enable_summary_prefilter: Active pré-filtre par summaries
        enable_query_expansion: Active LLM query expansion (multi-query retrieval)
        summaries_path: Chemin vers document_summaries.json (auto-détecté si None)
        rerank_candidates: Nombre de chunks à passer au reranker
        rerank_top_k: Nombre de chunks à garder après reranking
    
    Returns:
        RAGPipeline configuré et prêt à l'emploi
    """
    import time
    from pathlib import Path
    from .retriever import create_retriever
    from .context_builder import create_context_builder
    from .generator import create_generator
    from .bm25_index import SummaryBM25Index, ChunkBM25Index
    from .query_expander import QueryExpander
    
    init_start = time.time()
    
    # ── Query Expander (multi-query) ──
    query_expander = None
    if enable_query_expansion:
        logger.info("🔄 Activation Query Expansion (multi-query retrieval)")
        query_expander = QueryExpander(
            llm_provider=llm_provider,
            enabled=True,
            n_expansions=3,
            temperature=0.7,
            max_tokens=300,
        )
    
    # ── BM25 Indexes ──
    summary_bm25 = None
    chunk_bm25 = None
    
    if enable_hybrid or enable_summary_prefilter:
        # Auto-detect summaries path
        if summaries_path is None:
            default_path = Path(__file__).parent.parent.parent / "data" / "keep" / "cnil" / "document_summaries.json"
            if default_path.exists():
                summaries_path = str(default_path)
        
        # Summary BM25 Index (pré-filtre)
        if enable_summary_prefilter and summaries_path:
            logger.info("📋 Construction index BM25 sur les summaries...")
            summary_bm25 = SummaryBM25Index()
            summary_bm25.build(summaries_path)
        
        # Chunk BM25 Index (sparse retrieval)
        if enable_hybrid:
            logger.info("📦 Construction index BM25 sur les chunks...")
            chunk_bm25 = ChunkBM25Index()
            chunk_bm25.build_from_collection(collection)
    
    # ── Retriever ──
    retriever = create_retriever(
        collection=collection,
        llm_provider=llm_provider,
        summary_bm25_index=summary_bm25,
        chunk_bm25_index=chunk_bm25,
        query_expander=query_expander,
        n_documents=n_documents,
        n_chunks_per_doc=n_chunks_per_doc,
        summary_prefilter_k=40,
        enable_hybrid=enable_hybrid,
        enable_summary_prefilter=enable_summary_prefilter,
    )
    
    # ── Reranker ──
    reranker = None
    if enable_reranker:
        logger.info("🔄 Initialisation Cross-Encoder Reranker...")
        reranker = CrossEncoderReranker(
            device="cpu",  # Pas de VRAM consommée
            batch_size=32,
            trust_remote_code=True,  # Requis pour Jina
        )
        # Le modèle sera chargé en lazy au premier appel
    
    # ── Context Builder ──
    # Désactiver reverse packing quand le reranker est actif :
    # le reranker produit un ordre optimal par cross-encoder, mais le reverse
    # packing peut placer un faux positif contextuel (bien scoré mais hors sujet
    # précis) en dernière position → maximum de recency bias → réponse biaisée.
    use_reverse_packing = not enable_reranker
    context_builder = create_context_builder(
        max_context_length=max_context_length,
        include_metadata=True,
        llm_provider=llm_provider,  # Pour résumés intelligents
        reverse_packing=use_reverse_packing,
    )
    
    # ── Generator ──
    generator = create_generator(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
    )
    
    init_time = time.time() - init_start
    logger.info(
        f"✅ Pipeline initialisé en {init_time:.1f}s "
        f"(hybrid={'ON' if enable_hybrid else 'OFF'}, "
        f"reranker={'ON' if enable_reranker else 'OFF'}, "
        f"summary_filter={'ON' if enable_summary_prefilter else 'OFF'}, "
        f"query_expansion={'ON' if enable_query_expansion else 'OFF'})"
    )
    
    return RAGPipeline(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        llm_provider=llm_provider,  # Pour validators
        reranker=reranker,
        debug_mode=debug_mode,
        enable_validation=enable_validation,
        rerank_candidates=rerank_candidates,
        rerank_top_k=rerank_top_k,
    )
