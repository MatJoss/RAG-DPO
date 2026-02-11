"""
Classification Hybride Optimisée : Keywords + LLM
Compatible avec cnil_scraper_final.py
"""

import json
from pathlib import Path
import re
import logging
import signal
import sys
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

# Ajouter le chemin utils pour llm_provider
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from llm_provider import LLMFactory, RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Réduire verbosité de httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


class HybridClassifier:
    """Classification hybride keywords + LLM optimisée"""
    
    # Patterns d'exclusion ÉVIDENTS (pas besoin de LLM)
    OBVIOUS_EXCLUDE_PATTERNS = [
        r'/emploi',
        r'/recrutement',
        r'/presse',
        r'/communique',
        r'/contact',
        r'/mentions-legales',
        r'/plan-du-site',
        r'/accessibilite',
        r'/newsletter-inscription',
    ]
    
    # Mots-clés d'exclusion forte
    STRONG_EXCLUDE_KEYWORDS = [
        'exercer vos droits',
        'porter plainte en ligne',
        'vos droits en tant que citoyen',
        'plainte particulier',
    ]
    
    # Mots-clés par catégorie avec poids (pour pré-filtrage)
    KEYWORDS = {
        'dpo_core': {
            'poids': 3,
            'mots': [
                'dpo', 'délégué protection données', 'délégué à la protection',
                'responsable traitement', 'sous-traitant', 'aipd', 'pia',
                'registre traitement', 'analyse impact', 'violation données',
            ]
        },
        'rgpd_pro': {
            'poids': 2,
            'mots': [
                'rgpd', 'gdpr', 'protection données', 'données personnelles',
                'traitement données', 'base légale', 'consentement',
                'entreprise', 'organisme', 'professionnel',
            ]
        },
        'particulier': {
            'poids': -2,
            'mots': [
                'citoyen', 'consommateur', 'usager', 'internaute',
                'particulier', 'plainte en ligne',
            ]
        },
        'institutionnel': {
            'poids': -1,
            'mots': [
                'qui sommes-nous', 'recrutement', 'offre emploi',
                'communiqué presse', 'contact cnil',
            ]
        }
    }
    
    # Prompt LLM optimisé
    SYSTEM_PROMPT = """Tu es un DPO senior (15 ans d'expérience) évaluant des documents pour constituer ta base de connaissances professionnelle.

Ta mission : déterminer si ce document t'est UTILE dans l'exercice QUOTIDIEN de tes fonctions de DPO.

Tu cherches des documents qui t'aident concrètement à :
1. PILOTER la conformité (registre, AIPD/PIA, bases légales, durées de conservation)
2. GÉRER les incidents (violations de données, notifications, procédures de crise)
3. CONSEILLER l'organisation (avis sur traitements, privacy by design, sous-traitance)
4. FORMER et sensibiliser (supports pédagogiques, bonnes pratiques)
5. RÉPONDRE aux contrôles (préparation, documentation, jurisprudence CNIL)
6. APPLIQUER les référentiels sectoriels (santé, RH, marketing, vidéosurveillance)

Sont PERTINENTS (score >= 6) :
- Guides pratiques, méthodologies, checklists, modèles de documents
- Délibérations CNIL (sanctions, mises en demeure, avertissements)
- Lignes directrices, recommandations, référentiels
- FAQ et analyses juridiques sur le RGPD, la loi Informatique et Libertés
- Fiches thématiques (cookies, vidéosurveillance, données RH, sous-traitance...)
- Modèles de registre, clauses contractuelles, mentions d'information
- Tout document technique avec impact conformité (sécurité, pseudonymisation...)

Ne sont PAS pertinents (score < 4) :
- Pages destinées AUX PARTICULIERS (exercice des droits, plainte en ligne)
- Communication institutionnelle (presse, recrutement, organigramme CNIL)
- Pages de navigation, index, listes de liens sans contenu propre
- Actualités purement événementielles sans valeur opérationnelle durable
- Contenus sans rapport avec la protection des données

Règle d'or : EN CAS DE DOUTE, GARDE LE DOCUMENT (score 5-6). Il vaut mieux un document en trop qu'un document utile manquant.

Réponds UNIQUEMENT au format JSON suivant :
{
  "pertinent": true/false,
  "score": 0-10,
  "categorie": "essential" | "relevant" | "useful" | "neutral" | "irrelevant",
  "raison": "explication courte (max 100 mots)",
  "tags": ["tag1", "tag2", "tag3"]
}

Catégories :
- essential (8-10) : indispensable pour un DPO
- relevant (6-7.9) : clairement pertinent
- useful (4-5.9) : potentiellement utile
- neutral (2-3.9) : information générale sans valeur opérationnelle
- irrelevant (0-1.9) : hors sujet pour un DPO
"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.html_dir = self.project_root / 'data' / 'raw' / 'cnil' / 'html'
        self.metadata_dir = self.project_root / 'data' / 'metadata'
        self.cache_file = self.project_root / 'data' / 'raw' / 'cnil' / 'llm_classification_cache.json'
        self.results_file = self.project_root / 'data' / 'raw' / 'cnil' / 'hybrid_classification.json'
        
        # Charger cache LLM
        self.llm_cache = self._load_cache()
        
        # Charger résultats existants pour resume
        self._existing_results = self._load_existing_results()
        
        # Flag d'interruption gracieuse (Ctrl+C)
        self._interrupted = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        
        # Initialiser LLM
        try:
            config = RAGConfig()
            self.llm = config.llm_provider
            self.mode = config.mode
            logger.info(f"🤖 LLM initialisé en mode : {self.mode}")
        except Exception as e:
            logger.error(f"❌ Erreur init LLM : {e}")
            raise
        
        # Stats
        self.stats = {
            'total': 0,
            'obvious_exclude': 0,
            'keyword_exclude': 0,
            'llm_needed': 0,
            'llm_kept': 0,
            'llm_cached': 0,
            'resumed_skip': 0,
        }
    
    def _load_cache(self) -> Dict:
        """Charge le cache LLM"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📦 Cache chargé : {len(cache)} classifications")
                return cache
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache LLM"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde cache : {e}")
    
    def _load_existing_results(self) -> Dict:
        """Charge les résultats existants pour reprendre après interruption."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Construire un set de tous les hashes déjà traités
                already_done = set()
                
                # Docs classifiés par LLM
                for h in data.get('llm_classified', {}).keys():
                    already_done.add(h)
                
                # Docs exclus (obvious)
                for item in data.get('excluded_obvious', []):
                    already_done.add(item.get('hash', ''))
                
                # Docs exclus (keywords)
                for item in data.get('excluded_keywords', []):
                    already_done.add(item.get('hash', ''))
                
                already_done.discard('')
                
                if already_done:
                    logger.info(f"♻️  Résultats existants chargés : {len(already_done)} docs déjà traités")
                
                return data
            except Exception as e:
                logger.warning(f"⚠️  Impossible de charger les résultats existants : {e}")
                return {}
        return {}
    
    def _get_already_done_hashes(self) -> set:
        """Retourne le set des hashes déjà traités."""
        done = set()
        for h in self._existing_results.get('llm_classified', {}).keys():
            done.add(h)
        for item in self._existing_results.get('excluded_obvious', []):
            done.add(item.get('hash', ''))
        for item in self._existing_results.get('excluded_keywords', []):
            done.add(item.get('hash', ''))
        done.discard('')
        return done
    
    def _handle_interrupt(self, signum, frame):
        """Gestionnaire Ctrl+C : flag interruption pour sauvegarde propre."""
        if self._interrupted:
            # Deuxième Ctrl+C : quitter immédiatement
            print("\n\n⚠️  Deuxième Ctrl+C — arrêt immédiat !")
            sys.exit(1)
        
        self._interrupted = True
        print("\n\n🛑 Ctrl+C détecté — arrêt gracieux en cours...")
        print("   (sauvegarde des résultats en cours, patientez...)")
        print("   (Ctrl+C à nouveau pour arrêt immédiat)")
    
    def _extract_clean_text(self, html_file: Path, max_length: int = 4000) -> str:
        """Extrait le texte propre d'un HTML en ciblant le contenu principal.
        
        Stratégie : 
        1. Cherche le bloc region-content (structure CNIL)
        2. Supprime les éléments de navigation internes
        3. Fallback sur le body entier si region-content absent
        """
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            # Stratégie 1 : Cibler region-content (100% des pages CNIL)
            content_block = (
                soup.find(class_='region-content')
                or soup.find('main')
                or soup.find('article')
                or soup.find(class_='field-name-body')
            )
            
            if content_block:
                # Supprimer les blocs de navigation internes
                for tag in content_block(['script', 'style', 'nav', 'aside', 
                                          'iframe', 'noscript', 'svg']):
                    tag.decompose()
                
                # Supprimer menus, breadcrumbs, pagination
                for nav_block in content_block.find_all(class_=lambda c: c and any(
                    x in str(c).lower() for x in [
                        'menu-push', 'breadcrumb', 'pager', 'pagination',
                        'nav-', 'share-', 'social', 'cookie', 'back-to-top'
                    ]
                )):
                    nav_block.decompose()
                
                text = content_block.get_text(separator=' ', strip=True)
            else:
                # Fallback : full page nettoyée
                for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                                'aside', 'iframe', 'noscript', 'svg']):
                    tag.decompose()
                text = soup.get_text(separator=' ', strip=True)
            
            # Nettoyage whitespace
            text = ' '.join(text.split())
            
            # Tronquer si trop long
            if len(text) > max_length:
                half = max_length // 2
                text = text[:half] + "\n[...]\n" + text[-half:]
            
            return text
        except Exception as e:
            logger.warning(f"⚠️  Erreur extraction texte : {e}")
            return ""
    
    def is_obvious_exclude(self, url: str, text: str) -> Tuple[bool, str]:
        """Vérifie si le document est évidemment hors-sujet"""
        
        # 1. Patterns URL évidents
        for pattern in self.OBVIOUS_EXCLUDE_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True, f"URL pattern: {pattern}"
        
        # 2. Mots-clés d'exclusion forte
        text_lower = text.lower()
        for keyword in self.STRONG_EXCLUDE_KEYWORDS:
            if keyword in text_lower and text_lower.count(keyword) >= 2:
                return True, f"Strong keyword: {keyword}"
        
        return False, ""
    
    def calculate_keyword_score(self, text: str, url: str) -> float:
        """Calcule un score rapide basé sur keywords"""
        text_lower = text.lower()
        
        # Score texte
        text_score = 0
        for category, config in self.KEYWORDS.items():
            count = sum(text_lower.count(mot) for mot in config['mots'])
            text_score += count * config['poids']
        
        # Score URL
        url_score = 0
        if '/guide' in url or '/modele' in url or '/outil' in url:
            url_score += 2
        if '/professionnel' in url:
            url_score += 1
        if '/particulier' in url:
            url_score -= 3
        
        return (text_score * 0.7) + (url_score * 0.3)
    
    def classify_with_llm(self, text: str, url: str) -> Dict:
        """Classifie un document avec le LLM"""
        
        # Vérifier cache
        cache_key = f"{url}_{len(text)}"
        if cache_key in self.llm_cache:
            result = self.llm_cache[cache_key].copy()
            result['cached'] = True
            self.stats['llm_cached'] += 1
            logger.debug(f"💾 Cache hit pour {url[:60]}...")
            return result
        
        # Construire prompt
        user_prompt = f"""URL : {url}

Extrait du document :
{text}

Évalue la pertinence de ce document pour un DPO."""
        
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
        
        try:
            logger.debug(f"🤖 Appel LLM pour {url[:60]}...")
            
            # Appeler LLM
            response = self.llm.generate(
                full_prompt,
                temperature=0.1,
                max_tokens=500
            )
            
            logger.debug(f"📥 Réponse brute (100 premiers chars) : {response[:100]}")
            
            # Parser JSON
            response_clean = response.strip()
            
            # Nettoyer markdown si présent
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]
            
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            
            # Essayer de trouver le JSON s'il y a du texte avant/après
            if not response_clean.startswith('{'):
                # Chercher le premier {
                start_idx = response_clean.find('{')
                if start_idx != -1:
                    response_clean = response_clean[start_idx:]
                    logger.debug(f"✂️  JSON extrait après nettoyage")
            
            if not response_clean.endswith('}'):
                # Chercher le dernier }
                end_idx = response_clean.rfind('}')
                if end_idx != -1:
                    response_clean = response_clean[:end_idx+1]
            
            logger.debug(f"🧹 JSON nettoyé (100 premiers chars) : {response_clean[:100]}")
            
            # Parser
            result = json.loads(response_clean)
            
            # Valider structure
            required_fields = ['pertinent', 'score', 'categorie', 'raison', 'tags']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Champ manquant dans réponse LLM : {field}")
            
            result['cached'] = False
            
            # Log succès
            logger.info(f"✅ {result['categorie']:12s} ({result['score']:4.1f}/10) - {url[:50]}...")
            
            # Mettre en cache
            self.llm_cache[cache_key] = result
            
            # Rate limiting
            time.sleep(0.5 if self.mode == 'local' else 1.0)
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON invalide pour {url[:60]}...")
            logger.error(f"   Erreur : {e}")
            logger.error(f"   Réponse brute : {response[:200] if 'response' in locals() else 'N/A'}")
            logger.error(f"   Après nettoyage : {response_clean[:200] if 'response_clean' in locals() else 'N/A'}")
            
            # Fallback : garder par défaut
            return {
                "pertinent": True,
                "score": 5.0,
                "categorie": "useful",
                "raison": f"Erreur parsing JSON: {str(e)}",
                "tags": [],
                "cached": False,
                "error": f"JSONDecodeError: {str(e)}"
            }
        
        except Exception as e:
            logger.error(f"❌ Erreur LLM pour {url[:60]}...")
            logger.error(f"   Erreur : {type(e).__name__}: {str(e)}")
            
            # Fallback : garder par défaut
            return {
                "pertinent": True,
                "score": 5.0,
                "categorie": "useful",
                "raison": f"Erreur classification: {str(e)}",
                "tags": [],
                "cached": False,
                "error": f"{type(e).__name__}: {str(e)}"
            }
    
    def run(self, max_docs: Optional[int] = None, fresh: bool = False):
        """Exécute la classification hybride complète.
        
        Args:
            max_docs: Limiter à N documents (mode test)
            fresh: Si True, ignore les résultats existants et recommence à zéro
        """
        
        print("=" * 70)
        print("⚡🧠 CLASSIFICATION HYBRIDE : Keywords + LLM")
        print("=" * 70)
        
        # Installer gestionnaire Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Résultats existants pour resume
        already_done = set() if fresh else self._get_already_done_hashes()
        if already_done and not fresh:
            print(f"\n♻️  MODE RESUME : {len(already_done)} documents déjà traités (seront skippés)")
            print(f"   (utiliser --fresh pour recommencer à zéro)")
        
        # Récupérer listes existantes pour merge incrémental
        llm_results = dict(self._existing_results.get('llm_classified', {})) if not fresh else {}
        excluded_obvious = list(self._existing_results.get('excluded_obvious', [])) if not fresh else []
        excluded_keywords = list(self._existing_results.get('excluded_keywords', [])) if not fresh else []
        
        # Lister les HTML
        html_files = list(self.html_dir.glob('*.html'))
        
        if max_docs:
            html_files = html_files[:max_docs]
            print(f"\n🧪 MODE TEST : {max_docs} documents")
        
        self.stats['total'] = len(html_files)
        print(f"\n📄 {len(html_files)} documents à analyser\n")
        
        # PHASE 1 : Pré-filtrage Keywords
        print("⚡ Phase 1 : Pré-filtrage par keywords...")
        
        to_llm_classify = []
        
        for html_file in tqdm(html_files, desc="Pré-filtrage"):
            if self._interrupted:
                break
            
            try:
                # Skip si déjà traité (resume)
                if html_file.stem in already_done:
                    self.stats['resumed_skip'] += 1
                    continue
                
                # Charger métadonnées
                metadata_file = self.metadata_dir / f"{html_file.stem}.json"
                if not metadata_file.exists():
                    continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    url = metadata.get('url', '')
                
                # Extraire texte
                text = self._extract_clean_text(html_file, max_length=2000)
                
                # Vérifier exclusions évidentes
                is_obvious, reason = self.is_obvious_exclude(url, text)
                if is_obvious:
                    excluded_obvious.append({
                        'hash': html_file.stem,
                        'url': url,
                        'reason': reason
                    })
                    self.stats['obvious_exclude'] += 1
                    continue
                
                # Score keywords rapide
                keyword_score = self.calculate_keyword_score(text, url)
                
                # Si score très négatif, exclure
                if keyword_score < -3:
                    excluded_keywords.append({
                        'hash': html_file.stem,
                        'url': url,
                        'score': keyword_score
                    })
                    self.stats['keyword_exclude'] += 1
                    continue
                
                # Sinon, passer au LLM
                to_llm_classify.append({
                    'hash': html_file.stem,
                    'url': url,
                    'file': html_file,
                    'keyword_score': keyword_score
                })
            
            except Exception as e:
                logger.warning(f"⚠️  Erreur {html_file.name}: {e}")
                continue
        
        self.stats['llm_needed'] = len(to_llm_classify)
        
        # Résumé Phase 1
        total_for_pct = max(1, self.stats['total'])
        print(f"\n📊 Résultats Phase 1 :")
        if self.stats['resumed_skip'] > 0:
            print(f"   Déjà traités (skip) : {self.stats['resumed_skip']:5d} ({self.stats['resumed_skip']/total_for_pct*100:5.1f}%)")
        print(f"   Exclus (URL évidente)   : {self.stats['obvious_exclude']:5d} ({self.stats['obvious_exclude']/total_for_pct*100:5.1f}%)")
        print(f"   Exclus (keywords < -3)  : {self.stats['keyword_exclude']:5d} ({self.stats['keyword_exclude']/total_for_pct*100:5.1f}%)")
        print(f"   À classifier par LLM    : {self.stats['llm_needed']:5d} ({self.stats['llm_needed']/total_for_pct*100:5.1f}%)")
        
        new_excluded = self.stats['obvious_exclude'] + self.stats['keyword_exclude']
        if new_excluded > 0:
            gain_pct = new_excluded / total_for_pct * 100
            print(f"   Gain de temps           : {gain_pct:.1f}%")
        
        # Sauvegarde intermédiaire après Phase 1 (les exclusions sont déjà décidées)
        self._save_results(llm_results, excluded_obvious, excluded_keywords)
        
        # PHASE 2 : Classification LLM
        if self._interrupted:
            print("\n🛑 Interrompu après Phase 1. Résultats partiels sauvegardés.")
            self._save_cache()
            signal.signal(signal.SIGINT, self._original_sigint)
            return
        
        if len(to_llm_classify) == 0:
            print("\n✅ Tous les documents traités (exclus ou déjà classifiés) !")
            self._save_results(llm_results, excluded_obvious, excluded_keywords)
            signal.signal(signal.SIGINT, self._original_sigint)
            return
        
        estimated_min = len(to_llm_classify) * 3 / 60
        print(f"\n🧠 Phase 2 : Classification LLM de {len(to_llm_classify)} documents...")
        print(f"   Durée estimée : ~{estimated_min:.0f} minutes ({estimated_min/60:.1f}h)")
        print(f"   💡 Ctrl+C pour interrompre proprement (reprise possible)\n")
        
        save_counter = 0
        phase2_start = time.time()
        
        for item in tqdm(to_llm_classify, desc="Classification LLM"):
            if self._interrupted:
                break
            
            try:
                # Extraire texte complet
                text = self._extract_clean_text(item['file'])
                
                # Classifier
                classification = self.classify_with_llm(text, item['url'])
                
                llm_results[item['hash']] = {
                    'url': item['url'],
                    'keyword_score': item['keyword_score'],
                    'pertinent': classification['pertinent'],
                    'score': classification['score'],
                    'categorie': classification['categorie'],
                    'raison': classification['raison'],
                    'tags': classification.get('tags', []),
                    'cached': classification.get('cached', False),
                }
                
                if classification['pertinent']:
                    self.stats['llm_kept'] += 1
                
                # Sauvegarde régulière (cache + résultats)
                save_counter += 1
                if save_counter % 10 == 0:
                    self._save_cache()
                    self._save_results(llm_results, excluded_obvious, excluded_keywords)
                    
                    # Afficher progression temps
                    elapsed = time.time() - phase2_start
                    done_count = save_counter
                    remaining = len(to_llm_classify) - done_count
                    if done_count > 0:
                        eta_sec = (elapsed / done_count) * remaining
                        logger.info(f"💾 Checkpoint ({done_count}/{len(to_llm_classify)}) — ETA: {eta_sec/60:.0f}min")
            
            except Exception as e:
                logger.warning(f"⚠️  Erreur LLM {item['url']}: {e}")
                # En cas d'erreur, garder par défaut
                llm_results[item['hash']] = {
                    'url': item['url'],
                    'pertinent': True,
                    'score': 5.0,
                    'categorie': 'useful',
                    'raison': f'Erreur: {str(e)}',
                    'tags': [],
                    'error': str(e)
                }
                self.stats['llm_kept'] += 1
        
        # Sauvegarde finale (ou post-interruption)
        self._save_cache()
        self._save_results(llm_results, excluded_obvious, excluded_keywords)
        
        # Restaurer handler Ctrl+C
        signal.signal(signal.SIGINT, self._original_sigint)
        
        if self._interrupted:
            print(f"\n🛑 Interrompu après {save_counter} docs LLM. Résultats sauvegardés.")
            print(f"   Relancez la commande pour reprendre automatiquement.")
        else:
            # Résumé final
            self._print_final_summary()
    
    def _save_results(self, llm_results: Dict, excluded_obvious: List, excluded_keywords: List):
        """Sauvegarde les résultats complets"""
        results = {
            'llm_classified': llm_results,
            'excluded_obvious': excluded_obvious,
            'excluded_keywords': excluded_keywords,
            'stats': self.stats,
            'metadata': {
                'mode': self.mode,
                'total_documents': self.stats['total'],
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Résultats sauvegardés : {self.results_file}")
    
    def _print_final_summary(self):
        """Affiche le résumé final"""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ FINAL - CLASSIFICATION HYBRIDE")
        print("=" * 70)
        
        total = self.stats['total']
        
        print(f"\n📄 Documents analysés : {total}")
        
        if self.stats['resumed_skip'] > 0:
            print(f"\n♻️  Resume :")
            print(f"   Déjà traités (skip) : {self.stats['resumed_skip']:5d} ({self.stats['resumed_skip']/total*100:5.1f}%)")
        
        print(f"\n⚡ Phase 1 - Pré-filtrage :")
        print(f"   Exclus (URL)      : {self.stats['obvious_exclude']:5d} ({self.stats['obvious_exclude']/total*100:5.1f}%)")
        print(f"   Exclus (keywords) : {self.stats['keyword_exclude']:5d} ({self.stats['keyword_exclude']/total*100:5.1f}%)")
        
        print(f"\n🧠 Phase 2 - LLM :")
        print(f"   Analysés par LLM  : {self.stats['llm_needed']:5d} ({self.stats['llm_needed']/total*100:5.1f}%)")
        print(f"   Gardés par LLM    : {self.stats['llm_kept']:5d} ({self.stats['llm_kept']/self.stats['llm_needed']*100:5.1f}% des analysés)")
        print(f"   Cache utilisé     : {self.stats['llm_cached']:5d} ({self.stats['llm_cached']/self.stats['llm_needed']*100:5.1f}%)")
        
        print(f"\n✅ Résultat final :")
        kept = self.stats['llm_kept']
        excluded = total - kept
        print(f"   Documents gardés  : {kept:5d} ({kept/total*100:5.1f}%)")
        print(f"   Documents exclus  : {excluded:5d} ({excluded/total*100:5.1f}%)")
        
        # Optimisation
        time_saved_pct = (self.stats['obvious_exclude'] + self.stats['keyword_exclude']) / total * 100
        time_saved_min = (self.stats['obvious_exclude'] + self.stats['keyword_exclude']) * 3 / 60
        
        print(f"\n⏱️  Optimisation :")
        print(f"   Gain de temps LLM : ~{time_saved_pct:.0f}%")
        print(f"   Durée économisée  : ~{time_saved_min:.0f} minutes ({time_saved_min/60:.1f}h)")
        
        print("\n" + "=" * 70)
        print(f"💾 Résultats : {self.results_file}")
        print(f"💾 Cache LLM : {self.cache_file}")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification hybride Keywords + LLM')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--test', type=int, help='Tester sur N documents')
    parser.add_argument('--fresh', action='store_true', help='Ignorer les résultats existants, recommencer à zéro')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbose (debug)')
    
    args = parser.parse_args()
    
    # Activer debug si demandé
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        logger.info("🔍 Mode verbose activé")
    
    classifier = HybridClassifier(args.project_root)
    classifier.run(max_docs=args.test, fresh=args.fresh)


if __name__ == "__main__":
    main()
