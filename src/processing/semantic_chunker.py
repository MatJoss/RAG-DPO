"""
Chunking Sémantique
Découpage intelligent basé sur la structure des documents
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunker sémantique basé sur la structure des documents"""
    
    # Tailles cibles
    MIN_CHUNK_SIZE = 200   # mots
    TARGET_CHUNK_SIZE = 500  # mots
    MAX_CHUNK_SIZE = 1000  # mots
    
    def chunk_html(self, file_path: Path) -> List[Dict]:
        """Chunke un HTML sur les titres h2/h3"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            # Supprimer bruit
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Extraire titre principal
            title = ''
            title_tag = soup.find(['h1', 'title'])
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Découper sur h2
            chunks = []
            main = soup.find(['main', 'article', 'div']) or soup.body or soup
            
            current_chunk = {
                'heading': title,
                'content': [],
                'level': 1
            }
            
            for element in main.descendants:
                if element.name in ['h2', 'h3', 'h4']:
                    # Sauvegarder chunk précédent si non vide
                    if current_chunk['content']:
                        chunk_text = ' '.join(current_chunk['content'])
                        if len(chunk_text.split()) >= self.MIN_CHUNK_SIZE:
                            chunks.append({
                                'text': chunk_text,
                                'heading': current_chunk['heading'],
                                'type': 'section'
                            })
                    
                    # Nouveau chunk
                    current_chunk = {
                        'heading': element.get_text(strip=True),
                        'content': [],
                        'level': int(element.name[1])
                    }
                
                elif element.name in ['p', 'li', 'td']:
                    text = element.get_text(strip=True)
                    if text:
                        current_chunk['content'].append(text)
            
            # Dernier chunk
            if current_chunk['content']:
                chunk_text = ' '.join(current_chunk['content'])
                if len(chunk_text.split()) >= self.MIN_CHUNK_SIZE:
                    chunks.append({
                        'text': chunk_text,
                        'heading': current_chunk['heading'],
                        'type': 'section'
                    })
            
            # Si pas de chunks (page simple), prendre tout
            if not chunks:
                full_text = main.get_text(separator=' ', strip=True)
                chunks = self._split_by_size(full_text, title)
            
            logger.debug(f"  HTML chunked: {len(chunks)} chunks from {file_path.name}")
            return chunks
        
        except Exception as e:
            logger.error(f"  Erreur chunking HTML {file_path.name}: {e}")
            return []
    
    def chunk_pdf(self, file_path: Path) -> List[Dict]:
        """Chunke un PDF sur TOC + structure"""
        
        try:
            import fitz
            doc = fitz.open(file_path)
            
            # Essayer d'extraire TOC
            toc = doc.get_toc()
            
            if toc and len(toc) > 2:
                # Chunker sur TOC
                chunks = self._chunk_pdf_with_toc(doc, toc)
            else:
                # Chunker sur pages + font changes
                chunks = self._chunk_pdf_structural(doc)
            
            doc.close()
            
            logger.debug(f"  PDF chunked: {len(chunks)} chunks from {file_path.name}")
            return chunks
        
        except Exception as e:
            logger.error(f"  Erreur chunking PDF {file_path.name}: {e}")
            return []
    
    def _chunk_pdf_with_toc(self, doc, toc: List) -> List[Dict]:
        """Chunke PDF selon la table des matières"""
        
        chunks = []
        
        for i, (level, title, page_num) in enumerate(toc):
            # Ignorer niveaux trop profonds
            if level > 3:
                continue
            
            # Déterminer page de fin
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = len(doc) - 1
            
            # Extraire texte de la section
            section_text = []
            for page_idx in range(page_num - 1, min(end_page, len(doc))):
                if page_idx >= 0 and page_idx < len(doc):
                    page = doc[page_idx]
                    section_text.append(page.get_text())
            
            text = ' '.join(section_text).strip()
            
            if text and len(text.split()) >= self.MIN_CHUNK_SIZE:
                chunks.append({
                    'text': text,
                    'heading': title,
                    'type': 'toc_section',
                    'pages': f"{page_num}-{end_page+1}"
                })
        
        return chunks
    
    def _chunk_pdf_structural(self, doc) -> List[Dict]:
        """Chunke PDF sur changements de structure (pages)"""
        
        chunks = []
        current_chunk = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            
            if not page_text.strip():
                continue
            
            current_chunk.append(page_text)
            
            # Créer chunk tous les 3-5 pages ou si trop gros
            chunk_text = ' '.join(current_chunk)
            word_count = len(chunk_text.split())
            
            if word_count >= self.TARGET_CHUNK_SIZE or (page_num + 1) % 4 == 0:
                if word_count >= self.MIN_CHUNK_SIZE:
                    chunks.append({
                        'text': chunk_text.strip(),
                        'heading': f"Pages {page_num - len(current_chunk) + 2}-{page_num + 1}",
                        'type': 'page_group'
                    })
                    current_chunk = []
        
        # Dernier chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= self.MIN_CHUNK_SIZE:
                chunks.append({
                    'text': chunk_text.strip(),
                    'heading': f"Pages {len(doc) - len(current_chunk) + 1}-{len(doc)}",
                    'type': 'page_group'
                })
        
        return chunks
    
    def chunk_spreadsheet(self, file_path: Path) -> List[Dict]:
        """Chunke un tableur par feuille"""
        
        ext = file_path.suffix.lower()
        chunks = []
        
        try:
            if ext == '.xlsx':
                import openpyxl
                wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    rows = []
                    
                    for row in sheet.iter_rows(values_only=True):
                        row_text = ' '.join([str(cell) for cell in row if cell])
                        if row_text.strip():
                            rows.append(row_text)
                    
                    if rows:
                        chunk_text = '\n'.join(rows)
                        if len(chunk_text.split()) >= self.MIN_CHUNK_SIZE:
                            chunks.append({
                                'text': chunk_text,
                                'heading': f"Feuille: {sheet_name}",
                                'type': 'spreadsheet'
                            })
                
                wb.close()
            
            elif ext == '.ods':
                from odf import table as odf_table, text as odf_text
                from odf.opendocument import load
                
                doc = load(str(file_path))
                sheets = doc.spreadsheet.getElementsByType(odf_table.Table)
                
                for sheet in sheets:
                    sheet_name = sheet.getAttribute('name') or 'Sheet'
                    rows = []
                    
                    for row in sheet.getElementsByType(odf_table.TableRow):
                        cells = row.getElementsByType(odf_table.TableCell)
                        row_text = []
                        for cell in cells:
                            paras = cell.getElementsByType(odf_text.P)
                            cell_text = ' '.join([str(p) for p in paras])
                            if cell_text.strip():
                                row_text.append(cell_text.strip())
                        
                        if row_text:
                            rows.append(' '.join(row_text))
                    
                    if rows:
                        chunk_text = '\n'.join(rows)
                        if len(chunk_text.split()) >= self.MIN_CHUNK_SIZE:
                            chunks.append({
                                'text': chunk_text,
                                'heading': f"Feuille: {sheet_name}",
                                'type': 'spreadsheet'
                            })
            
            logger.debug(f"  Spreadsheet chunked: {len(chunks)} chunks from {file_path.name}")
            return chunks
        
        except Exception as e:
            logger.error(f"  Erreur chunking spreadsheet {file_path.name}: {e}")
            return []
    
    def _split_by_size(self, text: str, heading: str = "") -> List[Dict]:
        """Split texte par taille cible"""
        
        words = text.split()
        chunks = []
        
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            
            if len(current_chunk) >= self.TARGET_CHUNK_SIZE:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'heading': heading,
                    'type': 'size_split'
                })
                current_chunk = []
        
        # Dernier chunk
        if current_chunk and len(current_chunk) >= self.MIN_CHUNK_SIZE:
            chunks.append({
                'text': ' '.join(current_chunk),
                'heading': heading,
                'type': 'size_split'
            })
        
        return chunks
    
    def merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Fusionne les chunks trop petits"""
        
        if not chunks:
            return chunks
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            word_count = len(chunk['text'].split())
            
            if word_count < self.MIN_CHUNK_SIZE:
                # Trop petit, buffer
                if buffer is None:
                    buffer = chunk
                else:
                    # Fusionner avec buffer
                    buffer['text'] = buffer['text'] + '\n\n' + chunk['text']
            else:
                # Assez grand
                if buffer:
                    # Vider buffer d'abord
                    merged.append(buffer)
                    buffer = None
                merged.append(chunk)
        
        # Dernier buffer
        if buffer:
            if merged:
                # Fusionner avec le dernier
                merged[-1]['text'] = merged[-1]['text'] + '\n\n' + buffer['text']
            else:
                merged.append(buffer)
        
        return merged
    
    def split_large_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Split les chunks trop grands"""
        
        result = []
        
        for chunk in chunks:
            word_count = len(chunk['text'].split())
            
            if word_count > self.MAX_CHUNK_SIZE:
                # Trop grand, split
                sub_chunks = self._split_by_size(chunk['text'], chunk['heading'])
                result.extend(sub_chunks)
            else:
                result.append(chunk)
        
        return result


if __name__ == "__main__":
    # Tests
    chunker = SemanticChunker()
    
    print("Chunker prêt !")
    print(f"Tailles : MIN={chunker.MIN_CHUNK_SIZE}, TARGET={chunker.TARGET_CHUNK_SIZE}, MAX={chunker.MAX_CHUNK_SIZE}")
