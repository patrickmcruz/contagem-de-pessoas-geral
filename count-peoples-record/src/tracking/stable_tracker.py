import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class StableTracker:
    """Gerencia a identidade e contagem de pessoas usando Re-ID e Estabilização."""
    
    def __init__(self, reid_threshold=0.78, merge_threshold=0.86, stability_frames=20, max_samples=15):
        self.reid_threshold = reid_threshold
        self.merge_threshold = merge_threshold
        self.stability_frames = stability_frames
        self.max_samples = max_samples
        
        self.person_db = {}  # {id: {'signatures': [], 'frames_seen': 0, 'merged_into': None}}
        self.next_global_id = 1
        
    def update_batch(self, embeddings):
        """
        Processa múltiplos embeddings de uma vez para garantir unicidade de IDs no frame.
        Retorna uma lista de resultados: [(final_id, is_stable, status_label), ...]
        """
        if not embeddings:
            return []
            
        results = [None] * len(embeddings)
        used_ids_this_frame = set()
        
        # 1. Calcular similaridades de todos contra todos os IDs ativos
        # [embedding_idx][active_id] = max_sim
        active_ids = [pid for pid, data in self.person_db.items() if data['merged_into'] is None]
        
        matches = [] # List of (sim, emb_idx, pid)
        
        for i, emb in enumerate(embeddings):
            if emb is None: continue
            for pid in active_ids:
                data = self.person_db[pid]
                # Pega a maior similaridade com qualquer assinatura do ID
                sims = [cosine_similarity(emb, sig)[0][0] for sig in data['signatures']]
                max_sim = max(sims) if sims else -1
                
                if max_sim > self.reid_threshold:
                    matches.append((max_sim, i, pid))
                    
        # Ordena por similaridade descendente para atribuição gulosa
        matches.sort(key=lambda x: x[0], reverse=True)
        
        assigned_emb_indices = set()
        
        # 2. Atribuição Gulosa (Garante ID único por frame)
        for sim, emb_idx, pid in matches:
            if emb_idx in assigned_emb_indices or pid in used_ids_this_frame:
                continue
                
            # Sucesso no Matching
            assigned_emb_indices.add(emb_idx)
            used_ids_this_frame.add(pid)
            
            self.person_db[pid]['frames_seen'] += 1
            if sim < 0.92 and len(self.person_db[pid]['signatures']) < self.max_samples:
                self.person_db[pid]['signatures'].append(embeddings[emb_idx])
                
            results[emb_idx] = self._prepare_status(pid)
            # Tenta fusão (merging) apenas para IDs que acabaram de ser atualizados
            self._check_for_merges(pid, embeddings[emb_idx])

        # 3. Criar novos IDs para os não atribuídos
        for i, emb in enumerate(embeddings):
            if results[i] is None and emb is not None:
                new_id = self.next_global_id
                self.next_global_id += 1
                self.person_db[new_id] = {
                    'signatures': [emb],
                    'frames_seen': 1,
                    'merged_into': None
                }
                results[i] = self._prepare_status(new_id)
                self._check_for_merges(new_id, emb)
                
        return results

    def _prepare_status(self, best_match_id):
        """Resolve o ID final em caso de merge e prepara o status."""
        final_id = best_match_id
        while self.person_db[final_id]['merged_into'] is not None:
            final_id = self.person_db[final_id]['merged_into']
            
        frames = self.person_db[final_id]['frames_seen']
        is_stable = frames >= self.stability_frames
        
        status = f"ID #{final_id}" if is_stable else f"Analisando ({frames}/{self.stability_frames})"
        return final_id, is_stable, status

    def update(self, embedding):
        """Mantido para compatibilidade, apenas chama update_batch com um item."""
        res = self.update_batch([embedding])
        return res[0] if res else (None, False, "Erro")

    def _check_for_merges(self, current_id, current_embedding):
        """Tenta fundir o ID atual com algum ID estável antigo se forem muito parecidos."""
        # Se já foi fundido ou é o primeiro ID, ignora
        if self.person_db[current_id]['merged_into'] is not None:
            return

        for pid, data in self.person_db.items():
            if pid == current_id or data['merged_into'] is not None:
                continue
            
            # Somente funde se o destino for estável (para evitar cadeia de merges instáveis)
            if data['frames_seen'] >= self.stability_frames:
                for sig in data['signatures']:
                    sim = cosine_similarity(current_embedding, sig)[0][0]
                    if sim > self.merge_threshold:
                        # Faz o merge: transfere assinaturas e frames
                        self.person_db[pid]['signatures'].extend(self.person_db[current_id]['signatures'])
                        # Limita o número de assinaturas após o merge
                        if len(self.person_db[pid]['signatures']) > self.max_samples:
                            self.person_db[pid]['signatures'] = self.person_db[pid]['signatures'][-self.max_samples:]
                            
                        self.person_db[pid]['frames_seen'] += self.person_db[current_id]['frames_seen']
                        self.person_db[current_id]['merged_into'] = pid
                        print(f"Merged ID #{current_id} -> ID #{pid}")
                        return

    def get_total_unique(self):
        """Retorna o número de pessoas únicas confirmadas (estáveis)."""
        return len([p for p in self.person_db.values() 
                    if p['merged_into'] is None and p['frames_seen'] >= self.stability_frames])

    def reset(self):
        """Zera o banco de dados."""
        self.person_db = {}
        self.next_global_id = 1
