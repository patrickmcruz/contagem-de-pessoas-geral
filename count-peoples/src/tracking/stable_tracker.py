import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class StableTracker:
    """Gerencia a identidade e contagem de pessoas usando Re-ID e Estabilização."""
    
    def __init__(self, reid_threshold=0.72, merge_threshold=0.82, stability_frames=20, max_samples=15):
        self.reid_threshold = reid_threshold
        self.merge_threshold = merge_threshold
        self.stability_frames = stability_frames
        self.max_samples = max_samples
        
        self.person_db = {}  # {id: {'signatures': [], 'frames_seen': 0, 'merged_into': None}}
        self.next_global_id = 1
        
    def update(self, embedding):
        """
        Atualiza o estado do rastreador com um novo embedding de pessoa detectada.
        Retorna: (assigned_id, is_stable, status_label)
        """
        if embedding is None:
            return None, False, "Erro"
            
        best_match_id = None
        highest_sim = -1
        
        # 1. Matching contra IDs ativos (não fundidos)
        active_ids = [pid for pid, data in self.person_db.items() if data['merged_into'] is None]
        
        for pid in active_ids:
            data = self.person_db[pid]
            for sig in data['signatures']:
                sim = cosine_similarity(embedding, sig)[0][0]
                if sim > self.reid_threshold and sim > highest_sim:
                    highest_sim = sim
                    best_match_id = pid
                    
        # 2. Atribuição ou Criação
        if best_match_id is not None:
            self.person_db[best_match_id]['frames_seen'] += 1
            # Atualiza banco visual com variações
            if highest_sim < 0.90 and len(self.person_db[best_match_id]['signatures']) < self.max_samples:
                self.person_db[best_match_id]['signatures'].append(embedding)
        else:
            best_match_id = self.next_global_id
            self.next_global_id += 1
            self.person_db[best_match_id] = {
                'signatures': [embedding], 
                'frames_seen': 1, 
                'merged_into': None
            }
            
        # 3. Check para Fusão (Merging)
        self._check_for_merges(best_match_id, embedding)
            
        # 4. Status
        final_id = best_match_id
        while self.person_db[final_id]['merged_into'] is not None:
            final_id = self.person_db[final_id]['merged_into']
            
        frames = self.person_db[final_id]['frames_seen']
        is_stable = frames >= self.stability_frames
        
        status = f"ID #{final_id}" if is_stable else f"Analisando ({frames}/{self.stability_frames})"
        
        return final_id, is_stable, status

    def _check_for_merges(self, current_id, current_embedding):
        """Tenta fundir o ID atual com algum ID estável antigo se forem muito parecidos."""
        if current_id == 1 or self.person_db[current_id]['merged_into'] is not None:
            return

        for pid, data in self.person_db.items():
            if pid == current_id or data['merged_into'] is not None:
                continue
            
            if data['frames_seen'] >= self.stability_frames:
                for sig in data['signatures']:
                    sim = cosine_similarity(current_embedding, sig)[0][0]
                    if sim > self.merge_threshold:
                        self.person_db[pid]['signatures'].extend(self.person_db[current_id]['signatures'])
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
