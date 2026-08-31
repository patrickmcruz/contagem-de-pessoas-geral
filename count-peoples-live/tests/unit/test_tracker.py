import pytest
import numpy as np
from src.tracking.stable_tracker import StableTracker

def test_tracker_stability():
    """Verifica se o tracker só conta a pessoa após o número mínimo de frames."""
    tracker = StableTracker(stability_frames=5)
    
    # Simula 4 frames da mesma pessoa (mesmo embedding)
    dummy_embedding = np.random.rand(1, 1024)
    
    for i in range(4):
        assigned_id, is_stable, _ = tracker.update(dummy_embedding)
        assert assigned_id == 1
        assert is_stable is False
        assert tracker.get_total_unique() == 0
        
    # No 5º frame, deve ficar estável e contar
    _, is_stable, _ = tracker.update(dummy_embedding)
    assert is_stable is True
    assert tracker.get_total_unique() == 1

def test_tracker_merging():
    """Verifica se dois IDs muito parecidos são fundidos."""
    tracker = StableTracker(stability_frames=2, merge_threshold=0.99)
    
    # Pessoa 1 (ID #1)
    sig1 = np.ones((1, 1024))
    tracker.update(sig1)
    tracker.update(sig1) # Torna estável
    assert tracker.get_total_unique() == 1
    
    # Pessoa 2 (ID #2) que é IDÊNTICA (sim = 1.0)
    # Por causa do merge_threshold alto, ele deve fundir imediatamente
    tracker.update(sig1) 
    
    # O total deve continuar 1 porque o ID#2 foi fundido ao ID#1
    assert tracker.get_total_unique() == 1
