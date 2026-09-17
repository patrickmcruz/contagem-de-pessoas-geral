# Workplan 03: Eliminação do Fator 0.0001 e Integração Bayesiana Abs (torch.abs)

**Status:** Em Desenvolvimento  
**Branch:** `feat/03-integracao-bayesiana-abs`  
**Prioridade:** 3º Lugar (Correção Matemática da Formulação de Densidade)  
**Notebook Alvo:** `notebooks/DEF-rgbtcc/02_contagem_pessoas_rgbtcc.ipynb`  
**Referência:** `docs/future-improvements/improvements-by-order.md` | Bayesian Loss (Ma et al. / Feng et al., 2025)  

---

## 🎯 Objetivo da Tarefa

No código legado, existia um fator de escala artificial `0.0001 * raw_sum`. Esse fator foi introduzido como um paliativo empírico para tentar compensar o viés estritamente positivo da ativação `F.softplus(0) = \ln(2) \approx 0.693` somado sobre 1.310.720 pixels ($\approx 900.000 \times 0.0001 \approx 90$ pessoas).

Com a integração da arquitetura oficial do paper (`Net` em `dm_official.py`), a ativação da saída é estritamente `torch.abs(density)`. Sob a formulação de **Supervisão por Perda Bayesiana (Bayesian Loss)** adotada pelo autor, a integral do mapa de densidade:
$$N = \iint_{\Omega} D(x, y) \, dx \, dy \approx \sum_{i, j} D_{i, j}$$
representa diretamente o valor esperado da contagem de pessoas discretas, **sem necessidade de qualquer multiplicador ou fator artificial**.

Além disso, a detecção de picos discretos em alta resolução deve utilizar limiarização adaptativa proporcional à magnitude dos picos da densidade recomposta, eliminando limiares fixos rígidos (como `0.003`) que causavam a perda de 100% dos picos no modelo oficial.

---

## 📋 Lista de Tarefas (Tasks)

- [x] **Task 3.1:** Criar branch `feat/03-integracao-bayesiana-abs` a partir de `develop`.
- [x] **Task 3.2:** Atualizar a documentação matemática da Célula 11 de `02_contagem_pessoas_rgbtcc.ipynb` formalizando a integração Bayesiana direta $\sum D$ e a eliminação definitiva do fator `0.0001`.
- [x] **Task 3.3:** Atualizar a Célula 12 de `02_contagem_pessoas_rgbtcc.ipynb`:
  - Garantir cálculo estrito de `count_calibrated = float(np.sum(density_map_eval))`.
  - Implementar limiar de picos adaptativo baseado no percentil e na magnitude máxima do mapa (`thresh_adaptive = 0.15 * density_map_eval.max()`).
  - Ajustar a janela de vizinhança de picos para `size=5` (adequada para raio de cabeça em drone).
- [x] **Task 3.4:** Testar a integridade numérica e a consistência entre integral contínua e picos detectados em script automatizado.
- [x] **Task 3.5:** Atualizar o `walkthrough.md` com os novos resultados e guia de execução.
- [ ] **Task 3.6:** Entregar ao usuário para validação manual no Jupyter / VS Code.
- [ ] **Task 3.7:** Após validação e aprovação do usuário, realizar o merge em `develop`.
