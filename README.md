# lab

Experimentos numéricos em duas séries independentes. Originais em português nas pastas `entre/` e `bravais/`, versões em inglês em [`en/`](en/) — o conteúdo dos originais não foi modificado.

*Numerical experiments in two independent series. Portuguese originals in `entre/` and `bravais/`, English versions in [`en/`](en/) — the originals' contents were not modified.*

```text
lab-main/
├── README.md
├── LICENSE
│
├── entre/                        Série 1 · o "entre"
│   ├── simulate_observer_observed_relations.py    (1) base
│   ├── simulate_consciencia_entre.py              (2) + medidas do "entre"
│   ├── simulate_neurotransmissores_entre.py       (3) + 8 canais químicos
│   ├── simulate_filtro_vida_entre.py              (4) + memória e filtro de vida
│   ├── dados/                                     amostras JSON de cada script
│   └── resultados/                                PNG/GIF já gerados
│
├── bravais/                      Série 2 · Bravais 3D
│   ├── bravais_puro_3d.py                         evolução principal
│   ├── bravais_cordas.py                          pós-processamento (cordas)
│   ├── bravais_sweep_L.py                         teste de escala própria
│   ├── final_state.npz                            estado final salvo
│   └── resultados/                                PNG já gerados
│
└── en/                           tudo acima em inglês
    ├── entre/                                     4 scripts + 4 JSONs
    └── bravais/                                   3 scripts
```

---

## Série 1 — Observador/observado e o "entre"

*Observer/observed and the "between"*

Osciladores de fase acoplados (RK4, memória contínua por aresta) em quatro cenários — unilateral, mútuo, aninhado e "entre dos entres". Cada script **empilha uma camada** sobre o anterior, sem mudar a lei base:

```mermaid
graph LR
    A["(1) base<br>fases + memória de aresta"] --> B["(2) medidas do 'entre'<br>(nenhuma força nova)"]
    B --> C["(3) + 8 neurotransmissores<br>contínuos acoplados"]
    C --> D["(4) + 4 memórias<br>+ filtro de vida"]
```

### (1) Base — quem observa muda o observado?

`entre/simulate_observer_observed_relations.py` → `en/entre/simulate_observer_observed_relations.py`
Dados: `entre/dados/observador-observado-relacoes-dados.json` → `en/entre/observer-observed-relations-data.json`

![Observador e observado](entre/resultados/observador-observado-relacoes.png)

![Animação observador/observado](entre/resultados/observador-observado-relacoes.gif)

### (2) O "entre" — um estado que nasce no vínculo

`entre/simulate_consciencia_entre.py` → `en/entre/simulate_consciousness_between.py`
Dados: `entre/dados/consciencia-entre-dados.json` → `en/entre/consciousness-between-data.json`

![Consciência emergida no entre](entre/resultados/consciencia-emergida-no-entre.png)

![Animação do entre](entre/resultados/consciencia-emergida-no-entre.gif)

### (3) Química — a mesma relação com 8 canais acoplados

`entre/simulate_neurotransmissores_entre.py` → `en/entre/simulate_neurotransmitters_between.py`
Dados: `entre/dados/neurotransmissores-entre-dados.json` → `en/entre/neurotransmitters-between-data.json`

![Neurotransmissores no entre](entre/resultados/neurotransmissores-mudanca-no-entre.png)

![Animação neurotransmissores](entre/resultados/neurotransmissores-mudanca-no-entre.gif)

### (4) Filtro de vida — mesma lei, experiências diferentes

`entre/simulate_filtro_vida_entre.py` → `en/entre/simulate_life_filter_between.py`
Dados: `entre/dados/filtro-vida-entre-dados.json` → `en/entre/life-filter-between-data.json`

![Memória e filtro de vida](entre/resultados/memoria-filtro-vida-mudanca.png)

![Animação filtro de vida](entre/resultados/memoria-filtro-vida-mudanca.gif)

### Extras da série

![Campo quântico, três cristalizações](entre/resultados/campo-quantico-tres-cristalizacoes.png)

![Cristalização fixa vs contínua](entre/resultados/cristalizacao-fixa-vs-continua.png)

> Ao executar, cada script grava PNG/GIF/JSON/NPZ em `artifacts/` ao lado do próprio script.
> *When run, each script writes PNG/GIF/JSON/NPZ to `artifacts/` next to itself.*

---

## Série 2 — Bravais 3D emergente

*Emergent 3D Bravais*

Campo complexo Ψ em grade 3D periódica (FFT). Todos os coeficientes emergem de **uma única proposta acoplada** com ponto fixo implícito — sem calibração externa, sem colapso forçado.

```mermaid
graph LR
    P["bravais_puro_3d.py<br>evolução + espectro"] -->|final_state.npz| C["bravais_cordas.py<br>cordas de densidade/fase/nota"]
    P -.mesma equação, caixas L=24/32/48.-> S["bravais_sweep_L.py<br>a beirada é do sistema<br>ou do recipiente?"]
```

### Evolução e rede emergente

`bravais/bravais_puro_3d.py` → `en/bravais/bravais_pure_3d.py`

![Evolução da densidade](bravais/resultados/pure_evolution.png)

![Cortes finais](bravais/resultados/pure_final_slices.png)

![Picos 3D e vetores de rede](bravais/resultados/pure_final_peaks3d.png)

![Isosuperfície](bravais/resultados/pure_isosurface.png)

### Espectro e testes de forma

![FFT fatias centrais](bravais/resultados/pure_fft.png)

![Espectro em 9 ângulos](bravais/resultados/pure_fft_views.png)

![Relevo do espectro](bravais/resultados/pure_fft_relief.png)

![Casca 3D do espectro](bravais/resultados/pure_fft_shell.png)

![Teste da pirâmide L1 vs L2](bravais/resultados/pure_fft_pyramid_test.png)

![Andares do espectro](bravais/resultados/pure_fft_floors.png)

![Raios direcionais](bravais/resultados/pure_fft_rays.png)

![Projeções de máximo](bravais/resultados/pure_fft_projections.png)

![Espectro binarizado](bravais/resultados/pure_fft_bw.png)

![Densidade binarizada](bravais/resultados/pure_density_bw.png)

![Autocorrelação](bravais/resultados/pure_autocorr.png)

![Simetrias medidas](bravais/resultados/pure_symmetry.png)

![Nascimento da beirada no tempo](bravais/resultados/pure_fft_time.png)

![Séries temporais](bravais/resultados/pure_timeseries.png)

### Cordas (pós-processamento do `final_state.npz`)

`bravais/bravais_cordas.py` → `en/bravais/bravais_strings.py`

```sh
cd bravais && python3 bravais_cordas.py final_state.npz
```

![Cordas de densidade](bravais/resultados/cordas_densidade.png)

![Cordas-nota](bravais/resultados/cordas_notas.png)

### Teste de escala própria

`bravais/bravais_sweep_L.py` → `en/bravais/bravais_sweep_L.py` — mesma equação em caixas L=24/32/48 com dx fixo; gera `sweep_verdict.png` em `bravais_outputs_3d/`.

---

## Traduções · *Translations*

| Original (PT) | English | Conteúdo · *Content* |
|---|---|---|
| `entre/simulate_observer_observed_relations.py` | `en/entre/simulate_observer_observed_relations.py` | dinâmica base · *base dynamics* |
| `entre/simulate_consciencia_entre.py` | `en/entre/simulate_consciousness_between.py` | medidas do "entre" · *"between" measurements* |
| `entre/simulate_neurotransmissores_entre.py` | `en/entre/simulate_neurotransmitters_between.py` | camada química · *chemical layer* |
| `entre/simulate_filtro_vida_entre.py` | `en/entre/simulate_life_filter_between.py` | memória + filtro de vida · *memory + life filter* |
| `bravais/bravais_puro_3d.py` | `en/bravais/bravais_pure_3d.py` | evolução 3D · *3D evolution* |
| `bravais/bravais_cordas.py` | `en/bravais/bravais_strings.py` | cordas · *strings* |
| `bravais/bravais_sweep_L.py` | `en/bravais/bravais_sweep_L.py` | varredura de caixa · *box sweep* |
| `entre/dados/consciencia-entre-dados.json` | `en/entre/consciousness-between-data.json` | 25 amostras · *25 samples* |
| `entre/dados/observador-observado-relacoes-dados.json` | `en/entre/observer-observed-relations-data.json` | 25 amostras · *25 samples* |
| `entre/dados/neurotransmissores-entre-dados.json` | `en/entre/neurotransmitters-between-data.json` | 25 amostras · *25 samples* |
| `entre/dados/filtro-vida-entre-dados.json` | `en/entre/life-filter-between-data.json` | 25 amostras · *25 samples* |

Nas versões em inglês a dinâmica e os números são idênticos; mudam apenas textos, rótulos, chaves de JSON (`tempo`→`time`, `mutuo`→`mutual`, `aninhado`→`nested`, `entre_entres`→`between_of_betweens`, `dois_pares`→`two_pairs`, `*_fase`→`*_phase`, `*_entre`→`*_between`, `*_filtro`→`*_filter`) e nomes de saída (ex.: `cordas_densidade.png`→`strings_density.png`).

*English versions keep dynamics and numbers identical; only texts, labels, JSON keys and output names change.*

## Dependências · *Dependencies*

```text
numpy
matplotlib
pillow        # GIFs
scipy         # opcional · optional (fallback numpy incluso · numpy fallback included)
scikit-image  # opcional · optional (isosuperfícies · isosurfaces)
```

```sh
pip install numpy matplotlib pillow scipy scikit-image
```
