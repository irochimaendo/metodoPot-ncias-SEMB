[README.md](https://github.com/user-attachments/files/32691215/README.md)
# Método das Potências — Implementação em C

**Equipe:** Tiago de Oliveira Endo Marques, Gustavo Cavalcante Colares

Implementação do **Método das Potências** (Power Method), um algoritmo iterativo que aproxima o autovalor de maior módulo (autovalor dominante) de uma matriz quadrada e o autovetor associado.

---

## 1. Arquivos

| Arquivo | Descrição |
|---|---|
| `metodo_potencias.c` | Implementação do algoritmo em C, com 8 testes de validação embutidos |
| `comparacao_numpy.py` | Script auxiliar em Python que recalcula os mesmos autovalores via `numpy.linalg.eig`, para comparação com uma biblioteca de referência |

---

## 2. Como compilar e rodar

### C
```bash
gcc -Wall -Wextra -o metodo_potencias metodo_potencias.c -lm
./metodo_potencias
```
- `-Wall -Wextra`: ativa avisos extras do compilador.
- `-lm`: linka a biblioteca matemática (necessária por causa do `fabsf`).

No Windows sem WSL (MinGW), gere `metodo_potencias.exe` e rode `metodo_potencias.exe` no lugar de `./metodo_potencias`.

### Python (opcional, apenas para comparação)
```bash
pip install numpy
python comparacao_numpy.py
```

---

## 3. O algoritmo, em poucas palavras

Dada uma matriz `A` (n×n) e um vetor inicial `b0`, o método repete o seguinte processo:

1. Multiplica a matriz pelo vetor atual: `v = A * v`
2. Usa o componente de maior módulo de `v` como estimativa do autovalor (`lambda`)
3. Normaliza `v` dividindo todos os componentes por esse valor
4. Repete até `lambda` parar de mudar (dentro de uma tolerância) ou até atingir o número máximo de iterações

Ao final, `v` converge para o autovetor dominante (normalizado com o maior componente = 1) e `lambda` converge para o autovalor dominante.

**Por que funciona:** ao multiplicar repetidamente por `A`, a componente do vetor na direção do autovetor dominante cresce (relativamente) mais rápido que as demais, então o vetor "gira" em direção a essa componente a cada iteração.

**Complexidade:** a multiplicação matriz-vetor usa dois laços aninhados → O(n²) por iteração → O(k·n²) para `k` iterações.

---

## 4. Estruturas de dados

```c
typedef struct {
    float dado[MAX_N][MAX_N];
    int   n;
} Matriz;

typedef struct {
    float dado[MAX_N];
    int   n;
} Vetor;
```

- Arrays de tamanho fixo (`MAX_N`), **sem alocação dinâmica** (`malloc`/`free`) e **sem recursão** — decisão de projeto voltada para uso futuro em sistemas embarcados (ex.: ESP32), onde memória é limitada e alocação dinâmica é geralmente evitada.
- `n` guarda a dimensão real usada (pode ser menor que `MAX_N`).

---

## 5. Funções principais

| Função | O que faz |
|---|---|
| `multiplica_matriz_vetor` | Calcula `out = A * v` (os dois laços aninhados, O(n²)) |
| `norma_infinito` | Retorna o maior valor absoluto de um vetor e (opcionalmente) o índice desse componente |
| `normaliza_vetor` | Divide todos os componentes de um vetor por um escalar, evitando overflow/underflow ao longo das iterações |
| `metodo_potencias` | Função principal: executa o laço de iterações e retorna o autovalor, preenchendo o autovetor e o número de iterações por ponteiro |
| `residuo_autovalor` | Calcula `‖A·v − λ·v‖` — quanto mais perto de zero, melhor a aproximação. É a checagem que funciona em qualquer matriz, mesmo sem saber o autovalor teórico de antemão |
| `imprime_vetor` | Auxiliar de exibição (trocar por `Serial.print` numa porta para ESP32/Arduino) |

### Critério de parada
O laço para quando `|lambda_atual - lambda_anterior| < TOLERANCIA` (convergência) ou quando atinge `MAX_ITER` (limite de segurança, evita loop infinito em casos que não convergem bem).

---

## 6. Os 8 testes de validação

| Teste | O que valida | Resultado esperado |
|---|---|---|
| 1 — Matriz diagonal | Comparação com autovalor conhecido analiticamente (autovalores são a própria diagonal) | λ = 5.0 |
| 2 — Matriz simétrica 2×2 | Idem, com autovalores calculáveis à mão (`2±1`) | λ = 3.0 |
| 3 — Matriz tridiagonal 3×3 | Idem, com fórmula de referência bibliográfica | λ ≈ 5.4142 |
| 4 — Autovalores próximos | Comportamento quando os dois maiores autovalores estão pertos um do outro | λ ≈ 2.01 |
| 5 — Limite de iterações | Confirma que o algoritmo respeita `max_iter` e não trava, mesmo sem convergência plena | resultado parcial, resíduo alto |
| 6 — Sensibilidade ao vetor inicial | Roda a mesma matriz com 3 vetores iniciais diferentes | mesmo λ nos 3 casos |
| 7 — Variação da tolerância | Roda a mesma matriz com 3 tolerâncias diferentes | mais iterações ⇄ menor resíduo (trade-off precisão × custo) |
| 8 — Autovalores repetidos (caso-limite) | Matriz com autovalor dominante de multiplicidade 2 | λ correto, mas autovetor não é único (depende do vetor inicial) |

Todos os testes imprimem o **resíduo** `‖A·v − λ·v‖` como evidência adicional de que o resultado está correto, independente de já sabermos o valor teórico.

---

## 7. Comparação externa (`comparacao_numpy.py`)

Recalcula os autovalores das matrizes dos Testes 1, 2, 3, 4 e 8 usando `numpy.linalg.eig`, uma implementação de referência amplamente usada. Serve como segunda fonte de validação, independente da nossa implementação.

**Observação sobre os autovetores:** o numpy normaliza por norma euclidiana (`‖v‖ = 1`), enquanto nosso código em C normaliza pelo maior componente em módulo (`max|v_i| = 1`). Por isso os vetores aparecem com valores diferentes — mas representam a mesma direção. Os **autovalores** são o que deve ser comparado diretamente, e eles batem em todos os casos testados.

---

## 8. Limitações conhecidas

- Requer que o vetor inicial não seja ortogonal ao autovetor dominante (na prática, um vetor com todos os componentes iguais a 1 quase sempre funciona).
- A velocidade de convergência do *autovetor* depende da razão `|λ₂/λ₁|` entre o segundo e o primeiro autovalor em módulo — quanto mais próxima de 1, mais lenta (ver Teste 4).
- Quando o autovalor dominante tem multiplicidade maior que 1, o autovalor continua sendo encontrado corretamente, mas não existe um autovetor único — o método converge para um vetor dentro do subespaço correspondente (ver Teste 8).
- `MAX_N` limita o tamanho máximo de matriz suportado (ajustável em tempo de compilação).

---

## 9. Referência

WIKIPÉDIA. *Método das potências*. Disponível em: https://pt.wikipedia.org/wiki/M%C3%A9todo_das_pot%C3%AAncias
