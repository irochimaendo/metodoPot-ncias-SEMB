#include <stdio.h>
#include <math.h>

#define MAX_N        45      /* Tamanho maximo de matriz suportado.
                                 AJUSTE este valor conforme o tamanho de
                                 matriz exigido no enunciado do trabalho. */
#define MAX_ITER     1000     /* Numero maximo de iteracoes              */
#define TOLERANCIA   1e-6f    /* Criterio de convergencia (|dlambda|)    */

/* Matriz quadrada de ordem n <= MAX_N, armazenada em array estatico    */
typedef struct {
    float dado[MAX_N][MAX_N];
    int   n;
} Matriz;

/* Vetor de dimensao n <= MAX_N, armazenado em array estatico           */
typedef struct {
    float dado[MAX_N];
    int   n;
} Vetor;

/* out = A * v  (multiplicacao matriz-vetor, O(n^2)) */
static void multiplica_matriz_vetor(const Matriz *A, const Vetor *v, Vetor *out) {
    out->n = A->n;
    for (int i = 0; i < A->n; i++) {
        float soma = 0.0f;
        for (int j = 0; j < A->n; j++) {
            soma += A->dado[i][j] * v->dado[j];
        }
        out->dado[i] = soma;
    }
}

/* Retorna a norma infinita (maior valor absoluto) de um vetor.
 * Se indice_max != NULL, tambem devolve o indice do componente de
 * maior modulo (usado para recuperar o sinal correto do autovalor). */
static float norma_infinito(const Vetor *v, int *indice_max) {
    float max = fabsf(v->dado[0]);
    int idx = 0;
    for (int i = 1; i < v->n; i++) {
        float abs_val = fabsf(v->dado[i]);
        if (abs_val > max) {
            max = abs_val;
            idx = i;
        }
    }
    if (indice_max != NULL) *indice_max = idx;
    return max;
}

/* Normaliza o vetor dividindo todos os componentes por 'escalar'.
 * Evita que os valores cresçam ou diminuam sem controle ao longo das
 * iteracoes (fundamental em embarcados, onde o range de float é
 * limitado). */
static void normaliza_vetor(Vetor *v, float escalar) {
    if (fabsf(escalar) < 1e-12f) return; /* evita divisao por zero */
    for (int i = 0; i < v->n; i++) {
        v->dado[i] /= escalar;
    }
}

/*
 * metodo_potencias
 *
 * Entradas:
 *   A          - matriz quadrada de entrada
 *   b0         - vetor inicial (nao pode ser ortogonal ao autovetor
 *                dominante; na pratica, um vetor com todos os
 *                componentes iguais a 1 funciona bem na maioria dos casos)
 *   max_iter   - numero maximo de iteracoes permitido
 *   tolerancia - criterio de parada: |lambda_atual - lambda_anterior| < tol
 *
 * Saidas (por ponteiro):
 *   autovetor            - autovetor associado ao autovalor dominante,
 *                          normalizado (maior componente em modulo = 1)
 *   iteracoes_realizadas - numero de iteracoes ate a convergencia
 *
 * Retorno:
 *   aproximacao do autovalor dominante (float, com sinal)
 */
float metodo_potencias(const Matriz *A, const Vetor *b0, int max_iter,
                        float tolerancia, Vetor *autovetor,
                        int *iteracoes_realizadas) {
    Vetor v_atual = *b0;
    Vetor v_prox;
    float lambda_atual = 0.0f, lambda_anterior = 0.0f;
    int idx_max;

    /* Normaliza o vetor inicial antes de comecar as iteracoes */
    float norma_inicial = norma_infinito(&v_atual, &idx_max);
    normaliza_vetor(&v_atual, norma_inicial);

    int k;
    for (k = 0; k < max_iter; k++) {
        /* 1) Multiplicacao matriz-vetor: v_prox = A * v_atual */
        multiplica_matriz_vetor(A, &v_atual, &v_prox);

        /* 2) Estimativa do autovalor: como v_atual tem maior componente
         *    em modulo igual a 1, o maior componente de A*v_atual
         *    converge para o proprio autovalor dominante (em modulo). */
        lambda_atual = norma_infinito(&v_prox, &idx_max);
        if (v_prox.dado[idx_max] < 0) lambda_atual = -lambda_atual;

        /* 3) Normalizacao do vetor resultante (prepara a proxima iteracao) */
        normaliza_vetor(&v_prox, fabsf(lambda_atual));

        /* 4) Criterio de convergencia */
        if (k > 0 && fabsf(lambda_atual - lambda_anterior) < tolerancia) {
            v_atual = v_prox;
            k++;
            break;
        }

        lambda_anterior = lambda_atual;
        v_atual = v_prox;
    }

    *autovetor = v_atual;
    *iteracoes_realizadas = k;
    return lambda_atual;
}

static void imprime_vetor(const Vetor *v) {
    printf("[ ");
    for (int i = 0; i < v->n; i++) {
        printf("%9.5f ", v->dado[i]);
    }
    printf("]\n");
}

/*
 * residuo_autovalor
 *
 * Calcula ||A*v - lambda*v|| (norma infinita da diferenca) para o par
 * (lambda, v) obtido pelo metodo_potencias. Esta e a checagem mais
 * geral de validacao: funciona para QUALQUER matriz, mesmo quando o
 * autovalor teorico nao e conhecido de antemao. Quanto mais proximo de
 * zero, melhor a aproximacao (o valor esperado e da ordem da
 * TOLERANCIA usada na convergencia).
 */
static float residuo_autovalor(const Matriz *A, const Vetor *v, float lambda) {
    Vetor Av;
    multiplica_matriz_vetor(A, v, &Av);

    Vetor diferenca;
    diferenca.n = v->n;
    for (int i = 0; i < v->n; i++) {
        diferenca.dado[i] = Av.dado[i] - lambda * v->dado[i];
    }

    return norma_infinito(&diferenca, NULL);
}

/*
 * imprime_comparacao_vetores
 *
 * Imprime tres vetores lado a lado, para permitir uma comparacao
 * VISUAL, alem do residuo numerico:
 *   1) o vetor inicial informado (b0)
 *   2) o autovetor final, apos convergencia (ja normalizado)
 *   3) o vetor A * autovetor_final -- que, se o algoritmo convergiu
 *      bem, deve ficar bem proximo de "lambda * autovetor_final"
 *      (e exatamente essa diferenca que o residuo mede como um unico
 *      numero; aqui da para comparar componente a componente).
 */
static void imprime_comparacao_vetores(const Matriz *A, const Vetor *b0,
                                        const Vetor *autovetor, float lambda) {
    Vetor Av;
    multiplica_matriz_vetor(A, autovetor, &Av);

    printf("  Vetor inicial (informado):             "); imprime_vetor(b0);
    printf("  Autovetor final (normalizado):         "); imprime_vetor(autovetor);
    printf("  A * autovetor final (comparar c/ %.4f*v acima): ", lambda);
    imprime_vetor(&Av);
}

/*
 * le_entrada_teclado
 *
 * Pergunta ao usuario, na propria execucao do programa (via stdin), a
 * ordem da matriz, os elementos da matriz, o vetor inicial, o numero
 * maximo de iteracoes e a tolerancia. Nao depende de nenhum arquivo
 * externo -- os dados sao digitados na hora.
 */
void le_entrada_teclado(Matriz *A, Vetor *b0, int *max_iter, float *tolerancia) {
    int n;
    printf("\nDigite a ordem da matriz (n, entre 1 e %d): ", MAX_N);
    while (scanf("%d", &n) != 1 || n <= 0 || n > MAX_N) {
        printf("Valor invalido. Digite n entre 1 e %d: ", MAX_N);
        while (getchar() != '\n'); /* limpa entrada invalida do buffer */
    }
    A->n = n;

    printf("Digite os %d x %d elementos da matriz, linha por linha:\n", n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("  A[%d][%d] = ", i, j);
            scanf("%f", &A->dado[i][j]);
        }
    }

    b0->n = n;
    printf("Digite os %d elementos do vetor inicial:\n", n);
    for (int i = 0; i < n; i++) {
        printf("  b0[%d] = ", i);
        scanf("%f", &b0->dado[i]);
    }

    printf("Numero maximo de iteracoes (sugestao: %d): ", MAX_ITER);
    if (scanf("%d", max_iter) != 1 || *max_iter <= 0) {
        *max_iter = MAX_ITER;
    }

    printf("Tolerancia de convergencia (sugestao: %g): ", TOLERANCIA);
    if (scanf("%f", tolerancia) != 1 || *tolerancia <= 0.0f) {
        *tolerancia = TOLERANCIA;
    }
}

/*
 * escreve_saida_arquivo
 *
 * Grava o resultado do metodo das potencias (autovalor, autovetor,
 * numero de iteracoes e residuo) em um arquivo texto legivel, para
 * conferencia posterior sem precisar reler a saida do terminal. Isso
 * e so a SAIDA do programa -- nao tem relacao com como os dados de
 * ENTRADA sao fornecidos (exemplo embutido ou teclado).
 */
void escreve_saida_arquivo(const char *caminho, float lambda,
                            const Vetor *autovetor, int iteracoes,
                            float residuo) {
    FILE *f = fopen(caminho, "w");
    if (f == NULL) {
        fprintf(stderr, "Erro: nao foi possivel criar o arquivo de saida '%s'\n", caminho);
        return;
    }

    fprintf(f, "Autovalor dominante: %.6f\n", lambda);
    fprintf(f, "Autovetor associado (%d componentes):\n", autovetor->n);
    for (int i = 0; i < autovetor->n; i++) {
        fprintf(f, "  v[%d] = %.6f\n", i, autovetor->dado[i]);
    }
    fprintf(f, "Iteracoes realizadas: %d\n", iteracoes);
    fprintf(f, "Residuo ||A*v - lambda*v||: %.8f\n", residuo);

    fclose(f);
    printf("Resultado salvo em '%s'\n", caminho);
}

static void roda_exemplo_embutido(void) {
    printf("\n=== ENTRADA DE EXEMPLO (matriz de grande porte gerada programaticamente) ===\n");

    Matriz A;
    int n = 45; /* 45 x 45 = 2025 elementos float (~8.1 Kbytes) */
    A.n = n;

    /* Preenchimento programático da matriz (exemplo tridiagonal escalável) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A.dado[i][j] = 6.0f;
            } else if (j == i + 1 || j == i - 1) {
                A.dado[i][j] = 1.0f;
            } else {
                A.dado[i][j] = 3.0f;
            }
        }
    }

    Vetor b0;
    b0.n = n;
    for (int i = 0; i < n; i++) {
        b0.dado[i] = 1.0f; /* Vetor inicial com componentes iguais a 1 */
    }

    int   max_iter   = MAX_ITER;
    float tolerancia = TOLERANCIA;

    Vetor autovetor;
    int iteracoes;
    float lambda = metodo_potencias(&A, &b0, max_iter, tolerancia, &autovetor, &iteracoes);
    float residuo = residuo_autovalor(&A, &autovetor, lambda);

    printf("Ordem da matriz (n): %d (%d elementos float)\n", A.n, A.n * A.n);
    printf("Autovalor dominante (lambda):  %.6f\n", lambda);
    printf("Iteracoes realizadas:          %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||:    %.8f\n\n", residuo);
    
    imprime_comparacao_vetores(&A, &b0, &autovetor, lambda);

    escreve_saida_arquivo("resultado.txt", lambda, &autovetor, iteracoes, residuo);
}

/* Teste 1: matriz diagonal -> autovalores sao os proprios elementos da
 * diagonal. Autovalor dominante esperado: 5.0 */
static void teste1_matriz_diagonal(void) {
    printf("\n--- TESTE 1: Matriz diagonal ---\n");
    Matriz A = { .n = 3 };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A.dado[i][j] = 0.0f;
    A.dado[0][0] = 2.0f;
    A.dado[1][1] = 5.0f;
    A.dado[2][2] = -3.0f;

    Vetor b0 = { .n = 3, .dado = {1.0f, 1.0f, 1.0f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado: %.6f (esperado: 5.0)\n", lambda);
    printf("Autovetor: "); imprime_vetor(&autovetor);
    printf("Iteracoes: %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f\n", residuo_autovalor(&A, &autovetor, lambda));
}

/* Teste 2: matriz simetrica 2x2 classica A = [[2,1],[1,2]].
 * Autovalores conhecidos: 3 e 1 -> dominante = 3 */
static void teste2_matriz_simetrica(void) {
    printf("\n--- TESTE 2: Matriz simetrica 2x2 ---\n");
    Matriz A = { .n = 2, .dado = {{2.0f, 1.0f}, {1.0f, 2.0f}} };
    Vetor b0 = { .n = 2, .dado = {1.0f, 0.0f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado: %.6f (esperado: 3.0)\n", lambda);
    printf("Autovetor: "); imprime_vetor(&autovetor);
    printf("Iteracoes: %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f\n", residuo_autovalor(&A, &autovetor, lambda));
}

/* Teste 3: matriz tridiagonal simetrica 3x3, A = [[4,1,0],[1,4,1],[0,1,4]].
 * Autovalores: 4, 4+sqrt(2)~=5.4142, 4-sqrt(2)~=2.5858
 * Autovalor dominante esperado: ~5.4142 */
static void teste3_matriz_tridiagonal(void) {
    printf("\n--- TESTE 3: Matriz tridiagonal 3x3 ---\n");
    Matriz A = { .n = 3, .dado = {{4.0f,1.0f,0.0f},{1.0f,4.0f,1.0f},{0.0f,1.0f,4.0f}} };
    Vetor b0 = { .n = 3, .dado = {1.0f, 0.0f, 0.0f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado: %.6f (esperado: ~5.4142)\n", lambda);
    printf("Autovetor: "); imprime_vetor(&autovetor);
    printf("Iteracoes: %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f\n", residuo_autovalor(&A, &autovetor, lambda));
}

/* Teste 4: autovalores proximos (2.01 e 2.00). A teoria diz que a taxa
 * de convergencia do Metodo das Potencias depende da razao entre o
 * segundo e o primeiro autovalor em modulo (|lambda2/lambda1|); quanto
 * mais proxima de 1, mais iteracoes sao necessarias para o AUTOVETOR
 * convergir. Neste caso, como a matriz e diagonal e o vetor inicial ja
 * esta alinhado com os eixos, a ESTIMATIVA do autovalor (maior
 * componente) se estabiliza cedo mesmo com convergencia lenta do
 * autovetor -- por isso o numero de iteracoes reportado aqui pode ser
 * baixo. Serve para discutir na apresentacao a diferenca entre
 * convergencia do autovalor e do autovetor. */
static void teste4_convergencia_lenta(void) {
    printf("\n--- TESTE 4: Autovalores proximos (2.01 e 2.00) ---\n");
    Matriz A = { .n = 2, .dado = {{2.01f, 0.0f}, {0.0f, 2.00f}} };
    Vetor b0 = { .n = 2, .dado = {1.0f, 1.0f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado: %.6f (esperado: ~2.01)\n", lambda);
    printf("Iteracoes ate a estimativa do autovalor estabilizar: %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f\n", residuo_autovalor(&A, &autovetor, lambda));
}

/* Teste 5: limite de iteracoes atingido antes de convergir
 * (max_iter artificialmente baixo, = 1). Valida que o algoritmo nao
 * trava, respeita o limite maximo e retorna um resultado parcial de
 * forma segura mesmo sem satisfazer o criterio de tolerancia. */
static void teste5_limite_iteracoes(void) {
    printf("\n--- TESTE 5: Limite de iteracoes atingido (max_iter=1) ---\n");
    Matriz A = { .n = 3, .dado = {{4.0f,1.0f,0.0f},{1.0f,4.0f,1.0f},{0.0f,1.0f,4.0f}} };
    Vetor b0 = { .n = 3, .dado = {1.0f, 0.0f, 0.0f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, 1, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado (parcial, sem convergencia plena): %.6f\n", lambda);
    printf("Iteracoes: %d (limitado por max_iter=1, longe do valor final ~5.4142)\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f (esperado: bem maior que nos testes convergidos)\n",
           residuo_autovalor(&A, &autovetor, lambda));
}

/* Teste 6: sensibilidade ao vetor inicial. Mesma matriz do Teste 3, tres
 * vetores iniciais diferentes -> deve convergir para o MESMO autovalor
 * dominante em todos os casos (o autovetor pode sair com sinal trocado,
 * o que e normal e nao indica erro). */
static void teste6_sensibilidade_vetor_inicial(void) {
    printf("\n--- TESTE 6: Sensibilidade ao vetor inicial (matriz do Teste 3) ---\n");
    Matriz A = { .n = 3, .dado = {{4.0f,1.0f,0.0f},{1.0f,4.0f,1.0f},{0.0f,1.0f,4.0f}} };

    Vetor iniciais[3] = {
        { .n = 3, .dado = {1.0f, 0.0f, 0.0f} },
        { .n = 3, .dado = {0.0f, 1.0f, 0.0f} },
        { .n = 3, .dado = {1.0f, 1.0f, 1.0f} }
    };

    for (int t = 0; t < 3; t++) {
        Vetor autovetor;
        int iteracoes;
        float lambda = metodo_potencias(&A, &iniciais[t], MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);
        printf("Vetor inicial %d: lambda = %.6f (esperado: ~5.4142), iteracoes = %d\n",
               t + 1, lambda, iteracoes);
    }
    printf("Conclusao: o autovalor dominante converge para o mesmo valor\n");
    printf("independente do vetor inicial escolhido (exceto no caso raro de\n");
    printf("o vetor inicial ser ortogonal ao autovetor dominante).\n");
}

/* Teste 7: variacao da tolerancia. Mesma matriz e vetor inicial do Teste 3,
 * tolerancias diferentes -> demonstra o trade-off entre precisao (residuo
 * menor) e numero de iteracoes (custo computacional), especialmente
 * relevante para embarcados com recursos limitados. */
static void teste7_variacao_tolerancia(void) {
    printf("\n--- TESTE 7: Variacao da tolerancia (matriz do Teste 3) ---\n");
    Matriz A = { .n = 3, .dado = {{4.0f,1.0f,0.0f},{1.0f,4.0f,1.0f},{0.0f,1.0f,4.0f}} };
    Vetor b0 = { .n = 3, .dado = {1.0f, 0.0f, 0.0f} };

    float tolerancias[3] = {1e-2f, 1e-4f, 1e-6f};
    for (int t = 0; t < 3; t++) {
        Vetor autovetor;
        int iteracoes;
        float lambda = metodo_potencias(&A, &b0, MAX_ITER, tolerancias[t], &autovetor, &iteracoes);
        printf("Tolerancia = %.0e: lambda = %.6f, iteracoes = %d, residuo = %.8f\n",
               (double)tolerancias[t], lambda, iteracoes, residuo_autovalor(&A, &autovetor, lambda));
    }
    printf("Conclusao: tolerancia mais apertada exige mais iteracoes, mas reduz\n");
    printf("o residuo -- trade-off entre precisao e custo computacional.\n");
}

/* Teste 8: autovalores repetidos (caso-limite). Matriz diagonal com
 * autovalor dominante de multiplicidade 2 (3, 3, 1): o subespaco associado
 * ao autovalor dominante tem dimensao 2, entao o "autovetor" para o qual
 * o metodo converge depende da projecao do vetor inicial nesse subespaco
 * -- nao ha um autovetor unico neste caso. O AUTOVALOR, porem, continua
 * convergindo corretamente. Bom exemplo de limitacao do metodo para
 * discutir nas dificuldades tecnicas. */
static void teste8_autovalores_repetidos(void) {
    printf("\n--- TESTE 8: Autovalores repetidos (caso-limite) ---\n");
    Matriz A = { .n = 3 };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            A.dado[i][j] = 0.0f;
    A.dado[0][0] = 3.0f;
    A.dado[1][1] = 3.0f;
    A.dado[2][2] = 1.0f;

    Vetor b0 = { .n = 3, .dado = {1.0f, 2.0f, 0.5f} };
    Vetor autovetor;
    int iteracoes;

    float lambda = metodo_potencias(&A, &b0, MAX_ITER, TOLERANCIA, &autovetor, &iteracoes);

    printf("Autovalor calculado: %.6f (esperado: 3.0, com multiplicidade 2)\n", lambda);
    printf("Autovetor: "); imprime_vetor(&autovetor);
    printf("Iteracoes: %d\n", iteracoes);
    printf("Residuo ||A*v - lambda*v||: %.8f\n", residuo_autovalor(&A, &autovetor, lambda));
    printf("Nota: o autovetor aqui NAO e unico -- depende da direcao inicial\n");
    printf("dentro do subespaco de autovalor 3 (multiplicidade 2). O autovalor\n");
    printf("continua correto, mas o metodo so identifica UM representante do\n");
    printf("subespaco, nao um autovetor canonico.\n");
}

int main(void) {
    int escolha;

    printf("=====================================================\n");
    printf(" Metodo das Potencias -- Menu Principal\n");
    printf("=====================================================\n");
    printf("1 - Rodar os 8 testes de validacao (matrizes de exemplo no codigo)\n");
    printf("2 - Rodar com o dado de entrada de exemplo embutido no codigo\n");
    printf("3 - Digitar uma matriz pelo teclado\n");
    printf("Escolha uma opcao: ");

    if (scanf("%d", &escolha) != 1) {
        escolha = 1; /* entrada invalida -> cai no modo padrao (testes) */
    }

    if (escolha == 2) {
        roda_exemplo_embutido();
        return 0;
    }

    if (escolha == 3) {
        Matriz A;
        Vetor b0;
        int max_iter;
        float tolerancia;

        le_entrada_teclado(&A, &b0, &max_iter, &tolerancia);

        Vetor autovetor;
        int iteracoes;
        float lambda = metodo_potencias(&A, &b0, max_iter, tolerancia, &autovetor, &iteracoes);
        float residuo = residuo_autovalor(&A, &autovetor, lambda);

        printf("\n--- RESULTADO ---\n");
        printf("Autovalor dominante (lambda):  %.6f\n", lambda);
        printf("Iteracoes realizadas:          %d\n", iteracoes);
        printf("Residuo ||A*v - lambda*v||:    %.8f\n\n", residuo);
        imprime_comparacao_vetores(&A, &b0, &autovetor, lambda);

        escreve_saida_arquivo("resultado.txt", lambda, &autovetor, iteracoes, residuo);
        return 0;
    }

    /* Opcao 1 (padrao): roda a bateria de testes de validacao */
    teste1_matriz_diagonal();
    teste2_matriz_simetrica();
    teste3_matriz_tridiagonal();
    teste4_convergencia_lenta();
    teste5_limite_iteracoes();
    teste6_sensibilidade_vetor_inicial();
    teste7_variacao_tolerancia();
    teste8_autovalores_repetidos();
    return 0;
}