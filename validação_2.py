import numpy as np

TOLERANCIA_PADRAO = 1e-6
MAX_ITER_PADRAO = 1000


def metodo_potencias(A, b0, max_iter=MAX_ITER_PADRAO, tolerancia=TOLERANCIA_PADRAO):

    v = np.array(b0, dtype=float)
    v = v / np.max(np.abs(v))

    lambda_atual = 0.0
    lambda_anterior = 0.0

    for k in range(max_iter):
        v_prox = A @ v
        idx_max = np.argmax(np.abs(v_prox))
        lambda_atual = v_prox[idx_max]  # mantem o sinal correto do autovalor

        norma = abs(lambda_atual)
        if norma > 1e-12:
            v_prox = v_prox / norma

        if k > 0 and abs(lambda_atual - lambda_anterior) < tolerancia:
            return lambda_atual, v_prox, k + 1

        lambda_anterior = lambda_atual
        v = v_prox

    return lambda_atual, v, max_iter


def residuo_autovalor(A, v, lam):
    """Calcula ||A*v - lambda*v|| (norma infinita) -- checagem geral
    de validacao, funciona mesmo sem saber o autovalor teorico."""
    return np.max(np.abs(A @ v - lam * v))


def valida_teste(nome, A, b0, lambda_esperado, max_iter=MAX_ITER_PADRAO,
                  tolerancia=TOLERANCIA_PADRAO):
    """
    Roda o metodo das potencias, calcula o residuo, compara o autovalor
    obtido com o valor esperado (analitico) e com numpy.linalg.eigvals,
    e imprime PASS/FAIL. Retorna True/False.
    """
    lam, v, iteracoes = metodo_potencias(A, b0, max_iter, tolerancia)
    residuo = residuo_autovalor(A, v, lam)

    autovalores_numpy = np.linalg.eigvals(A)
    lam_numpy = autovalores_numpy[np.argmax(np.abs(autovalores_numpy))].real

    ok_esperado = abs(lam - lambda_esperado) < 1e-2
    ok_numpy = abs(lam - lam_numpy) < 1e-2
    status = "PASS" if (ok_esperado and ok_numpy) else "FAIL"

    print(f"[{status}] {nome}")
    print(f"       lambda calculado = {lam:.6f} | esperado = {lambda_esperado:.6f} | numpy = {lam_numpy:.6f}")
    print(f"       iteracoes = {iteracoes} | residuo = {residuo:.8f}")
    print()

    return status == "PASS"


if __name__ == "__main__":
    resultados = []

    # Teste 1: matriz diagonal -> autovalor dominante = 5.0
    A1 = np.diag([2.0, 5.0, -3.0])
    b1 = [1.0, 1.0, 1.0]
    resultados.append(valida_teste("Teste 1 - Matriz diagonal", A1, b1, 5.0))

    # Teste 2: matriz simetrica 2x2 -> autovalores 3 e 1
    A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    b2 = [1.0, 0.0]
    resultados.append(valida_teste("Teste 2 - Matriz simetrica 2x2", A2, b2, 3.0))

    # Teste 3: matriz tridiagonal 3x3 -> autovalor dominante = 4 + sqrt(2)
    A3 = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
    b3 = [1.0, 0.0, 0.0]
    lambda3_esperado = 4.0 + np.sqrt(2.0)
    resultados.append(valida_teste("Teste 3 - Matriz tridiagonal 3x3", A3, b3, lambda3_esperado))

    # Teste 4: autovalores proximos (2.01 e 2.00)
    A4 = np.diag([2.01, 2.00])
    b4 = [1.0, 1.0]
    resultados.append(valida_teste("Teste 4 - Autovalores proximos", A4, b4, 2.01))

    # Teste 5: limite de iteracoes atingido (max_iter=1) -- espera resultado PARCIAL
    lam5, v5, it5 = metodo_potencias(A3, b3, max_iter=1)
    residuo5 = residuo_autovalor(A3, v5, lam5)
    ok_teste5 = (it5 == 1) and (residuo5 > 0.1)
    print(f"[{'PASS' if ok_teste5 else 'FAIL'}] Teste 5 - Limite de iteracoes (max_iter=1)")
    print(f"       lambda parcial = {lam5:.6f} (longe do valor final ~{lambda3_esperado:.4f})")
    print(f"       iteracoes = {it5} | residuo = {residuo5:.8f} (esperado: alto, sem convergencia)")
    print()
    resultados.append(ok_teste5)

    # Teste 6: sensibilidade ao vetor inicial (mesma matriz do Teste 3)
    print("[INFO] Teste 6 - Sensibilidade ao vetor inicial (matriz do Teste 3)")
    lambdas_teste6 = []
    for nome_vetor, vetor in [("[1,0,0]", [1.0, 0.0, 0.0]),
                              ("[0,1,0]", [0.0, 1.0, 0.0]),
                              ("[1,1,1]", [1.0, 1.0, 1.0])]:
        lam6, _, it6 = metodo_potencias(A3, vetor)
        lambdas_teste6.append(lam6)
        print(f"       vetor inicial {nome_vetor}: lambda = {lam6:.6f}, iteracoes = {it6}")
    ok_teste6 = (max(lambdas_teste6) - min(lambdas_teste6)) < 1e-3
    print(f"       [{'PASS' if ok_teste6 else 'FAIL'}] todos convergem para o mesmo autovalor\n")
    resultados.append(ok_teste6)

    # Teste 7: variacao da tolerancia (mesma matriz do Teste 3)
    print("[INFO] Teste 7 - Variacao da tolerancia (matriz do Teste 3)")
    residuos_teste7 = []
    for tol in [1e-2, 1e-4, 1e-6]:
        lam7, v7, it7 = metodo_potencias(A3, b3, tolerancia=tol)
        res7 = residuo_autovalor(A3, v7, lam7)
        residuos_teste7.append(res7)
        print(f"       tolerancia = {tol:.0e}: lambda = {lam7:.6f}, iteracoes = {it7}, residuo = {res7:.8f}")
    ok_teste7 = residuos_teste7[0] > residuos_teste7[1] > residuos_teste7[2]
    print(f"       [{'PASS' if ok_teste7 else 'FAIL'}] residuo diminui conforme a tolerancia aperta\n")
    resultados.append(ok_teste7)

    # Teste 8: autovalores repetidos (multiplicidade 2) -- caso-limite
    A8 = np.diag([3.0, 3.0, 1.0])
    b8 = [1.0, 2.0, 0.5]
    resultados.append(valida_teste("Teste 8 - Autovalores repetidos", A8, b8, 3.0))

    print("=" * 55)
    print(f"RESUMO: {sum(resultados)}/{len(resultados)} testes passaram")
    print("=" * 55)