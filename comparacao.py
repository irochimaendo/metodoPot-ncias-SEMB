import numpy as np


def autovalor_dominante(A, nome):
    """Calcula todos os autovalores de A e retorna o de maior modulo."""
    autovalores, autovetores = np.linalg.eig(A)

    idx_dominante = np.argmax(np.abs(autovalores))
    lambda_dominante = autovalores[idx_dominante].real
    v_dominante = autovetores[:, idx_dominante].real

    print(f"--- {nome} ---")
    print(f"Todos os autovalores: {np.round(autovalores.real, 6)}")
    print(f"Autovalor dominante:  {lambda_dominante:.6f}")
    print(f"Autovetor associado:  {np.round(v_dominante, 5)}")
    print()

    return lambda_dominante


if __name__ == "__main__":
    # Teste 1: matriz diagonal (esperado no C: 5.0)
    A1 = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, -3.0],
    ])
    autovalor_dominante(A1, "Teste 1 - Matriz diagonal")

    # Teste 2: matriz simetrica 2x2 (esperado no C: 3.0)
    A2 = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])
    autovalor_dominante(A2, "Teste 2 - Matriz simetrica 2x2")

    # Teste 3: matriz tridiagonal 3x3 (esperado no C: ~5.4142)
    A3 = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0],
    ])
    autovalor_dominante(A3, "Teste 3 - Matriz tridiagonal 3x3")

    # Teste 4: autovalores proximos (esperado no C: ~2.01)
    A4 = np.array([
        [2.01, 0.0],
        [0.0, 2.00],
    ])
    autovalor_dominante(A4, "Teste 4 - Autovalores proximos")

    # Teste 8: autovalores repetidos, multiplicidade 2 (esperado no C: 3.0)
    A8 = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    autovalor_dominante(A8, "Teste 8 - Autovalores repetidos")