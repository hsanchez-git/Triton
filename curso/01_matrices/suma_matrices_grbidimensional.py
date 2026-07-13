import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def sumar_matriz_kernel(matriz1_ptr, matriz2_ptr, salida_ptr, n_columnas, n_filas, str_filas1, str_columnas1, str_filas2, str_columnas2,
                        str_fila_salida, str_col_salida, BLOCK_X:tl.constexpr, BLOCK_Y: tl.constexpr):
    #comienzo que se repite
    pid_fila = tl.program_id(axis=0)
    pid_columna = tl.program_id(axis=1)

    filas_locales = tl.arange(0, BLOCK_X)
    columnas_locales = tl.arange(0, BLOCK_Y)

    filas_globales= (pid_fila*BLOCK_X) + filas_locales
    columnas_globales = (pid_columna*BLOCK_Y) + columnas_locales
    #es muy importante el [:, None] ya que pone filas en vertical y el [None, :] ya que pone columnas en hoprizontal
    mascara = (
    (filas_globales[:, None] < n_filas)
    & (columnas_globales[None, :] < n_columnas)
    )   

    
    offset_m1= filas_globales[:, None]*str_filas1 + columnas_globales[None, :]*str_columnas1
    offset_m2= filas_globales[:, None]*str_filas2 + columnas_globales[None, :]*str_columnas2

    valores1 = tl.load (matriz1_ptr + offset_m1, mask=mascara, other = 0.0)
    valores2 = tl.load (matriz2_ptr + offset_m2, mask=mascara, other = 0.0)

    suma=valores1+valores2

    offset_salida= filas_globales[:, None]*str_fila_salida + columnas_globales[None, :]*str_col_salida
    tl.store(salida_ptr + offset_salida, suma, mask = mascara)


# Código Python normal que prepara y lanza el kernel
def sumar_matriz(matriz1, matriz2):
    #vemos cuantos elementos tiene la matriz de entrada
    #Creamos una matriz de salida como la de entrada pero vacio
    salida = torch.empty(
    matriz1.shape,
    device=matriz1.device,
    dtype=matriz1.dtype,
    )
    str_salida_filas = salida.stride()[0]
    str_salida_columnas = salida.stride()[1]
    
    n_filas=matriz1.shape[0]
    n_columnas=matriz1.shape[1]
    BLOCK_X = 2
    BLOCK_Y = 4
    grid=(triton.cdiv(n_filas, BLOCK_X), triton.cdiv(n_columnas, BLOCK_Y))
    #no se puede hacer el stride en el codigo triton porque es un puntero por eso se hace antes
    str_filas1 = matriz1.stride()[0]
    str_columnas1 = matriz1.stride()[1]

    str_filas2 = matriz2.stride()[0]
    str_columnas2 = matriz2.stride()[1]
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    sumar_matriz_kernel[grid](matriz1, matriz2, salida, n_columnas, n_filas, str_filas1, str_columnas1, str_filas2, str_columnas2,
    str_salida_filas, str_salida_columnas,  BLOCK_X=BLOCK_X, BLOCK_Y=BLOCK_Y)
    return salida


def sumar_matriz_pre():
    matriz1 = torch.tensor([
    [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    ], device="cuda", dtype=torch.float32)

    matriz2 = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    ], device="cuda", dtype=torch.float32)

    print ("la matriz es: ", matriz1)
    print ("Su shape es: ", matriz1.shape)
    print ("Su num de elementos es: ", matriz1.numel())
    print ("Su stride es: ", matriz1.stride())
    print ("¿Es contigua? ", matriz1.is_contiguous())
    print ("------------------------------------------------------------------------------------------------")
    print ("la matriz es: ", matriz2)
    print ("Su shape es: ", matriz2.shape)
    print ("Su num de elementos es: ", matriz2.numel())
    print ("Su stride es: ", matriz2.stride())
    print ("¿Es contigua? ", matriz2.is_contiguous())
    print ("------------------------------------------------------------------------------------------------")
    assert matriz1.is_cuda and matriz2.is_cuda
    assert matriz1.ndim == 2 and matriz2.ndim == 2
    assert matriz1.shape == matriz2.shape
    assert matriz1.dtype == matriz2.dtype
    assert matriz1.device == matriz2.device
    resultado = sumar_matriz(matriz1, matriz2)
    print("matrices", matriz1, matriz2)
    print ("------------------------------------------------------------------------------------------------")
    print("suma de las matrices: ", resultado)
    esperado = matriz1 + matriz2
    torch.testing.assert_close(
    resultado,
    esperado,
    )




def main():
    sumar_matriz_pre()


if __name__ == "__main__":
    main()
