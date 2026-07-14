import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def maximo_matriz_kernel(matriz_ptr, salida_ptr, n_columnas, str_filas, str_columnas, str_salida_filas, BLOCK_SIZE: tl.constexpr):
    #comienzo que se repite
    fila = tl.program_id(axis=0)
    columnas = tl.arange(0, BLOCK_SIZE)
    offset_entrada=str_filas * fila + (str_columnas*columnas)
    
    mascara = columnas < n_columnas 
    #cargamos los elementos con el puntero de entrada
    valores = tl.load (matriz_ptr + offset_entrada, mask=mascara, other=-float("inf"))
    
    maximo = tl.max(valores, axis=0)
    tl.device_print("En el programa:", fila)
    tl.device_print("El maximo es:", maximo)
    offset_salida=str_salida_filas*fila
    #guardamos en el puntero de salida los elementos, como jno es un vector lo pasamos sin mascara
    tl.store(salida_ptr + offset_salida, maximo)


# Código Python normal que prepara y lanza el kernel
def maximo_matriz(matriz):
    #comprobaciones para que esten en cuda, la dimension y kla forma de la mtriz
    if not matriz.is_cuda:
        raise ValueError("La matriz debe estar en la GPU.")

    if matriz.ndim != 2:
        raise ValueError("La entrada debe ser una matriz bidimensional.")

    if matriz.shape[1] == 0:
        raise ValueError("La matriz debe tener al menos una columna.")
    #vemos cuantos elementos tiene la matriz de entrada
    #Creamos una matriz de salida como la de entrada pero vacio

    n_filas=matriz.shape[0]
    n_columnas=matriz.shape[1]
    salida = torch.empty(
    n_filas,
    device=matriz.device,
    dtype=matriz.dtype,
    )
    str_salida_filas = salida.stride()[0]
    
    
    BLOCK_SIZE = triton.next_power_of_2(n_columnas)
    grid=(n_filas,)
    #no se puede hacer el stride en el codigo triton porque es un puntero por eso se hace antes+
    str_filas = matriz.stride()[0]
    str_columnas = matriz.stride()[1]
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    maximo_matriz_kernel[grid](matriz, salida, n_columnas, str_filas, str_columnas, str_salida_filas, BLOCK_SIZE=BLOCK_SIZE)
    return salida


def maximo_matriz_pre():
    matriz = torch.tensor(
    [
        [-5.0, -2.0, -8.0, -3.0, -6.0],
        [1.0, 7.0, 3.0, 4.0, 2.0],
        [10.0, 6.0, 20.0, 11.0, 9.0],
    ],
    device="cuda",
    )
    print ("la matriz es: ", matriz)
    print ("Su shape es: ", matriz.shape)
    print ("Su num de elementos es: ", matriz.numel())
    print ("Su stride es: ", matriz.stride())
    print ("¿Es contigua? ", matriz.is_contiguous())

    resultado = maximo_matriz(matriz)
    print("prueba de la matriz: \n", matriz, "\nel resultado es: \n", resultado)

def main():
    maximo_matriz_pre()


if __name__ == "__main__":
    main()
