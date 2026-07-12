import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def copiar_vector_kernel(matriz_ptr, salida_ptr, n_columnas, str_filas, str_columnas, str_salida_filas, str_salida_columnas, BLOCK_SIZE: tl.constexpr):
    #comienzo que se repite
    fila = tl.program_id(axis=0)
    inicio=str_filas * fila
    columnas = tl.arange(0, BLOCK_SIZE)
    offset_entrada=inicio + (str_columnas*columnas)
    
    mascara = columnas < n_columnas 
    #cargamos los elementos con el puntero de entrada
    valores = tl.load (matriz_ptr + offset_entrada, mask=mascara, other = 0.0)
    inicio_salida= str_salida_filas*fila
    offset_salida=inicio_salida + (str_salida_columnas*columnas)
    #guardamos en el puntero de salida los elementos
    tl.store(salida_ptr + offset_salida, valores, mask = mascara)


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
    BLOCK_SIZE = triton.next_power_of_2(n_columnas)
    grid=(n_filas,)
    #no se puede hacer el stride en el codigo triton porque es un puntero por eso se hace antes+
    str_filas = matriz1.stride()[0]
    str_columnas = matriz1.stride()[1]
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    copiar_vector_kernel[grid](matriz1, salida, n_columnas, str_filas, str_columnas, str_salida_filas, str_salida_columnas,  BLOCK_SIZE=BLOCK_SIZE)
    return salida


def copiar_matriz_pre():
    matriz1=torch.tensor([[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],[9.0, 10.0, 11.0, 12.0]], device="cuda", dtype = torch.float32)
    matriz2=torch.tensor([[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],[9.0, 10.0, 11.0, 12.0]], device="cuda", dtype = torch.float32)
    print ("la matriz es: ", matriz1)
    print ("Su shape es: ", matriz1.shape)
    print ("Su num de elementos es: ", matriz1.numel())
    print ("Su stride es: ", matriz1.stride())
    print ("¿Es contigua? ", matriz1.is_contiguous())
    print ("la matriz es: ", matriz2)
    print ("Su shape es: ", matriz2.shape)
    print ("Su num de elementos es: ", matriz2.numel())
    print ("Su stride es: ", matriz2.stride())
    print ("¿Es contigua? ", matriz2.is_contiguous())
    if (matriz1.shape == matriz2.shape):
        resultado = sumar_matriz(matriz1, matriz2)
        print("matrices", matriz1, matriz2)
        print("suma de las matrices: ", resultado)
    else:
        print("tienen diferente shape")
    
    
    




def main():
    copiar_matriz_pre()


if __name__ == "__main__":
    main()
