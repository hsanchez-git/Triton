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
def copiar_matriz(matriz):
    #vemos cuantos elementos tiene la matriz de entrada
    #Creamos una matriz de salida como la de entrada pero vacio
    salida = torch.empty(
    matriz.shape,
    device=matriz.device,
    dtype=matriz.dtype,
    )
    str_salida_filas = salida.stride()[0]
    str_salida_columnas = salida.stride()[1]
    
    n_filas=matriz.shape[0]
    n_columnas=matriz.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(n_columnas)
    grid=(n_filas,)
    #no se puede hacer el stride en el codigo triton porque es un puntero por eso se hace antes+
    str_filas = matriz.stride()[0]
    str_columnas = matriz.stride()[1]
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    copiar_vector_kernel[grid](matriz, salida, n_columnas, str_filas, str_columnas, str_salida_filas, str_salida_columnas,  BLOCK_SIZE=BLOCK_SIZE)
    return salida


def copiar_matriz_pre():
    matriz=torch.tensor([[1.0, 2.0, 3.0, 4.0],[5.0, 6.0, 7.0, 8.0],[9.0, 10.0, 11.0, 12.0]], device="cuda", dtype = torch.float32)
    print ("la matriz es: ", matriz)
    print ("Su shape es: ", matriz.shape)
    print ("Su num de elementos es: ", matriz.numel())
    print ("Su stride es: ", matriz.stride())
    print ("¿Es contigua? ", matriz.is_contiguous())
    resp= input ("¿Normal o traspuesta? (n/t)")
    if (resp=="n"):
        resultado = copiar_matriz(matriz)
        print("prueba de la amtriz", matriz, resultado)
        print("¿Tienen los mismos strides?", matriz.stride() == resultado.stride())
        son_iguales = torch.equal(matriz, resultado)
        print("¿La copia es correcta?", son_iguales)
    elif(resp=="t"):
        matrizt=matriz.T
        resultado = copiar_matriz(matrizt)
        print("prueba de la amtriz", matrizt, resultado)
        print("¿Tienen los mismos strides?", matrizt.stride() == resultado.stride())
        son_iguales = torch.equal(matrizt, resultado)
        print("¿La copia es correcta?", son_iguales)
    




def main():
    copiar_matriz_pre()


if __name__ == "__main__":
    main()
