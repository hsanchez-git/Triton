import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def desplazar_vector_kernel(entrada_ptr, salida_ptr, desplazamiento, factor, n_elementos, BLOCK_SIZE: tl.constexpr):
    #comienzo que se repite
    program_id = tl.program_id(axis=0)
    inicio=BLOCK_SIZE * program_id
    offsets = inicio + tl.arange(0, BLOCK_SIZE)
    #se pueden hacer diferentes mascarqas segun las entradas diferentes que queramos
    mascara = offsets < n_elementos 
    #cargamos los elementos con el puntero de entrada
    valores = tl.load (entrada_ptr + offsets, mask=mascara, other = 0.0)
    resultado = valores *factor + desplazamiento
    #guardamos en el puntero de salida los elementos
    tl.store(salida_ptr + offsets, resultado, mask = mascara)


# Código Python normal que prepara y lanza el kernel
def desplazar_vector(entrada):
    #vemos cuantos elementos tiene el vector de entrada
    n_elementos = entrada.numel()
    #Creamos uin vector de salida como el de entrada pero vacio
    salida = torch.empty_like(entrada)
    desplazamiento = 3
    factor = 2
    BLOCK_SIZE = 4
    grid = (triton.cdiv(n_elementos, BLOCK_SIZE),)
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    desplazar_vector_kernel[grid](entrada, salida, desplazamiento, factor, n_elementos, BLOCK_SIZE=BLOCK_SIZE)
    return salida


def desplazar_vectores():
    entrada=torch.tensor([1.0, 2.0, 3.0, 4.0, 7.0, 3.5, 2.8, 1.23, 9.32, 6.49, 5.34], device="cuda", dtype=torch.float32)
    resultado = desplazar_vector(entrada)
    assert torch.equal(resultado, entrada*2+3)
    print("prueba del vector correcta", entrada, 3, 2, resultado)





def main():
    desplazar_vectores()


if __name__ == "__main__":
    main()
