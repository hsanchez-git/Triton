import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def sumar_vector_kernel(entrada1_ptr, entrada2_ptr, salida_ptr, n_elementos, BLOCK_SIZE: tl.constexpr):
    #comienzo que se repite
    program_id = tl.program_id(axis=0)
    inicio=BLOCK_SIZE * program_id
    offsets = inicio + tl.arange(0, BLOCK_SIZE)
    #se pueden hacer diferentes mascarqas segun las entradas diferentes que queramos
    mascara = offsets < n_elementos 
    #cargamos los elementos con el puntero de entrada
    valores1 = tl.load (entrada1_ptr + offsets, mask=mascara, other = 0.0)
    valores2 = tl.load (entrada2_ptr + offsets, mask=mascara, other = 0.0)
    suma=valores1 + valores2
    #guardamos en el puntero de salida los elementos
    tl.store(salida_ptr + offsets, suma, mask = mascara)


# Código Python normal que prepara y lanza el kernel
def sumar_vector(entrada1, entrada2):
    #vemos cuantos elementos tiene el vector de entrada
    n_elementos = entrada1.numel()
    #Creamos uin vector de salida como el de entrada pero vacio
    salida = torch.empty_like(entrada1)
    
    BLOCK_SIZE = 4
    grid = (triton.cdiv(n_elementos, BLOCK_SIZE),)
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    sumar_vector_kernel[grid](entrada1, entrada2, salida, n_elementos, BLOCK_SIZE=BLOCK_SIZE)
    return salida


def suma_vectores():
    entrada1=torch.tensor([1.0, 2.0, 3.0, 4.0, 7.0], device="cuda", dtype=torch.float32)
    entrada2=torch.tensor([5.6, 3.8, 4.9, 12.3, 5.3], device="cuda", dtype=torch.float32)
    resultado = sumar_vector(entrada1, entrada2)
    assert torch.equal(resultado, entrada1+entrada2)
    print("prueba del vector correcta", entrada1, entrada2, resultado)





def main():
    suma_vectores()


if __name__ == "__main__":
    main()
