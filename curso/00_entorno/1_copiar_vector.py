import torch
import triton
import triton.language as tl


# Código que se ejecutará en la GPU
@triton.jit
def copiar_vector_kernel(entrada_ptr, salida_ptr, n_elementos, BLOCK_SIZE: tl.constexpr):
    #comienzo que se repite
    program_id = tl.program_id(axis=0)
    inicio=BLOCK_SIZE * program_id
    offsets = inicio + tl.arange(0, BLOCK_SIZE)
    mascara = offsets < n_elementos 
    #cargamos los elementos con el puntero de entrada
    valores = tl.load (entrada_ptr + offsets, mask=mascara, other = 0.0)
    #guardamos en el puntero de salida los elementos
    tl.store(salida_ptr + offsets, valores, mask = mascara)


# Código Python normal que prepara y lanza el kernel
def copiar_vector(entrada):
    #vemos cuantos elementos tiene el vector de entrada
    n_elementos = entrada.numel()
    #Creamos uin vector de salida como el de entrada pero vacio
    salida = torch.empty_like(entrada)
    
    BLOCK_SIZE = 4
    grid = (triton.cdiv(n_elementos, BLOCK_SIZE),)
    #hacemos la llamada y aunque de una especie de error es porque pylance no cuenta con triton
    copiar_vector_kernel[grid](entrada, salida, n_elementos, BLOCK_SIZE=BLOCK_SIZE)
    return salida


def prueba_vector_pequeno():
    entrada=torch.tensor([1.0, 2.0, 3.0, 4.0,], device="cuda", dtype=torch.float32)
    resultado = copiar_vector(entrada)
    assert torch.equal(resultado, entrada)
    print("prueba del vector correcta", entrada, resultado)


def prueba_tamano_no_multiplo():
    entrada=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], device="cuda", dtype=torch.float32)
    resultado = copiar_vector (entrada)
    assert torch.equal(resultado, entrada)
    print("misma estructura", entrada, resultado)


def main():
    prueba_vector_pequeno()
    prueba_tamano_no_multiplo()


if __name__ == "__main__":
    main()
