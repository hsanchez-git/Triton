import torch


def mostrar_matriz():
    matriz=torch.tensor([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]], dtype = torch.float32)
    matriz_lineal=matriz.view(-1) #estro se hace para obtener una vista unidimensional de la matriz
    print ("la matriz es: ", matriz)
    print ("Su shape es: ", matriz.shape)
    print ("Su num de elementos es: ", matriz.numel())
    print ("Su stride es: ", matriz.stride())
    print ("¿Es contigua? ", matriz.is_contiguous())
    matrizt=matriz.T
    print ("La traspuesta es: ", matrizt)
    print ("Su shape es: ", matrizt.shape)
    print ("Su stride es: ", matrizt.stride())
    print ("¿Es contigua? ", matrizt.is_contiguous())
    print (matriz.data_ptr() == matrizt.data_ptr())
    print("Vamos a buscar el 4, el 7 y el 9")
    offset = 0*matriz.stride()[0] + 3*matriz.stride()[1]
    print(matriz_lineal[offset])
    offset = 1*matriz.stride()[0] + 2*matriz.stride()[1]
    print(matriz_lineal[offset])
    offset = 2*matriz.stride()[0] + 0*matriz.stride()[1]
    print(matriz_lineal[offset])

#como el stride de la traspuesta cambia al hacer la operacion da el mismo resultado que en la normal y se puede buiscar en la unidimensional de la normal
    print("Vamos a buscar el 4, el 7 y el 9")
    offset = 3*matrizt.stride()[0] + 0*matrizt.stride()[1]
    print(matriz_lineal[offset])
    offset = 2*matrizt.stride()[0] + 1*matrizt.stride()[1]
    print(matriz_lineal[offset])
    offset = 0*matrizt.stride()[0] + 2*matrizt.stride()[1]
    print(matriz_lineal[offset])
    








def main():
    mostrar_matriz()


if __name__ == "__main__":
    main()
