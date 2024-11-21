import numpy as np

matrix_A = np.array([
    [1, 3, 7],
    [2, 4, 5],
    [3, 1, 2]
])

print("Matrix A in Originalform: ")
print(matrix_A)

def Factor_R_and_L(matrix_A): #Funktion die in A die Matrizen R und L speichert
    n = matrix_A.shape[0] #Anpassen der Dimension an A
    for k in range(n):
        for i in range(k+1, n):
            matrix_A[i, k] = matrix_A[i, k] / matrix_A[k, k]
            for j in range(k+1, n):
                matrix_A[i, j] = matrix_A[i, j] - matrix_A[i, k]*matrix_A[k, j]


def R_and_L_print(matrix_A): #Funktion die R und L ausgibt
    n = matrix_A.shape[0] #Anpassen der Dimension an A

    print("Obere Dreiecksmatrix R:")
    for i in range(n):
        for j in range(n):
            if i <= j:
                print(matrix_A[i, j], end=" ") #Werte aus A ab Haupdiagonale ausgeben
            else:
                print(0.00, end=" ") #restliche Werte 0
        print()


    print("\nUntere Dreiecksmatrix L:")

    for i in range(n):
        for j in range(n):
            if i > j:
                print(matrix_A[i, j], end=" ") #Werte unterhalb Hauptdiagonale ausgeben
            elif i == j:
                print(1.00, end=" ") #Werte auf Hauptdiagonale -> 1
            else:
                print(0.00, end=" ") #restliche Werte 0
        print()


def Check_R_L_A(matrix_A, original_A):
    n = matrix_A.shape[0] #Anpassen der Dimension an A
    matrix_L = np.eye(n) #Einheitsmatrix erzeugen
    matrix_R = np.zeros((n, n)) #0-Matrix erzeugen

    for i in range(n):
        for j in range(n):
            if i > j:
                matrix_L[i, j] = matrix_A[i, j]  #Werte unterhalb Hauptdiagonale in L speichern
            else:
                matrix_R[i, j] = matrix_A[i, j] #restliche Werte in R speichern

    check_A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                check_A[i, j] += matrix_L[i, k] * matrix_R[k, j] #Punktweise Matrixmultiplikation von L und R

    #Überprüfung ob die rekonstruierte Matrix check_A mit A übereinstimmt
    for i in range(n):
        for j in range(n):
            if abs(check_A[i, j] - original_A[i, j]) >= 1e-8:
                print("L*R ist ungleich A")
                return
    print("L*R ist gleich A")

#Funktionen aufrufen
original_A = matrix_A.copy() #Kopiert A um sie später zu überprüfen

Factor_R_and_L(matrix_A)

R_and_L_print(matrix_A)

Check_R_L_A(matrix_A, original_A)

#Ergebnisse
#Obere Dreiecksmatrix R:
#1 3 7
#0.0 -2 -9
#0.0 0.0 17
#Untere Dreiecksmatrix L:
#1.0 0.0 0.0
#2 1.0 0.0
#3 4 1.0
#L*R ist gleich A

