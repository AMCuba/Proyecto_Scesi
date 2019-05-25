import numpy as np
import scipy.io.wavfile as waves
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pandas as pd
import random
MAXIMO = 1

def estandarizar_longitudes(intensidad_maxima , intensidades):
    factor = MAXIMO / intensidad_maxima
    intensidades = (np.array(intensidades))*factor
    return intensidades


def datos(str1 , str2):
    frecuencias = []
    intensidades = []
    max = 0
    for i in range(32):
        muestra , sonido = waves.read(str1+str(i)+str2)
        frecuencia = np.fft.fftfreq(sonido[:,0].size)
        freq_in_hertz = abs(frecuencia * muestra)
        intensidad = np.fft.fft(sonido[:,0])
        intensidad = abs(intensidad[:7000])
        freq_in_hertz = freq_in_hertz[:7000]
        frecuencias.append(freq_in_hertz)
        intensidades.append(intensidad)
        if (intensidad.max() > max):
            max = intensidad.max()
    intensidades = estandarizar_longitudes(max , intensidades)
    return np.round(frecuencias , 6) , np.round(intensidades , 6)


def mostrar(list1 , list2 , color="blue"):
    plt.figure(figsize=(15,10))
    plt.plot(list1 , list2 , color=color)
    plt.show()


def obtener_maximos(fre , int , filtro):
    indices = []
    maximos = []
    for j in range(len(int)):
        max_actual = 0
        pos_max = 0
        indice = 0
        puntos = []
        inten = int[j]
        frec  = fre[j]
        index = []
        picos = 4
        for i in range((len(inten))):
            if(inten[i] > filtro):
                if(abs(i - indice) < 25):
                    max_actual = max(max_actual,inten[i])
                    indice = list(inten).index(max_actual)
                    pos_max = frec[indice]
                else:
                    if(picos > 0 and max_actual != 0):
                        puntos.append(round(max_actual , 6))
                        index.append(indice)
                        picos -=1
                    max_actual = inten[i]
                    pos_max = frec[i]
                    indice = i
        maximos.append(puntos)
        indices.append(index)
    return maximos , indices


def extender(arreglo , indices, rango):
    extendido = []
    indice = []
    for i in range(len(arreglo)):
        puntos = arreglo[i]
        index = indices[i]
        extendido.append(puntos)
        indice.append(index)
        for j in range(rango-1):
            puntos_ex = []
            indice_ex = []
            var_intensidad = random.uniform(0.9 , 1.1)
            for i in range(4):
                indice_ex.append(index[i])
                puntos_ex.append(round(puntos[i] * var_intensidad,6))
                #puntos_ex.append(round(puntos[i] , 6))
            extendido.append(puntos_ex)
            indice.append(indice_ex)
    return extendido,indice


def mostrar_cada_uno(frecuencia , intensidad , color='blue'):
    for i in range(len(frecuencia)):
        mostrar(frecuencia[i] , intensidad[i] , color)


def to_matrix(intensidades_instrumento , indices_maximos):
    int_ins = list(intensidades_instrumento)
    rangos = []
    for i in range(len(int_ins)):
        intensidad_nota = list(intensidades_instrumento[i])
        indices_nota = indices_maximos[i]
        rango = []
        for i in range(4):
            arreglo = intensidad_nota[ indices_nota[i] - 12:indices_nota[i] + 13]
            rango = rango + obtener_arrays(arreglo)
        rangos.append(rango)
    return rangos

def obtener_arrays(arreglo):
    arreglo = np.array(arreglo)*100
    arreglo = list(np.array(arreglo,dtype= 'uint8'))
    matrix = []
    for dato in arreglo:
        columna = unos_ceros(dato)
        matrix.append(columna)
    return matrix

def unos_ceros(num):
    arreglo = []
    if (num < 100):
        unos = list(np.ones(num, dtype='uint8')*255)
        ceros = list(np.zeros(100-num , dtype='uint8'))
        arreglo = unos + ceros
    else:
        arreglo = np.ones(100 , dtype='uint8')*255
    return arreglo

frec_guitarra, inten_guitarra  = datos("Guitarra2/","G.wav")
frec_bajo, inten_bajo  = datos("Bajo/","B.wav")
frec_piano, inten_piano = datos("Piano2/","P.wav")
frec_electrica, inten_electrica = datos("Electrica/","E.wav")

#se puede graficar la frecuencia

maximos_guitarra ,ig = obtener_maximos(frec_guitarra , inten_guitarra , 0.045)
maximos_bajo, ib = obtener_maximos(frec_bajo , inten_bajo , 0.01)
maximos_piano , ip = obtener_maximos(frec_piano , inten_piano , 0.045)
maximos_electrica, ie = obtener_maximos(frec_electrica , inten_electrica , 0.017)

#Datos para en entrenamiento de la red, con los puntos maximos de los 4 primeros picos de la grafica
dataset = maximos_guitarra + maximos_bajo + maximos_piano + maximos_electrica
labels = np.concatenate((np.zeros(22400),np.ones(22400) , np.ones(22400)*2 , np.ones(22400)*3) , axis=0)
test_labels = np.concatenate((np.zeros(1600),np.ones(1600), np.ones(1600)*2 , np.ones(1600)*3) , axis=0)
test , ind = extender(dataset, data_index , 50)
df = pd.DataFrame(dataset)
df.to_csv('Datos_en_csv/eval.csv' , header=False , index=False)
dataset1,a = extender(dataset2,data_index , 700)
df = pd.DataFrame(dataset1)
df.to_csv('Datos_en_csv/dataset.csv' , header=False , index=False)
df = pd.DataFrame(labels)
df.to_csv('Datos_en_csv/labels.csv', header=False , index=False)
df = pd.DataFrame(test_labels)
df.to_csv('Datos_en_csv/test_labels.csv', header=False , index=False)
df = pd.DataFrame(test)
df.to_csv('Datos_en_csv/test.csv',header=False , index=False)
