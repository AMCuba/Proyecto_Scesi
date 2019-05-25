import numpy as np
import scipy.io.wavfile as waves
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import random
MAXIMO = 1
modelo = keras.models.load_model("Modelos_Guardados/modelo_puntos98.h5")
def estandarizar_longitudes(intensidad_maxima , intensidad):
    factor = MAXIMO / intensidad_maxima
    intensidad = (np.array(intensidad))*factor
    return intensidad

def leer_Archivo(archivo):
    max = 0
    muestra , sonido = waves.read(archivo)
    frecuencia = np.fft.fftfreq(sonido[:,0].size)
    freq_in_hertz = abs(frecuencia * muestra)
    intensidad = np.fft.fft(sonido[:,0])
    intensidad = abs(intensidad[:7000])
    freq_in_hertz = freq_in_hertz[:7000]
    if (intensidad.max() > max):
        max = intensidad.max()
    intensidad = estandarizar_longitudes(max , intensidad)
    freq_in_hertz =  np.round(freq_in_hertz , 6)
    intensidad =  np.round(intensidad , 6)
    max = obtener_Maximos(freq_in_hertz , intensidad)
    pred = modelo.predict(max)
    return pred[0]
def Mostrar(list1 , list2 , color="blue"):
    plt.figure(figsize=(15,10))
    plt.plot(list1 , list2 , color=color)
    plt.show()


def obtener_Maximos(frec , inten ):
    maximos = []
    max_actual = 0
    pos_max = 0
    indice = 0
    puntos = []
    picos = 4
    for i in range((len(inten))):
        if(inten[i] > 0.01):
            if(abs(i - indice) < 45):
                max_actual = max(max_actual , inten[i])
                indice = list(inten).index(max_actual)
                pos_max = frec[indice]
            else:
                if(picos > 0 and max_actual != 0):
                    puntos.append(round(max_actual , 6))
                    picos -=1
                max_actual = inten[i]
                pos_max = frec[i]
                indice = i
    maximos.append(puntos)
    return np.array(maximos)

def graficar_resultados(resultado):
    plt.figure(figsize=(4,2))
    plt.axes((0 , 1 , 2 , 3))
    plt.bar(np.arange(4) , resultado[0])
    plt.ylim(0,1)
    plt.title("prediccion")
    plt.xticks(np.arange(4) , ["guitarra" , "bajo" , "piano" , "GElectrica"])
    plt.plot()
    plt.show()

def prediccion(array):
    index = list(array).index(array.max())
    if index == 0:
        print("Es una Guitarra")
    elif index == 1:
        print("Es un Bajo")
    elif index == 2:
        print("Es un Piano")
    elif index == 3:
        print("Es una Guitarra Electrica")
    print("=========================================================================")
print("=========================================================================")
archivo = sys.argv[1]
print("El archivo de sonido: " + "'" + archivo + "'")
pred = leer_Archivo(archivo)
prediccion(pred)
