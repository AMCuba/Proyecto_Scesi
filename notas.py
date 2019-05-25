import scipy.io.wavfile as waves
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
i = 0
frecuencias = AudioSegment.from_file("Instrumentos/P.mp3")
frecuencias = frecuencias[69000:71500]
frecuencias.export(out_f="Piano/"+str(i)+"G.wav", format="wav")
muestra , sonido = waves.read("Instrumentos/muscle.wav")
frecuencia = np.fft.fftfreq(sonido[:,0].size)
freq_in_hertz = abs(frecuencia * muestra)
intensidad = np.fft.fft(sonido[:,0])
intensidad = abs(intensidad[:7000])
freq_in_hertz = freq_in_hertz[:7000]
plt.figure(figsize=(15,10))
plt.plot(freq_in_hertz , intensidad , color="blue")
plt.show()
frecuencias
