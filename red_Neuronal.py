import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
data = np.genfromtxt('Datos_en_csv/dataset.csv' , delimiter=',')
labels = np.genfromtxt('Datos_en_csv/labels.csv' )
labels = keras.utils.to_categorical(labels , 4)
test = np.genfromtxt('Datos_en_csv/test.csv', delimiter=',')
test_labels = np.genfromtxt('Datos_en_csv/test_labels.csv')
test_labels = keras.utils.to_categorical(test_labels , 4)

modelo = tf.keras.models.Sequential([
    keras.layers.Dense(20,input_shape=(4,), activation=tf.nn.relu),
    keras.layers.Dense(20 , activation = tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4 , activation=tf.nn.softmax)
])
modelo.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
modelo.summary()
modelo.fit(data , labels , epochs = 5,shuffle = True)
modelo.evaluate(test , test_labels)
# modelo guardado: modelo.save('Modelos_Guardados/modelo_puntos98.h5')
# modelo cargado: modelo = keras.models.load_model('Modelos_Guardados/modelo_puntos.h5')


eval = np.genfromtxt('Datos_en_csv/eval.csv' , delimiter=',')
prediction = modelo.predict(eval)
"""for i in range (32):
    plt.axes((0 , 1 , 2 , 3))
    plt.bar(np.arange(4) , p[i])
    plt.ylim(0,1)
    plt.title("prediccion numero "+ str(i+1) )
    plt.xticks(np.arange(4) , ["guitarra" , "bajo" , "piano" , "GElectrica"])
    plt.show()
"""
