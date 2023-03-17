from sklearn import datasets
import logistic_regression as lr
from matplotlib import pyplot as plt
import numpy as np

data_set = datasets.load_breast_cancer()
X = data_set['data']
X.shape
y = data_set['target']
y.shape

# Corrige el error de que los valores de X estén muy alejados
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

lr = lr.LogisticRegression()
iteraciones, errores = lr.fit(X, y)

plt.ion() # Modo interactivo activado
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Iteraciones')
ax.set_ylabel('Error')
ax.set_title('Evolución del Error')
line, = ax.plot([], [], 'b-')

for i in range(len(iteraciones)):
    line.set_xdata(iteraciones[:i+1])
    line.set_ydata(errores[:i+1])
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.1) # Intervalo de tiempo entre actualizaciones de la gráfica

plt.ioff() # Modo interactivo desactivado
plt.show()
