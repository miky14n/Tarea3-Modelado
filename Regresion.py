import numpy as np
import matplotlib.pyplot as plt


class Regresion:
    # cosntructor
    def __init__(self):
        # variable caracteristica
        self.__X = None
        # variable objetivo
        self.__y = None
        # data para test
        self.__X_testing = None 
        self.__y_testing = None
        # parametros del modelo
        self.__theta = None
        # historial para el descenso
        self.__history = None

    # metodo para cargar data
    def fit(self, x, y):
        m, n = x.shape
        # aniadir unidad de sesgo X0, columna de 1s
        self.__X = np.append(np.ones((m, 1)), x.reshape(m, -1), axis=1)
        # convertimos el vector y en matriz de mx1
        self.__y = y.reshape(-1, 1)
        # inicializamos parametros en 0
        self.__theta = np.zeros(n + 1)

    def split_test(self, porcen):
        shuffled_indices = np.random.permutation(len(self.__X))
        test_set_size = int(len(self.__X) * porcen)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.__X_testing = self.__X[test_indices]
        self.__X = self.__X[train_indices]
        self.__y_testing = self.__y[test_indices]
        self.__y = self.__y[train_indices]
        
    @property
    def get_X(self):
        return self.__X

    @property
    def get_y(self):
        return self.__y
    
    @property
    def get_X_test(self):
        return self.__X_testing

    @property
    def get_y_test(self):
        return self.__y_testing

    def get_param(self):
        return self.__theta

    # metodo para inicializar parametros
    def inicializar(self, t=None):
        m, n = self.__X.shape
        if t is None:
            self.__theta = np.zeros(n)
        else:
            self.__theta = t

    # normalizar datos
    def normalizar(self):
        u = self.__X[:, 1:].mean(0)
        desv = self.__X[:, 1:].std(0)
        self.__X[:, 1:] = (self.__X[:, 1:] - u) / desv

    # devolver funcion error: costo
    def get_j(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        h = self.__X.dot(theta)
        error = h - self.__y
        j = 1 / (2 * m) * np.power(error, 2)
        return j.sum()

    # devolver gradiente
    def get_gradiente(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        h = self.__X.dot(theta)
        error = h - self.__y
        t = 1 / m * self.__X.T.dot(error)
        return t.flatten()

    # implementacion del algoritmo del descenso de gradiente
    def descenso_de_gradiente(self, alpha, epsilon=10e-6, itera=None):
        js = []
        theta = self.__theta
        i = 0
        while True:
            js.append(self.get_j(theta))
            theta = theta - alpha * self.get_gradiente(theta)
            # si el costo actual menos costo anterior es menor a epsilon... fin
            if abs(self.get_j(theta) - js[-1]) < epsilon:
                break
            # si no... si iter no es none, verificamos si llegamo a la iteracion iter
            elif itera is not None:
                if i >= itera:
                    break
            i = i + 1
        print("Numero de iteraciones: ", i)
        print("Costo: ", js[-1])
        
        self.__theta = theta
        self.__history = np.array(js)
        print("Parametros: ", self.__theta)

    # implementacion de la ecuacion normal
    def get_ecu_norm(self):
        ecu = (np.linalg.pinv((self.__X.T.dot(self.__X))).dot(self.__X.T)).dot(self.__y)
        self.__theta = ecu.flatten()

    def predecir(self, x):
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        res = x.dot(self.__theta.reshape(-1, 1))
        return res

    # Error medio cuadrado, mean square error (MSE)
    # útil para saber que tan cerca es la línea de ajuste de nuestra regresión a las observaciones
    def get_ECM(self, x=None, y=None):
        if x is None:
            x = self.__X_testing
        if y is None:
            y = self.__y_testing

        m = x.shape[0]
        h = x.dot(self.__theta.reshape(-1,1))
        error = h - y
        ecm = 1 / m * np.power(error, 2)
        return ecm.sum()
    
    # Raíz del error medio cuadrado, root mean square error (RMSE)
    # 
    def get_RECM(self, x=None, y=None):
        if x is None:
            x = self.__X_testing
        if y is None:
            y = self.__y_testing

        m = x.shape[0]
        h = x.dot(self.__theta.reshape(-1, 1))
        error = h - y
        ecm = 1 / m * np.power(error, 2)
        return np.sqrt(ecm.sum())
    
    # coeficiente de determinacion
    def get_r2(self, x=None, y=None):
        if x is None:
            x = self.__X_testing
        if y is None:
            y = self.__y_testing

        m = x.shape[0]
        y = y.reshape(-1,1)
        h = x.dot(self.__theta.reshape(-1,1))
        
        src = (h - y) ** 2
        sec = (h - y.mean())**2
        stc = sec.sum() + src.sum()
        r2 = sec.sum() / stc 
        return r2
    
    def graficar_historial(self):
        fig1 = plt.figure()
        plt.plot(range(self.__history.size), self.__history)
        plt.grid()
        plt.xlabel("iteraciones")
        plt.ylabel(r"$J(\theta)$")
        plt.title("Evolución de costo en el descenso de Gradente")
        plt.show()

    def graficar_data(self, model=False):
        fig2 = plt.figure()
        if self.__X.shape[1] > 2:
            ax = fig2.add_subplot(111, projection='3d')
            ax.scatter(self.__X[:, 1], self.__X[:, 2], self.__y)
            if model:
                # calculamos los valores del plano para los puntos x e y
                xx1 = np.linspace(self.__X[:, 1].min(), self.__X[:, 1].max(), 100)
                xx2 = np.linspace(self.__X[:, 2].min(), self.__X[:, 2].max(), 100)
                xx1, xx2 = np.meshgrid(xx1, xx2)
                x1 = (self.__theta[1] * xx1)
                x2 = (self.__theta[2] * xx2)
                z = (x1 + x2 + self.__theta[0])
                ax.plot_surface(xx1, xx2, z, alpha=0.4, cmap='hot')
        else:
            plt.scatter(self.__X[:, 1], self.__y)
            if model:
                x = np.linspace(self.__X[:, 1].min(), self.__X[:, 1].max(), 100)
                plt.plot(x, self.__theta[0] + self.__theta[1] * x, c="red")
            plt.grid()
        plt.show()

    # graficar J
    def graficar_j_3d(self):
        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')

        theta0 = np.linspace(-10, 10, 100)
        theta1 = np.linspace(-1, 4, 100)

        theta0, theta1 = np.meshgrid(theta0, theta1)
        j_cost = np.zeros((100, 100))

        for i in range(100):
            for j in range(100):
                t = np.array([theta0[i, j], theta1[i, j]])
                j_cost[i, j] = self.get_j(t)
        ax.scatter(self.__theta[0], self.__theta[1], self.get_j(self.__theta), c='red')

        ax.plot_surface(theta0, theta1, j_cost)
        plt.xlabel("theta0")
        plt.ylabel("theta1")
        plt.show()

    def graficar_j_curvas(self):
        fig4 = plt.figure()
        theta0 = np.linspace(-10, 10, 100)
        theta1 = np.linspace(-1, 4, 100)
        theta0, theta1 = np.meshgrid(theta0, theta1)
        j_costo = np.zeros((100, 100))

        for i in range(100):
            for j in range(100):
                t = np.array([theta0[i, j], theta1[i, j]])
                j_costo[i, j] = self.get_j(t)

        plt.contour(theta0, theta1, j_costo, np.logspace(-2, 3, 20))
        plt.scatter(self.__theta[0], self.__theta[1], c='red')
        plt.xlabel("theta0")
        plt.ylabel("theta1")
        plt.grid()
        plt.show()
