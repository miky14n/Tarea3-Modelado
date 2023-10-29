# Practica de Regresión Lineal - Futbol
## Pasos para realizar el entrenamiento
### Limpieza de datos
Por parte de la limpieza de datos, lo hicimos en el archivo llamado `1. Limpieza de Datos y Normalización - Regresión Lineal.ipynb` y los pasos para la limpieza fueron los siguientes:
- Reconocimiento de columnas innecesarias
  - Eliminación de columna `date`
- Reconocimiento de variables categóricas
- Reconocimiento de valores nulos o 0s
- Evaluación de correlaciones
  Se tomaron las primeras 10 columnas con correlaciones más fuertes.
- Normalización de datos

### Generación del modelo
Para generar el modelo, tomamos en cuenta los 10 parámetros necesarios para generar nuestra regresión. Teniendo en cuenta que al tener tanto parámetros tendremos que recurrir a una regresión múltiple, tendríamos 11 $\theta_s$ , donde $\theta_0$ seria nuestro intercepto de nuestra ecuación y las demás corresponderían a nuestros parámetros respectivamente. Para comprar el rendimiento de nuestro modelo, usamos la métrica de $R^2$ y lo contrastamos con la cantidad de $\theta_s$ usados para notar si la métrica mejora o no.

Todo el proceso anteriormente descrito esta más detallado en el notebook `Practica_3_-_Entrenamiento.ipynb`.
