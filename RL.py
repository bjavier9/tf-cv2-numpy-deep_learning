import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque
# puntos claves
# El épsilon es el número decimal más pequeño que, sumado a 1, la computadora nos arroja un valor diferente de 1, es decir, 
# que no es redondeado.Representa la exactitud relativa de la aritmética del computador. La existencia del épsilon de la 
# máquina es una consecuencia de la precisión finita de la aritmética en coma flotante.
############################################################################################################################
# tensors: La lista o diccionario de tensores para enquear.
# batch_size: El nuevo tamaño de lote extraído de la cola.
# num_threads: La cantidad de hilos en secuencia tensors. El procesamiento por lotes no será definimos parametros.
# Las canalizaciones de entrada basadas en Colas no son compatibles cuando la ejecución ansiosa está habilitada. 
# Utilice la tf.dataAPI para introducir datos en ejecución ansiosa.
# Digamos que quiere hacer reconocimiento de dígitos (MNIST) y ha definido su arquitectura de la red (CNN). 
# Ahora, puede alimentar las imágenes de los datos de entrenamiento uno por uno a la red, obtener predicción (también llamada inferencia), 
# calcular la pérdida, calcular el gradiente y luego actualizar los parámetros ( pesos y sesgos ) y continuar con la siguiente imagen. Esto se llama aprendizaje en línea .
####################################################################################################################
# informacion en : https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
# tf.resshape
# Cambia la forma de un tensor.
# Teniendo en cuenta tensor, esta operación devuelve un tensor que tiene los mismos valores que tensorcon la forma shape.
# Si un componente de shapees el valor especial -1, el tamaño de esa dimensión se calcula de modo que el tamaño total permanezca constante. 
# En particular, a shape de [-1]aplana en 1-D. Como máximo, un componente de shapepuede ser -1.
# Si shapees 1-D o superior, entonces la operación devuelve un tensor con forma shapellena con los valores de tensor. 
# En este caso, el número de elementos implicados shapedebe ser el mismo que el número de elementos en tensor.
#####################################################################################################################
#tf.placeholder
# Inserta un marcador de posición para un tensor que siempre se alimentará.
# Importante : este tensor generará un error si se evalúa. 
# Su valor debe ser alimentado con el feed_dictargumento opcional a Session.run(), Tensor.eval()o Operation.run().
######################################################################################################################
# El Variable()constructor requiere un valor inicial para la variable, que puede ser Tensorde cualquier tipo y forma.
# El valor inicial define el tipo y la forma de la variable. Después de la construcción, el tipo y la forma de la variable son fijos. 
# El valor se puede cambiar usando uno de los métodos de asignación.
# Si desea cambiar la forma de una variable más adelante, debe usar un assignOp con validate_shape=False.
# Al igual que cualquiera Tensor, las variables creadas con Variable()pueden usarse como entradas para otras Ops en el gráfico.
#  Además, todos los operadores sobrecargados para la Tensorclase se transfieren a las variables, 
# por lo que también puede agregar nodos al gráfico simplemente haciendo aritmética en las variables.
#####################################################################################################################
# # tf.nn.relu
# Args:
# features: Una Tensor. Debe ser uno de los siguientes tipos: 
# float32, float64, int32, uint8, int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
# name: Un nombre para la operación (opcional).
# Devoluciones:
# Una Tensor. Tiene el mismo tipo que features.
####################################################################################################################
#tf.matmul
# Multiplica matriz apor matriz b, produciendo a* b.
# Las entradas deben, después de cualquier transposición, ser tensores de rango> = 2 donde las 2 dimensiones internas especifican 
# argumentos válidos de multiplicación de matrices, y cualquier dimensión externa adicional coincide.
# Ambas matrices deben ser del mismo tipo. Los tipos soportados son: float16, float32, float64, int32, complex64, complex128.
# Cualquiera de las matrices puede transponerse o unirse (conjugarse y transponerse) sobre la marcha configurando una de las banderas correspondientes True. Estos son False por defecto.
# Si una o ambas de las matrices contienen una gran cantidad de ceros, 
# un algoritmo de multiplicación más eficiente puede ser utilizado por el establecimiento de la correspondiente a_is_sparseo b_is_sparsela bandera a True. 
# Estos son Falsepor defecto. Esta optimización solo está disponible para matrices simples (tensores de rango 2) con tipos de datos bfloat16o float32
####################################################################################################################
# # sess
# Una clase para ejecutar operaciones de TensorFlow.
# Un Sessionobjeto encapsula el entorno en el que Operation se ejecutan los Tensorobjetos y se evalúan los objetos.
# Una sesión puede ser propietaria de los recursos, tales como tf.Variable, tf.QueueBase, y tf.ReaderBase. Es importante liberar estos recursos cuando ya no sean necesarios. 
# Para hacerlo, invoque el tf.Session.closemétodo en la sesión o use la sesión como administrador de contexto.
###################################################################################################################
# tf.square
# Calcula el cuadrado de x elemento-sabio.
# y = x * x = x^2
# Args:
# x: A Tensoro SparseTensor. Debe ser uno de los siguientes tipos: half, float32, float64, int32, int64, complex64, complex128.
# name: Un nombre para la operación (opcional).
# Devoluciones:
# A Tensoro SparseTensor. Tiene el mismo tipo que x.



#hyper parametros
ACTIONS = 3 #arriba,abajo, quieto
# tasa de aprendisaje
GAMMA = 0.99
# definimos epsilon
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# cuantos frame de epsilon
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 50000
# tamaño de batch
BATCH = 100

# diagrama de flujo de tensores


def createGraph():
    # primera capa convucional-definimos el tamaño del flujo de variables
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    # la siguiente capa
    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    # capa 3
    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))
    # capa 4
    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    # ultima capa
    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # recibimos los datos
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Computa la función de activación de la unidad lineal rectificada en una convolución 2-D con entrada de 4-D y tensores de filtr
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[
                       1, 4, 4, 1], padding="VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[
                       1, 2, 2, 1], padding="VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[
                       1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4 + b_fc4))

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

# deep learning. alimenta los datos de píxeles para graficar la sesión


def trainGraph(inp, out, sess):

    # para calcular el argmax, multiplicamos el resultado pronosticado con un vector con un valor 1 y rest como 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])  # ground verdadero

    # action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1)
    # función de costo que reduciremos a través de la retropropagación
    cost = tf.reduce_mean(tf.square(action - gt))
    # función de optimización para reducir nuestra función de minimizar su cost
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize el juego
    game = pong.pongGame()

    # crear una cola para la reproducción de experiencia para almacenar políticas
    D = deque()

    # marco inicial
    frame = game.getPresentFrame()
    # convertir rgb en escala de grises para procesar
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # colores binarios, negro o blanco
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # marcos de pila, ese es nuestro tensor de entrada
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # entrenador
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON

    # tiempo de entrenamiento
    while(1):
        # tensor de salida
        out_t = out.eval(feed_dict={inp: [inp_t]})[0]
        # function argmax
        argmax_t = np.zeros([ACTIONS])

        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward si el tensor colecta puntaje positivo
        reward_t, frame = game.getNextFrame(argmax_t)
        # obtener datos de pixeles
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # nuevo tensor de entrada
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # añadir nuestro tensor de entrada, tensor de argmax, reward y tensor de entrada actualizado a la vista de las experiencias
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # si nos quedamos sin memoria hace un sitio
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # iteracion de entrenamiento
        if t > OBSERVE:

            # obtiene los valores de nuestra memoria de repeticion
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            # agrega valores a los batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # entrenamiento
            train_step.run(feed_dict={
                           gt: gt_batch,
                           argmax: argmax_batch,
                           inp: inp_batch
                           })

        # actualiza el tensor al siguiente fotograma
        inp_t = inp_t1
        t = t+1

        # imprime nuestro dónde estamos después de guardar dónde estamos
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex,
              "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    # crea sesion
    sess = tf.InteractiveSession()
    # capa de entrada y capa de salida
    inp, out = createGraph()
    # entrena nuestro graficos en entrada y salida con las variables de la secion
    trainGraph(inp, out, sess)


if __name__ == "__main__":
    main()
