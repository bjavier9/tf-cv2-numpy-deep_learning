import pygame
import random

# variable del juego
FPS = 60

# tamaño de la ventana
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
# distancia de la ventana
PADDLE_BUFFER = 10

# tamaño de la paleta
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

# tamaño de la bola
BALL_WIDTH = 10
BALL_HEIGHT = 10

# velocidad de la paleta y la bola
PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

# rgb color de la paleta y la bola
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# inicial la screen
# aqui usamos las variables definidas anteriormente para darle ancho y largo a la pantalla
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# esta funcion dibuja la bola y el rectangulo y recibe como parametro las posiciones donde queremos que se ubique


def drawball(bolaXpos, bolaYpos):
    ball = pygame.Rect(bolaXpos, bolaYpos, BALL_WIDTH, BALL_HEIGHT)
    # dibujamos la pelota
    pygame.draw.rect(screen, WHITE, ball)


def drawpalo1(paloYpos):
    palo1 = pygame.Rect(PADDLE_BUFFER, paloYpos,
                          PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, palo1)


def drawpalo2(palo2Ypos):
    palo2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, palo2Ypos, PADDLE_WIDTH, PADDLE_HEIGHT)
    #dibuja el palo, pasa por parametro el color 
    pygame.draw.rect(screen, WHITE, palo2)


def updateBall(palo1YPos, palo2YPos, bolaXpos, bolaYpos, bolaXDirection, bolaYDirection):
    # actualiza la posicion y la velocidad de la pelota
    bolaXpos = bolaXpos + bolaXDirection * BALL_X_SPEED
    bolaYpos = bolaYpos + bolaYDirection * BALL_Y_SPEED
    score = 0
    # checkea ssi la pelota choca
    # si la pelota choca el lado dizquierdo cambia la direccion
    if (bolaXpos <= PADDLE_BUFFER + PADDLE_WIDTH and bolaYpos + BALL_HEIGHT >= palo1YPos and bolaYpos - BALL_HEIGHT <= palo1YPos + PADDLE_HEIGHT):
      #cambia la direccion
      bolaXDirection = 1
    elif (bolaXpos <= 0):
        #puntaje negativo
        bolaXDirection = 1
        score = -1
        return[score, palo1YPos, palo2YPos, bolaXpos, bolaYpos, bolaXDirection, bolaYDirection]
    #verifica si llega al otro lado
    if (bolaXpos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and bolaYpos + BALL_HEIGHT >= palo2YPos and bolaYpos - BALL_HEIGHT <= palo2YPos + PADDLE_HEIGHT):
        #direccion de cambio
        bolaXDirection = -1
        #pasando
    elif(bolaXpos >= WINDOW_WIDTH - BALL_WIDTH):
        #puntaje positico
        bolaXDirection = -1
        score = 1
        return[score, palo1YPos, palo2YPos, bolaXpos, bolaYpos, bolaXDirection, bolaYDirection]
    #si llega al tope
    #mueve hacia abajo
    if(bolaYpos <= 0):
        bolaYpos = 0
        bolaYDirection = 1
        #si toca fondo que suba
    elif(bolaYpos >= WINDOW_HEIGHT - BALL_HEIGHT):
        bolaYpos = WINDOW_HEIGHT - BALL_HEIGHT
        #puntaje negativo
        bolaYDirection = -1
    return[score, palo1YPos, palo2YPos, bolaXpos, bolaYpos, bolaXDirection, bolaYDirection]

#actualizador de la posicion de la paleta
def updatepalo1(action, palo1YPos):
    if(action[1] == 1):
        palo1YPos = palo1YPos - PADDLE_SPEED

    if(action[2] == 1):
        palo1YPos = palo1YPos + PADDLE_SPEED

    if(palo1YPos < 0):
        palo1YPos = 0
    if(palo1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        palo1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return palo1YPos


def updatepalo2(palo2YPos, bolaYpos):
    #mueve hacia abajo si la pelota esta en la mitad superior
    if (palo2YPos + PADDLE_HEIGHT/2 < bolaYpos + BALL_HEIGHT/2):
        palo2YPos = palo2YPos + PADDLE_SPEED
    # move la pelota hacia arriba si esta en la mitad inferior
    if (palo2YPos + PADDLE_HEIGHT/2 > bolaYpos + BALL_HEIGHT/2):
        palo2YPos = palo2YPos - PADDLE_SPEED
    # no deja llegar a la parte superior
    if (palo2YPos < 0):
        palo2YPos = 0
    # no deja que toque fondo 
    if (palo2YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        palo2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return palo2YPos  
#clase del juego que lo inicializa 
class pongGame:
    def __init__(self):
        num = random.randint(0, 9)
        #score
        self.tally = 0  
        #posicion inicial del palo
        self.palo1YPos = WINDOW_HEIGHT /2 - PADDLE_HEIGHT /2
        self.palo2YPos = WINDOW_HEIGHT /2 - PADDLE_HEIGHT /2
        self.bolaXDirection = 1
        self.bolaYDirection = 1
        #empieza en este punto
        self.bolaXpos = WINDOW_HEIGHT/2 - BALL_WIDTH/2
        #decide mover la bola a una posicion random 
        if(0 < num < 3):
            self.bolaXDirection = 1
            self.bolaYDirection = 1
        if (3 <= num < 5):
            self.bolaXDirection = -1
            self.bolaYDirection = 1
        if (5 <= num < 8):
            self.bolaXDirection = 1
            self.bolaYDirection = -1
        if (8 <= num < 10):
            self.bolaXDirection = -1
            self.bolaYDirection = -1
        #nuevo numero randon
        num = random.randint(0,9)
        #donde empieza, y part
        self.bolaYpos = num*(WINDOW_HEIGHT - BALL_HEIGHT)/9

    def getPresentFrame(self):
        #repinta en la ventana principal
        pygame.event.pump()
        # el fondo debe ser negro 
        screen.fill(BLACK)
        drawpalo1(self.palo1YPos)
        drawpalo2(self.palo2YPos)
        # dibuja la bola 
        drawball(self.bolaXpos, self.bolaYpos)
        # optenemos los pixeles
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # actualiza la ventana
        pygame.display.flip()
        # retorna los datos de la pantalla
        return image_data

    def getNextFrame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        #actualiza el palo
        self.palo1YPos = updatepalo1(action, self.palo1YPos)
        drawpalo1(self.palo1YPos)
        #actualiza la malvada :v ia
        self.palo2YPos = updatepalo2(self.palo2YPos, self.bolaYpos)
        drawpalo2(self.palo2YPos)
        #actualiza la posicion de la bola
        [score, self.palo1YPos, self.palo2YPos, self.bolaXpos, self.bolaYpos, self.bolaXDirection, self.bolaYDirection] = updateBall(self.palo1YPos, self.palo2YPos, self.bolaXpos, self.bolaYpos, self.bolaXDirection, self.bolaYDirection)
        #draw the ball
        drawball(self.bolaXpos, self.bolaYpos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        self.tally = self.tally + score
        return [score, image_data]
