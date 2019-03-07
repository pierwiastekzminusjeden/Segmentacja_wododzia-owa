import matplotlib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from skimage import io
from skimage import filters
from skimage import morphology
from skimage import color

#wbudowany watershed
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


class Image:

    def __init__(self, name):
        ''' name - nazwa obrazu. Inicjalizuje obiekty przechowujące obraz ryginalny, 
        gradient obrazu, macierz po segmentacji wododziałowej implementowanej i wbudowanej.''' 
        
        self.image = io.imread(name)
        self.grad = None # Powstaje w wyniku wywołania self.prepareGrad
        self.label = None # Powstaje w wyniku wywołania self.watershed
        self.buildInLabel = None # Powstaje w wyniku wywołania self.buildInWatershed

    def showImages(self, images, cols = 1, titles = None):
        '''Wyświetlanie wielu obrazów w formie jednego przy użyciu biblioteki matplotlib.
        Funkcja przerobiona pod własne potrzeby.
        source: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
        '''
        assert((titles is None)or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            if n == 1 or n == 2:
                plt.gray()
            plt.imshow(image, cmap = "jet")
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()


    def buildInWatershed(self):
        ''' Funkcja uruchamiająca wbudowaną segmentację wododzialową. Macierz jest zapisywana do self.buildInLabel
        Zwraca otrzymany rezultat.
        '''

        distance = ndi.distance_transform_edt(self.image)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                    labels=self.image)
        markers = ndi.label(local_maxi)[0]
        self.buildInLabel = watershed(-distance, markers, mask = self.image)
        return self.buildInLabel

    def prepareGrad(self, diskSize):
        '''Odpowiada za przygotowanie gradientu obrazu. Zapisuje rezultat do self.grad. Zwraca otrzymany rezultat'''
        
        gray = color.rgb2gray(self.image)
        disk = morphology.disk(diskSize)
        imErosion = morphology.reconstruction(gray, gray, 'erosion', disk )
        imDilate = morphology.reconstruction(gray,imErosion, 'dilation', disk, )
        self.grad = morphology.dilation(imDilate,disk) - morphology.erosion(imErosion,disk)
        
        return self.grad

    def watershed(self, diskSize = 2):
        '''Implementacja algorytmu Vincet-Soille segmentacji wododzialowej'''
        
        self.prepareGrad(diskSize)
        pic = self.grad
        init = -1
        mask = -2
        wshed = 0
        fic = (-3,-3) 
        curlab = 0
        fifo = deque()

        #matrix init
        heigth, width = pic.shape

        print(pic)

        lab = np.zeros(pic.shape)
        dist = np.zeros(pic.shape)
        lab[:] = init
        grayScale = np.unique(pic)
        print(grayScale)
        for h in grayScale:
            for x in range(1, heigth-1):
                for y in range(1,width-1):
                    if pic[x,y] == h:
                        lab[x,y] = mask
                        #srodek - wystaracza do  dzialania algorytmu
                        if x != 0 and y != 0 and x != heigth - 1 and y != width - 1:
                            for i in range(-1,2):
                                for j in range(-1, 2):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #krawedzie - obsluga szczególnych przypadków, czyli pikseli będących brzegami obrazu
                        # gora
                        elif x == 0 and y != 0 and x != heigth - 1 and y != width - 1:
                            for i in range(0,2):
                                for j in range(-1, 2):
                                     if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #dol
                        elif x != 0 and y != 0 and x == heigth - 1 and y != width - 1:
                            for i in range(-1, 1):
                                for j in range(-1, 2):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #lewa
                        elif x != 0 and y == 0 and x != heigth - 1 and y != width - 1:
                            for i in range(-1, 2):
                                for j in range(0, 2):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #prawa
                        elif x != 0 and y != 0 and x != heigth - 1 and y == width - 1:
                            for i in range(-1, 2):
                                for j in range(-1, 1):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #wierzcholki
                        #gorny lewy 
                        elif x==0 and y == 0:
                            for i in range(0, 2):
                                for j in range(0, 2):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #gorny prawy
                        elif x==0 and y == width:
                            for i in range(0, 2):
                                for j in range(-1, 1):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #dolny lewy
                        elif x==heigth and y == 0:
                            for i in range(-1, 1):
                                for j in range(0, 2):
                                    if i != 0 and j != 0:

                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
                        #dolny prawy
                        elif x==heigth and y == width:
                            for i in range(-1, 1):
                                for j in range(-1, 1):
                                    if i != 0 and j != 0:
                                        if lab[x+i,y+j] == wshed or lab[x+i,y+j] > 0:
                                            dist[x,y] = 1
                                            fifo.append((x,y))
            curdist = 1
            fifo.append(fic)

            while True:
                p = fifo.popleft()
                if p == fic:
                    if not fifo:
                        break
                    else:
                        fifo.append(fic)
                        curdist+=1
                        p = fifo.popleft()
                #sprawdzanie sasiadow
                #środek - wystarcza do skutecznego działania algorytmu
                if p[0] != 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] != width - 1:
                    for i in range(-1,2):
                        for j in range(-1, 2):
                            if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:
                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #krawedzie - obsluga szczególnych przypadków, czyli pikseli będących brzegami obrazu
                #gorna krawedz
                elif p[0] == 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] != width - 1:
                    for i in range(0,2):
                        for j in range(-1, 2):
                            if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:
                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #dol
                elif p[0] != 0 and p[1] != 0 and p[0] == heigth - 1 and p[1] != width - 1:
                    for i in range(-1, 1):
                        for j in range(-1, 2):
                                if i != 0 and j != 0:
                                    if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                        if lab[ p[0]+i, p[1]+j ] > 0 :
                                            if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                                lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                            elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                                lab[p[0],p[1]] = wshed
                                        elif lab[p[0], p[1]] == mask:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                        dist[p[0]+i,p[1]+j] = curdist + 1
                                        fifo.append((p[0]+i, p[1] +j))
                #lewa
                elif p[0] != 0 and p[1] == 0 and p[0] != heigth - 1 and p[1] != width - 1:
                    for i in range(-1, 2):
                        for j in range(0, 2):
                           if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #prawa
                elif p[0] != 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] == width - 1:
                    for i in range(-1, 2):
                        for j in range(-1, 1):
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                 #wierzcholki
                #gorny lewy 
                elif p[0]==0 and p[1] == 0:
                    for i in range(0, 2):
                        for j in range(0, 2):
                           if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #gorny prawy
                elif p[0]==0 and p[1] == width:
                    for i in range(0, 2):
                        for j in range(-1, 1):
                           if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #dolny lewy
                elif p[0]==heigth and p[1] == 0:
                    for i in range(-1, 1):
                        for j in range(0, 2):
                           if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
                #dolny prawy
                elif p[0]==heigth and p[1] == width:
                    for i in range(-1, 1):
                        for j in range(-1, 1):
                           if i != 0 and j != 0:
                                if dist[ p[0]+i, p[1]+j ] < curdist and (lab[ p[0]+i, p[1]+j ] > 0 or lab[ p[0]+i, p[1]+j ] == wshed):
                                    if lab[ p[0]+i, p[1]+j ] > 0 :
                                        if lab[ p[0], p[1] ] == wshed or lab[ p[0], p[1]] == mask:
                                            lab[p[0], p[1]] = lab[p[0]+i, p[1]+j]
                                        elif lab[p[0], p[1]] != lab[p[0]+i, p[1]+j]:
                                            lab[p[0],p[1]] = wshed
                                    elif lab[p[0], p[1]] == mask:                                        lab[p[0],p[1]] = wshed
                                elif lab[p[0]+i, p[1]+j] == mask and dist[p[0]+i,p[1]+j] == 0:
                                    dist[p[0]+i,p[1]+j] = curdist + 1
                                    fifo.append((p[0]+i, p[1] +j))
            for x in range(heigth):
                for y in range(width):
                    if pic[x,y] == h:
                        dist[x,y] = 0
                        if lab[x,y] == mask:
                            curlab = curlab + 1
                            fifo.append((x,y))
                            lab[x,y] = curlab
                            while fifo:
                                p = fifo.popleft()
                                # srodek - wystarcza do skutecznego działania algorytmu
                                if p[0] != 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] != width - 1:
                                    for i in range(-1,2):
                                        for j in range(-1, 2):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                #krawedzie - obsluga szczególnych przypadków, czyli pikseli będących brzegami obrazu
                                # gora
                                elif p[0] == 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] != width - 1:
                                    for i in range(0,2):
                                        for j in range(-1, 2):
                                            if i != 0 and j != 0:    
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab

                                # dol
                                elif p[0] != 0 and p[1] != 0 and p[0] == heigth - 1 and p[1] != width - 1:
                                    for i in range(-1, 1):
                                        for j in range(-1, 2):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # lewa
                                elif p[0] != 0 and p[1] == 0 and p[0] != heigth - 1 and p[1] != width - 1:
                                    for i in range(-1, 2):
                                        for j in range(0, 2):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # prawa
                                elif p[0] != 0 and p[1] != 0 and p[0] != heigth - 1 and p[1] == width - 1:
                                    for i in range(-1, 2):
                                        for j in range(-1, 1):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # wierzcholki
                                # gora lewy

                                elif p[0]==0 and p[1] == 0:
                                    for i in range(0, 2):
                                        for j in range(0, 2):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # gora prawy
                                elif p[0]==0 and p[1] == width:
                                    for i in range(0, 2):
                                        for j in range(-1, 1):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # dol lewy
                                elif p[0]==heigth and p[1] == 0:
                                    for i in range(-1, 1):
                                        for j in range(0, 2):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
                                # dol prawy
                                elif p[0]==heigth and p[1] == width:
                                    for i in range(-1, 1):
                                        for j in range(-1, 1):
                                            if i != 0 and j != 0:
                                                if lab[p[0]+i,p[1]+j] == mask:
                                                    fifo.append((p[0]+i,p[1]+j))
                                                    lab[p[0]+i,p[1]+j] = curlab
        self.label = lab
        return lab

       
if __name__ == '__main__':
    pass