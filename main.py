# Autorzy: Krystian Molenda, Anna Cięciara
# Temat: Segmentacja wododzialowa Vincent-Soille - implementacja algorytmu
# Źródła: http://www.cs.rug.nl/roe/publications/parwshed.pdf 4.1.1 Algorithm 4.1
# Program zrealizowany w ramach projektu z przedmiotu Analiza Obrazów  


# Domyslnie - przy bezargumentowym wywołaniu skryptu algorytm testowany jest na obrazie coins.png
# Możliwe jest wywołanie skryptu z argumentem będącym ścieżką do obrazu który chcemy poddać segmentacji

from ImageClass import *
import time
import sys

#utworzenie obiektu klasy Image
if len(sys.argv) > 1:
    im = Image(sys.argv[1])
else:
    im = Image('coins.png')

#uruchomienie algorytmu segmentacji wododzialowej Vincent-Soille - pomiar czasu działania
start_time = time.time()
im.watershed(3)
impl_elapsed_time = time.time() - start_time
print("Czas wykonywania algorytmu implementowanego: ", impl_elapsed_time, "\n")

# #uruchomienie wbudowanej funkcji algorytmu segmentacji wododzialowej - pomiar czasu działania
start_time = time.time()
im.buildInWatershed()
build_in_elapsed_time = time.time() - start_time
print("Czas wykonywania algorytmu wbudowanego: ", build_in_elapsed_time)


# Wyświetlenie wyników 
images = [im.image,im.grad,im.label,color.label2rgb(im.buildInLabel,image = im.image)]
titles = ["Obraz wejściowy - szarość", "gradient", 
"efekt algorytmu implementowanego", "efekt algorytmu wbudowanego"]

im.showImages(images, 2,titles)

