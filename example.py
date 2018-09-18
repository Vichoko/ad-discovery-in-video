import numpy

from src.configurations import DATA_FOLDER
from src.video_tools import mostrar_frame
import cv2

capture = cv2.VideoCapture(str(DATA_FOLDER / "pantene expert.mpg"))
sobel_threshold = 150
fps = 30
sps = 30  # samples per second

skipped_frames = 0
delta = fps/sps

while capture.grab():
    # sampling mechanism
    retval, frame = capture.retrieve()
    skipped_frames += 1
    if skipped_frames % delta != 0 or not retval:
        continue
    skipped_frames = 0
    # convertir a gris
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("VIDEO", frame_gris)
    # calcular filtro de sobel
    sobelX = cv2.Sobel(frame_gris, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobelY = cv2.Sobel(frame_gris, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    mostrar_frame("X", sobelX, valorAbsoluto=True, escalarMin0Max255=True)
    mostrar_frame("Y", sobelY, valorAbsoluto=True, escalarMin0Max255=True)
    # magnitud del gradiente
    magnitud = numpy.sqrt(numpy.square(sobelX) + numpy.square(sobelY))
    mostrar_frame("MAGNITUD GRADIENTE", magnitud, escalarMin0Max255=True)
    # aproximacion de la magnitud del gradiente
    magnitud_aprox = numpy.abs(sobelX) + numpy.abs(sobelY)
    mostrar_frame("APROX GRADIENTE", magnitud_aprox, escalarMin0Max255=True)
    # umbral sobre la magnitud del gradiente
    retval, bordes = cv2.threshold(magnitud, thresh=sobel_threshold, maxval=255, type=cv2.THRESH_BINARY)
    retval, bordes_aprox = cv2.threshold(magnitud_aprox, thresh=sobel_threshold, maxval=255, type=cv2.THRESH_BINARY)
    mostrar_frame("BORDES", bordes)
    mostrar_frame("BORDES APROX", bordes_aprox)
    # esperar por una tecla
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        break
capture.release()
cv2.destroyAllWindows()