"""
$ python adlookup.py "full-length video filename" "ad video-clip folder"

Debe entregar como salida nada.
Debe generar archivo <<detecciones.txt>>  con  todas  las  apariciones  de  los  comerciales
encontradas  en  el  video  de  televisión

El  archivo  de  salida  debe  tener  formato  de cuatro  columnas  separadas  por  un  tabulador  (\t),
cada  aparición  en  una  línea siguiendo el siguiente formato:

    video_television \t segundos_inicio \t segundos_largo \t video_comercial .
"""

import sys

from src import ft_extractor, similarity_tools

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise AttributeError("Script receives 2 parameters: \"full-length video filename\" and \"ad video-clip folder\"")
    video_filename = sys.argv[1]
    ads_foldername = sys.argv[2]

    video_features = ft_extractor.extract_features_from_video(video_filename)  # [frame, feature]
    ads_features = ft_extractor.extract_features_from_video_folder(ads_foldername)  # [clip_no, frame, feature]

    knn_list = similarity_tools.video_batch_knn(video_features, ads_features)  # [video_frame, i_nearest_neighbor]
    