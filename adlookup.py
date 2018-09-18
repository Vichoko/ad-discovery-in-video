"""
$ python adlookup.py "full-length video filename" "ad video-clip folder"

Debe generar archivo <<detecciones.txt>>  con  todas  las  apariciones  de  los  comerciales
encontradas  en  el  video  de  televisión

El  archivo  de  salida  debe  tener  formato  de cuatro  columnas  separadas  por  un  tabulador  (\t),
cada  aparición  en  una  línea siguiendo el siguiente formato:

    video_television \t segundos_inicio \t segundos_largo \t video_comercial .
"""

import sys

from src import feature_extraction, similarity_tools, video_tools

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise AttributeError("Script receives 2 parameters: \"full-length video filename\" and \"ad video-clip folder\"")
    print("Welcome to the advertising clip detector!")
    video_filename = sys.argv[1]
    ads_foldername = sys.argv[2]

    print("info: Extracting (or loading cached) {} features".format(video_filename))
    video_features = feature_extraction.extract_features_from_video(video_filename)  # [frame, feature]

    print("info: Extracting (or loading cached) {} folder features".format(ads_foldername))
    ads_features = feature_extraction.extract_features_from_video_folder(ads_foldername)  # [clip_no, frame, feature]

    print("info: Starting (or loading cached) KNN")
    knn_list = video_tools.batch_knn(video_features, ads_features)  # [video_frame, i_nearest_neighbor]

    print("info: Detecting ads")
    video_tools.ads_detector(knn_list)
