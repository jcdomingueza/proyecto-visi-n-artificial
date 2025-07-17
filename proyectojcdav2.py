# -*- coding: utf-8 -*-
"""
Created on Sun May 25 12:34:03 2025

@author: jdomi
"""
import cv2
import numpy as np
import os
import time
import csv
from chroma_module import combinar_texturas, aplicar_chroma_y_fusion, calibrar_chroma_key, aplicar_filtro_avanzado

# === CONFIGURACIÓN DE RUTAS ===
base_path = r"f:\VISION\TrabajoFinal\Proyecto_Final"
fondo_path = os.path.join(base_path, "fondo.jpg")
video_path = os.path.join(base_path, "video.mp4")
camuflaje_path = os.path.join(base_path, "camuflaje.jpg")
skull_path = os.path.join(base_path, "skull.png")
ruta_capturas = os.path.join(base_path, "capturas")
ruta_datos = os.path.join(base_path, "datos_filtros.csv")
os.makedirs(ruta_capturas, exist_ok=True)

# === CARGAR TEXTURA FUSIONADA Y PARÁMETROS DE CROMA ===
video_size = (300, 300)
textura_fusionada = combinar_texturas(camuflaje_path, skull_path, video_size)

# === MENÚ DE OPCIÓN PARA CALIBRAR CROMA ===
print("[MENÚ] ¿Deseas calibrar el chroma key ahora?")
print("1. Sí, calibrar con cámara en vivo")
print("2. No, usar valores por defecto (camiseta actual)")
opcion = input("Selecciona una opción [1/2]: ").strip()

if opcion == "1":
    lower_green, upper_green = calibrar_chroma_key()
else:
    lower_green = np.array([18, 108, 0])
    upper_green = np.array([90, 255, 216])
    print(f"[INFO] Usando valores predefinidos: lower={lower_green.tolist()}, upper={upper_green.tolist()}")

# === DEFINICIÓN DE POLÍGONOS POR EL USUARIO ===
fondo = cv2.imread(fondo_path)
clone = fondo.copy()
polygons = []
current_polygon = []

def seleccionar_poligonos():
    global current_polygon, polygons
    polygons = []
    current_polygon = []

    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                current_polygon.clear()

    cv2.namedWindow("Selecciona Zonas")
    cv2.setMouseCallback("Selecciona Zonas", draw_polygon)

    print("[INFO] Clic izquierdo para marcar puntos")
    print("[INFO] Clic derecho para cerrar un polígono")
    print("[INFO] ENTER para continuar, ESC para reiniciar")

    while True:
        temp = clone.copy()
        for poly in polygons:
            cv2.polylines(temp, [np.array(poly)], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in current_polygon:
            cv2.circle(temp, pt, 3, (0, 0, 255), -1)

        cv2.imshow("Selecciona Zonas", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(polygons) > 0:
            break
        elif key == 27:
            polygons.clear()
            current_polygon.clear()

    cv2.destroyWindow("Selecciona Zonas")

seleccionar_poligonos()

# === CONFIGURACIÓN DE FUENTES Y FILTROS ===
fuente_por_poligono = ["vid", "cam", "cam", "vid", "cam"]
filtro_por_poligono = [None, None, "chroma", "clahe", "clahe"]
rotacion_por_poligono = [None] * len(fuente_por_poligono)

cap_cam = cv2.VideoCapture(0)
cap_vid = cv2.VideoCapture(video_path)

if not cap_cam.isOpened() or not cap_vid.isOpened():
    print("❌ Error al abrir las fuentes de video.")
    exit()

cv2.namedWindow("Fusion con Chroma Key", cv2.WINDOW_NORMAL)
fps_list = []

try:
    while True:
        start_time = time.time()

        ret_cam, frame_cam = cap_cam.read()
        ret_vid, frame_vid = cap_vid.read()

        if not ret_cam or not ret_vid or frame_cam is None or frame_vid is None:
            break

        frame_cam = cv2.resize(frame_cam, video_size)
        frame_vid = cv2.resize(frame_vid, video_size)
        output = fondo.copy()

        for i, poly in enumerate(polygons):
            fuente = fuente_por_poligono[i]
            filtro = filtro_por_poligono[i]

            frame_src = frame_cam if fuente == "cam" else frame_vid
            if rotacion_por_poligono[i] is not None:
                frame_src = cv2.rotate(frame_src, rotacion_por_poligono[i])

            if filtro == "chroma":
                frame_src = aplicar_chroma_y_fusion(frame_src, textura_fusionada, lower_green, upper_green)
            elif filtro:
                frame_src = aplicar_filtro_avanzado(frame_src, filtro)

            src_pts = np.array([[0, 0], [video_size[0]-1, 0],
                                [video_size[0]-1, video_size[1]-1],
                                [0, video_size[1]-1]], dtype=np.float32)
            dst_pts = np.array(poly, dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame_src, M, (output.shape[1], output.shape[0]))

            mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(poly)], 255)
            for c in range(3):
                output[:, :, c] = np.where(mask == 255, warped[:, :, c], output[:, :, c])

        cv2.imshow("Fusión con Chroma Key", output)

        fps = 1.0 / (time.time() - start_time)
        fps_list.append(fps)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap_cam.release()
    cap_vid.release()
    cv2.destroyAllWindows()

    if fps_list:
        fps_avg = sum(fps_list) / len(fps_list)
        print(f"\n[📊 RESULTADO] Promedio de FPS durante ejecución: {fps_avg:.2f}")

    print("\n¿Deseas ejecutar el análisis de técnicas de filtrado y guardar las capturas?")
    print("1. Sí, generar comparativas por filtro")
    print("2. No, finalizar")
    analisis = input("Selecciona una opción [1/2]: ").strip()

    if analisis == "1":
        print("[🔬] Ejecutando pruebas de filtros y guardando resultados...")

        cap_cam = cv2.VideoCapture(0)
        cap_vid = cv2.VideoCapture(video_path)

        ret_cam, frame_cam = cap_cam.read()
        ret_vid, frame_vid = cap_vid.read()

        frame_cam = cv2.resize(frame_cam, video_size)
        frame_vid = cv2.resize(frame_vid, video_size)

        fuentes = {"cam": frame_cam, "vid": frame_vid}
        filtros = ["clahe", "hist_eq", "morph_open", "morph_close", "canny", "blur", "clahe_canny"]

        with open(ruta_datos, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Fuente", "Filtro", "TiempoProcesamiento_ms"])

            for fuente_nombre, fuente_frame in fuentes.items():
                for filtro in filtros:
                    path_antes = os.path.join(ruta_capturas, f"{fuente_nombre}_antes_{filtro}.png")
                    path_despues = os.path.join(ruta_capturas, f"{fuente_nombre}_despues_{filtro}.png")

                    cv2.imwrite(path_antes, fuente_frame)
                    start = time.perf_counter()
                    frame_filtrado = aplicar_filtro_avanzado(fuente_frame.copy(), filtro)
                    end = time.perf_counter()
                    duracion_ms = (end - start) * 1000
                    cv2.imwrite(path_despues, frame_filtrado)
                    writer.writerow([fuente_nombre, filtro, f"{duracion_ms:.6f}"])

        cap_cam.release()
        cap_vid.release()
        print("[✅] Comparativas e información de rendimiento guardadas en 'capturas/' y 'datos_filtros.csv'.")
