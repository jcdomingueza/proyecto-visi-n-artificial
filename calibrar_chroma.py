import cv2
import numpy as np

def combinar_texturas(camuflaje_path, skull_path, size):
    textura1 = cv2.imread(camuflaje_path)
    textura2 = cv2.imread(skull_path)

    if textura1 is None or textura2 is None:
        raise FileNotFoundError("No se pudieron cargar las texturas.")

    textura1 = cv2.resize(textura1, size)
    textura2 = cv2.resize(textura2, size)

    fusionada = cv2.addWeighted(textura1, 0.5, textura2, 0.5, 0)
    return fusionada

def aplicar_chroma_y_fusion(frame, textura, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    fondo = cv2.bitwise_and(textura, textura, mask=mask)
    sujeto = cv2.bitwise_and(frame, frame, mask=mask_inv)
    resultado = cv2.add(fondo, sujeto)
    return resultado

def aplicar_filtro_avanzado(frame, filtro):
    if filtro == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif filtro == "hist_eq":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    elif filtro == "morph_open":
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    elif filtro == "morph_close":
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    elif filtro == "canny":
        edges = cv2.Canny(frame, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filtro == "blur":
        return cv2.GaussianBlur(frame, (7, 7), 0)
    elif filtro == "clahe_canny":
        clahe = aplicar_filtro_avanzado(frame, "clahe")
        return aplicar_filtro_avanzado(clahe, "canny")
    else:
        return frame

def calibrar_chroma_key():
    import cv2
    import numpy as np
    import json

    def nothing(x): pass

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[❌] No se pudo abrir la cámara. Usando valores por defecto.")
        return np.array([18, 108, 0]), np.array([90, 255, 216])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cv2.namedWindow("Calibrador Chroma", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrador Chroma", 1000, 500)

    # Sliders HSV
    cv2.createTrackbar("H Low", "Calibrador Chroma", 35, 179, nothing)
    cv2.createTrackbar("H High", "Calibrador Chroma", 85, 179, nothing)
    cv2.createTrackbar("S Low", "Calibrador Chroma", 40, 255, nothing)
    cv2.createTrackbar("S High", "Calibrador Chroma", 255, 255, nothing)
    cv2.createTrackbar("V Low", "Calibrador Chroma", 40, 255, nothing)
    cv2.createTrackbar("V High", "Calibrador Chroma", 255, 255, nothing)

    lower, upper = None, None
    frame_valido = False

    print("[INFO] Ajusta los sliders hasta que SOLO tu camiseta aparezca blanca a la derecha.")
    print("[INFO] Presiona ESC cuando termines para guardar los valores.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[⚠️] No se pudo capturar imagen. Verifica la cámara.")
            continue

        frame_valido = True
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hL = cv2.getTrackbarPos("H Low", "Calibrador Chroma")
        hH = cv2.getTrackbarPos("H High", "Calibrador Chroma")
        sL = cv2.getTrackbarPos("S Low", "Calibrador Chroma")
        sH = cv2.getTrackbarPos("S High", "Calibrador Chroma")
        vL = cv2.getTrackbarPos("V Low", "Calibrador Chroma")
        vH = cv2.getTrackbarPos("V High", "Calibrador Chroma")

        lower = np.array([hL, sL, vL])
        upper = np.array([hH, sH, vH])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        combined = np.hstack((frame, result))
        cv2.imshow("Calibrador Chroma", combined)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_valido:
        print("\n[✅ RESULTADO FINAL]")
        print("lower_green =", lower.tolist())
        print("upper_green =", upper.tolist())

        valores_hsv = {
            "lower": lower.tolist(),
            "upper": upper.tolist()
        }

        with open("hsv_config.json", "w") as archivo:
            json.dump(valores_hsv, archivo, indent=4)
            print("[💾] Valores HSV guardados en hsv_config.json")

        return lower, upper
    else:
        print("[⚠️] No se pudo calibrar. Usando valores por defecto.")
        return np.array([18, 108, 0]), np.array([90, 255, 216])