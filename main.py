import cv2
import numpy as np

# Buka video
cap = cv2.VideoCapture('video.mp4')

# Koordinat garis (misalnya, garis horizontal di tengah layar)
line_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mengubah citra menjadi citra keabuan (grayscale)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Melakukan thresholding Otsu
    ret, imgh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Peroleh citra biner menggunakan thresholding
    # _, binary_frame = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # Mendeteksi kontur
    contours, hierarchy = cv2.findContours(imgh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Mendeteksi kontur pada citra biner
    # contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop melalui setiap kontur
    for contour in contours:
        # Ambil kotak pembatas untuk setiap kontur
        x, y, w, h = cv2.boundingRect(contour)
        # Hitung area kontur
        area = cv2.contourArea(contour)
        
        # Tandai objek jika area kontur berada dalam rentang tertentu
        if 500 < area < 10000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Periksa apakah kontur melewati garis
            if y + h >= line_y and y <= line_y:
                cv2.putText(frame, "Object Crossing Line", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            # Perbedaan antara mobil dan motor
            if area > 5000:  # Anggap sebagai mobil
                cv2.putText(frame, "Mobil", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:  # Anggap sebagai motor
                cv2.putText(frame, "Motor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

    # Tampilkan frame
    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
