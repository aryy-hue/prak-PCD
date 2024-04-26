def canny_edge_detection(self, weak_threshold=100, strong_threshold=150):
        # Membaca gambar sebagai grayscale
        img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        
        # Step 1: Gaussian Blur
        # Melakukan Gaussian Blur pada gambar
        img_smoothed = cv2.GaussianBlur(img, (5, 5), 1.4)
        cv2.imshow("Gaussian Blur", img_smoothed)
        cv2.waitKey(0)
        
        # Step 2: Gradient Calculation
        # Menghitung gradien menggunakan operator Sobel
        gx = cv2.Sobel(img_smoothed, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_smoothed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude dan arah dari gradien
        magnitude = np.sqrt(gx**2 + gy**2)
        theta = np.arctan2(gy, gx)
        
        # Mengonversi sudut ke dalam derajat
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        
        # Step 3: Non-maximum Suppression
        # Supresi non-maksimum untuk mendapatkan tepi tipis
        Z = np.zeros((H, W), dtype=np.uint8)
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # Sudut 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = magnitude[i, j + 1]
                        r = magnitude[i, j - 1]
                    # Sudut 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = magnitude[i + 1, j - 1]
                        r = magnitude[i - 1, j + 1]
                    # Sudut 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = magnitude[i + 1, j]
                        r = magnitude[i - 1, j]
                    # Sudut 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = magnitude[i - 1, j - 1]
                        r = magnitude[i + 1, j + 1]
                    
                    if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                        Z[i, j] = magnitude[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        
        cv2.imshow("Non-maximum Suppression", Z)
        cv2.waitKey(0)
        
        # Step 4: Double Thresholding
        # Melakukan double thresholding
        img_N = np.zeros((H, W), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                if Z[i, j] > strong_threshold:
                    img_N[i, j] = 255
                elif Z[i, j] > weak_threshold:
                    img_N[i, j] = weak_threshold
        cv2.imshow("Double Thresholding", img_N)
        cv2.waitKey(0)
        
        # Step 5: Hysteresis Thresholding
        # Melakukan hysteresis thresholding untuk menghubungkan tepi-tepi
        img_H1 = np.zeros((H, W), dtype=np.uint8)
        strong_pixel = 255
        weak_pixel = weak_threshold
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_N[i, j] == weak_pixel:
                    if (img_N[i + 1, j - 1] == strong_pixel) or \
                       (img_N[i + 1, j] == strong_pixel) or \
                       (img_N[i + 1, j + 1] == strong_pixel) or \
                       (img_N[i, j - 1] == strong_pixel) or \
                       (img_N[i, j + 1] == strong_pixel) or \
                       (img_N[i - 1, j - 1] == strong_pixel) or \
                       (img_N[i - 1, j] == strong_pixel) or \
                       (img_N[i - 1, j + 1] == strong_pixel):
                        img_H1[i, j] = strong_pixel
                    else:
                        img_H1[i, j] = 0
                elif img_N[i, j] == strong_pixel:
                    img_H1[i, j] = strong_pixel 
        
        cv2.imshow("Hysteresis Thresholding", img_H1)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()