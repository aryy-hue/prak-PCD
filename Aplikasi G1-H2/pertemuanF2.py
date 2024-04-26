import sys
import cv2
import numpy as np
from PyQt5 import QtCore , QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import math
import matplotlib.pyplot as plt
class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui',self)
        self.Image = None
        self.btn_img.clicked.connect(self.fungsi)
        self.btn_gray.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binerImage)
        
        # HISTOGRAM
        self.actionHistogram_Grayscale.triggered.connect(self.histogramGrayscale)
        self.actionHistogram_RGB.triggered.connect(self.RGBhistogram)
        self.actionHistogram_Equalization.triggered.connect(self.histogramEqu)

        # OPERASI GEOMETRI
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotation90deg)
        self.action180_Derajat.triggered.connect(self.rotation180deg)
        self.action45_Derajat.triggered.connect(self.rotation45deg)
        self.action_45_Derajat.triggered.connect(self.rotationMin45deg)
        self.action_90_Derajat.triggered.connect(self.rotationMin90deg)
        
        self.actionZoom_in.triggered.connect(self.zoomIn2)
        self.actionZoom_in_3x.triggered.connect(self.zoomIn3)
        self.actionZoom_in_4x.triggered.connect(self.zoomIn4)
        self.actionZoom_out_1_2.triggered.connect(self.zoomOut0_5)
        self.actionZoom_out_1_4.triggered.connect(self.zoomOut0_25)
        self.actionZoom_out_3_4.triggered.connect(self.zoomOut0_75)
        
        self.actionCrop.triggered.connect(self.cropImg)
        
        # OPERASI ARITMATIKA
        self.actionTambah_dan_kurang.triggered.connect(self.aritmatika1)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika2)
        
        # OPERASI BOOLEAN
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)
        
        # OPERASI KONVOLUSI
        self.actionKonvulusi_2D.triggered.connect(self.konvolusi2D)
        self.actionMean_Filter.triggered.connect(self.meanFilter)
        self.actionGaussian_Filter.triggered.connect(self.gaussianFilter)
        self.actionImage_Sharpening_Laplacian.triggered.connect(self.imageSharpeningLaplacian)
        self.actionImage_Sharpening.triggered.connect(self.imageSharpening)
        self.actionMedian_Filter.triggered.connect(self.medianFilter)
        self.actionMax_Filter.triggered.connect(self.maxFiltering)
        self.actionMin_Filter.triggered.connect(self.minFiltering)
        
        # OPERASI TRANSFORMASI FOURIER 
        self.actionSmoothing_Image.triggered.connect(self.lpf)
        self.actionDeteksi_Tepi.triggered.connect(self.hpf)
        
        #OPERASI DETEKSI TEPI
        self.actionSobel.triggered.connect(self.sobel_edge_detection)
        self.actionRoberts.triggered.connect(self.roberts_edge_detection)
        self.actionPrewitt.triggered.connect(self.prewitt_edge_detection)
        
        self.actionCanny_Edge_Detection.triggered.connect(self.canny_edge_detection)
        
        
        # SLIDER
        self.brightnessSlider.valueChanged.connect(self.brightnessChanged)
        self.contrastSlider.valueChanged.connect(self.contrastChanged)
        self.brightnessValue = 80
        self.contrastValue = 5.7
        
        ## Morfologi
        self.actionDilasi.triggered.connect(self.dilasi)
        self.actionErosi.triggered.connect(self.erosi)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)

        ## Segmenatasi Citra
        #Global TRESHOLD
        self.actionBinary.triggered.connect(self.binary)
        self.actionBinary_Invers.triggered.connect(self.binaryinvers)
        self.actionTrunc.triggered.connect(self.turnc)
        self.actionTo_Zero.triggered.connect(self.tozero)
        self.actionTo_Zero_Invers.triggered.connect(self.tozeroinvers)

        ## local Treshold
        self.actionMean.triggered.connect(self.meanthres)
        self.actionGaussian.triggered.connect(self.gaussianthres)
        self.actionOtsu.triggered.connect(self.otsuthres)

        ###Contour
        self.actionContour.triggered.connect(self.contour)
        

        
    def fungsi(self):
        self.Image = cv2.imread('g.jpg')
        self.displayImage(self.label)
        
    def grayscale(self):
        H,W = self.Image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.Image[i,j,0] + 
                                    0.587 * self.Image[i,j,1] + 
                                    0.114 * self.Image[i,j,2],0,255)
        self.Image = gray
        self.displayImage(self.citra_label)
    
    def brightnessChanged(self, value):
        self.brightnessValue = value
        self.brightness()
        self.displayImage(self.label)

    def contrastChanged(self, value):
        self.contrastValue = value
        self.contrast()
        
        self.displayImage(self.label)
    def brightness(self):
        # Agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image,  cv2.COLOR_BGR2GRAY)
        except:
            pass
            
        H,W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i,j),b)
        self.displayImage(self.label)
     
    def contrast(self):
        # Agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image,  cv2.COLOR_BGR2GRAY)
        except:
            pass
            
        H,W = self.Image.shape[:2]
        contrast = 5.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i,j),b)
        self.displayImage(self.label)
    
    def contrastStreching(self):
        # Agar menghindari error ketika melewati proses grayscale citra
        try:
            self.Image = cv2.cvtColor(self.Image,  cv2.COLOR_BGR2GRAY)
        except:
            pass
            
        H,W = self.Image.shape[:2]
        
        minV = np.min(self.Image)
        maxv = np.max(self.Image)
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = float(a - minV)/(maxv - minV) * 255

                self.Image.itemset((i,j),b)
        self.displayImage(self.label)
    
    def negativeImage(self):
        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)
                self.Image.itemset((i, j), b)
        self.displayImage(self.label)
    
    def binerImage(self):
        # Mengonversi citra ke citra keabuan jika belum dalam keabuan
        if len(self.Image.shape) > 2:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        
        # Melakukan iterasi melalui setiap pixel citra
        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                pixel_value = self.Image.item(i, j)
                if pixel_value == 180:
                    self.Image.itemset((i, j), 0)  # Set pixel menjadi 0
                elif pixel_value < 180:
                    self.Image.itemset((i, j), 1)  # Set pixel menjadi 1
                else:
                    self.Image.itemset((i, j), 255)  # Set pixel menjadi 255
        
        # Menampilkan citra biner
        self.displayImage(self.label)

    def histogramGrayscale(self):
        H,W = self.Image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.Image[i,j,0] + 
                                    0.587 * self.Image[i,j,1] + 
                                    0.114 * self.Image[i,j,2],0,255)
        self.Image = gray
        self.displayImage(self.citra_label)
        plt.hist(self.Image.ravel(),255,[0,255])
        plt.show()
        
    def RGBhistogram(self):
        color = ('b', 'g', 'r') # tuple adalah koleksi yang disimpan dan tidak dapat diubah
        for i,col in enumerate(color): # membuat perulangan berdasarkan warna
            histr = cv2.calcHist([self.Image],[i],None,[256],[0,256]) # menghitung histogram dari sekumpulan koleksi / array
            plt.plot(histr,color = col) # plot histogram
            plt.xlim([0,256]) # mengatur batas sumbu x
        self.displayImage(self.citra_label)
        plt.show()
        
    def translasi(self):
        h,w = self.Image.shape[:2]
        quarter_h , quarter_w = h/4, w/4
        T = np.float32([[1,0,quarter_w],[0,1,quarter_h]]) 
        img = cv2.warpAffine(self.Image,T,(w,h))
        self.Image = img
        self.displayImage(self.citra_label)
    
    # ROTASI 90, 180, 45, -45, -90
    def rotation90deg(self):
        self.rotasi(90)
    def rotation180deg(self):
        self.rotasi(180)
    def rotation45deg(self):
        self.rotasi(45)
    def rotationMin45deg(self):
        self.rotasi(-45)
    def rotationMin90deg(self):
        self.rotasi(-90)
    def rotasi(self,degree):
        h,w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w/2,h/2),degree,1)
        cos = np.abs(rotationMatrix[0,0])
        sin = np.abs(rotationMatrix[0,1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * sin) + (w * cos))
        
        rotationMatrix[0,2] += (nW / 2) - w/2
        rotationMatrix[1,2] += (nH / 2) - h/2
        
        rotate_img = cv2.warpAffine(self.Image,rotationMatrix,(w,h))
        self.Image = rotate_img
        self.displayImage(self.citra_label)
    
    # ZOOM IN AND ZOOM OUT    
    def zoomIn2(self):
        skala = 2 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 2x' , resize_img)
        cv2.waitKey()
    def zoomIn3(self):
        skala = 3 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 3x' , resize_img)
        cv2.waitKey()
    def zoomIn4(self):
        skala = 4 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 4x' , resize_img)
        cv2.waitKey()
    def zoomOut0_5(self):
        skala = 0.5 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 4x' , resize_img)
        cv2.waitKey()    
    def zoomOut0_25(self):
        skala = 0.25 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 4x' , resize_img)
        cv2.waitKey()
        
    def zoomOut0_75(self):
        skala = 0.75 
        resize_img = cv2.resize(self.Image, None , fx=skala, fy=skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)   
        cv2.imshow('Zoomed in 4x' , resize_img)
        cv2.waitKey()
        
    def cropImg(self):
        h , w = self.Image.shape[:2]
        # dimulai dari pixel 0
        start_row =0
        start_col =0
        
        # sampai pixel ke 250
        end_row = 250
        end_col = 250
        
        cropped_img = self.Image[start_row:end_row, start_col:end_col]
        cv2.imshow('Cropped Image', cropped_img)
        cv2.waitKey(0)
        
    # ARITMATIKA    
    def aritmatika1(self):
        image1 = cv2.imread('lilypichu.jpg' , 0)   
        image2 = cv2.imread('toothless.jpg' , 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original' , image1)
        cv2.imshow('Image 2 Original' , image2)
        cv2.imshow('Image Tambah' , image_tambah)
        cv2.imshow('Image Kurang' , image_kurang)
        cv2.waitKey()
    def aritmatika2(self):
        image1 = cv2.imread('lilypichu.jpg' , 0)   
        image2 = cv2.imread('toothless.jpg' , 0)
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image Kali' , image_kali)
        cv2.imshow('Image Bagi' , image_bagi)
        cv2.waitKey()
    
    # OPERASI BOOLEAN
    def operasiAND(self):
        image1 = cv2.imread('lilypichu.jpg' , 1)
        image2 = cv2.imread('toothless.jpg' , 1)
        # Konversi warna gambar dari BGR ke RGB (OpenCV membaca gambar dalam format BGR)
        image1 = cv2.cvtColor(image1 , cv2.COLOR_BGR2RGB)        
        image2 = cv2.cvtColor(image2 , cv2.COLOR_BGR2RGB) 
        operation = cv2.bitwise_and(image1 , image2)
        cv2.imshow('Image Kali' , image1)
        cv2.imshow('Image Bagi' , image2)
        cv2.imshow('Operasi AND' , operation)
    def operasiOR(self):
        image1 = cv2.imread('lilypichu.jpg' , 1)
        image2 = cv2.imread('toothless.jpg' , 1)
        # Konversi warna gambar dari BGR ke RGB (OpenCV membaca gambar dalam format BGR)
        image1 = cv2.cvtColor(image1 , cv2.COLOR_BGR2RGB)        
        image2 = cv2.cvtColor(image2 , cv2.COLOR_BGR2RGB) 
        operation = cv2.bitwise_or(image1 , image2)
        cv2.imshow('Image Kali' , image1)
        cv2.imshow('Image Bagi' , image2)
        cv2.imshow('Operasi OR' , operation)
    def operasiXOR(self):
        image1 = cv2.imread('lilypichu.jpg' , 1)
        image2 = cv2.imread('toothless.jpg' , 1)
        # Konversi warna gambar dari BGR ke RGB (OpenCV membaca gambar dalam format BGR)
        image1 = cv2.cvtColor(image1 , cv2.COLOR_BGR2RGB)        
        image2 = cv2.cvtColor(image2 , cv2.COLOR_BGR2RGB) 
        operation = cv2.bitwise_xor(image1 , image2)
        cv2.imshow('Image Kali' , image1)
        cv2.imshow('Image Bagi' , image2)
        cv2.imshow('Operasi XOR' , operation)
        
    def histogramEqu(self):
        # Mengubah image array menjadi 1 dimensi
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()  # Menghitung jumlah array pada sumbu tertentu
        cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi
        # Menutup nilai yang sama dengan yang diberikan
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Perhitungan
        # Mengisi nilai array dengan skalar
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.Image = cdf[self.Image]  # Mengganti nilai image menjadi nilai kumulatif
        self.displayImage(self.citra_label)  # Menampilkan Image di window ke-2

        plt.plot(cdf_normalized, color="b")  # Plotting sesuai normalisasi
        # Membuat histogram sesuai dengan nilai gambar
        plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])  # Mengatur batas sumbu x
        plt.legend(("CDF", "Histogram"), loc="upper left")  # Text di histogram
        plt.show()  # Melakukan visualisasi dari histogram 
        
    # --------------------------------------------------------- D1 - D6 ---------------------------------------------------------
    def konvolusi2D(self):
        # Open file dialog to select an image
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.png)")

        if filepath:
            # Read the image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Define the kernel
            kernel = np.array([[6, 0, -6],
                                [6, 1, -6],
                                [6, 0, -6]])

            # Apply convolution
            img_out = cv2.filter2D(img, -1, kernel)

            # Display the filtered image within the application
            self.displayFilteredImg(img_out)

    def displayFilteredImg(self, img):
        # Convert the numpy array to QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # Display the QPixmap in the QLabel
        self.citra_label.setPixmap(pixmap)
        self.citra_label.setAlignment(Qt.AlignCenter)
    
    def meanFilter(self):
        # Define kernel for mean filter (3x3)
        kernel = np.ones((3, 3), np.float32) / 9  
        # Apply mean filter convolution to the image
        self.Image = cv2.filter2D(self.Image, -1, kernel)
        # Display the filtered image
        self.displayImage(self.citra_label)
    
    def gaussianFilter(self):
        # Tentukan ukuran kernel (misalnya, 3x3, 5x5)
        ukuran_kernel = (3, 3)
        # Tentukan nilai standar deviasi (sigma)
        sigma = 1
        
        # Buat kernel Gaussian menggunakan fungsi bawaan OpenCV
        kernel = cv2.getGaussianKernel(ukuran_kernel[0], sigma)
        # Lakukan konvolusi 2D menggunakan kernel Gaussian
        self.Image = cv2.filter2D(self.Image, -1, kernel * np.transpose(kernel))
        
        # Tampilkan citra hasil filtering
        self.displayImage(self.citra_label)
        
    def imageSharpeningLaplacian(self):
         # Tentukan kernel filter yang digunakan untuk penajaman citra (misalnya, Laplacian)
        kernel = np.array([[0, 0, -1, 0, 0],
                        [0, -1, -2, -1, 0],
                        [-1, -2, 16, -2, -1],
                        [0, -1, -2, -1, 0],
                        [0, 0, -1, 0, 0]]) * (1.0 / 16)

        # Lakukan konvolusi 2D menggunakan kernel filter Laplacian
        self.Image = cv2.filter2D(self.Image, -1, kernel)

        # Tampilkan citra hasil penajaman
        self.displayImage(self.citra_label)
    def imageSharpening(self):
        # Define the kernel for basic image sharpening
        kernel = np.array([[0, 1, 0],
                        [1, 4, 1],
                        [0, 1, 0]])

        # Apply 2D convolution using the kernel
        self.Image = cv2.filter2D(self.Image, -1, kernel)

        # Display the sharpened image
        self.displayImage(self.citra_label)


    def medianFilter(self):
        # Convert image to grayscale
        gray_img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        # Copy grayscale image
        img_out = gray_img.copy()

        # Ukuran citra
        h, w = gray_img.shape

        # Proses median filtering
        for i in range(3, h-3):
            for j in range(3, w-3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray_img[i+k, j+l]
                        neighbors.append(a)
                # Mengurutkan neighbors
                neighbors.sort()
                # Mengambil nilai median
                median = neighbors[24]
                # Mengganti nilai piksel dengan nilai median
                img_out[i, j] = median

        # Menampilkan citra hasil median filtering
        self.Image = img_out
        self.displayImage(self.citra_label)
    
    def maxFiltering(self):
        # Convert image to grayscale
        gray_img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        # Copy grayscale image
        img_out = gray_img.copy()

        # Ukuran citra
        h, w = gray_img.shape

        # Proses Maximum Filtering
        for i in range(3, h-3):
            for j in range(3, w-3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray_img[i + k, j + l]
                        neighbors.append(a)
                # Mencari nilai maksimum dari tetangga
                max_value = max(neighbors)
                # Menetapkan nilai maksimum ke piksel output
                img_out.itemset((i, j), max_value)

        # Menampilkan citra hasil Maximum Filtering
        self.Image = img_out
        self.displayImage(self.citra_label)

    def minFiltering(self):
        # Convert image to grayscale
        gray_img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        # Copy grayscale image
        img_out = gray_img.copy()

        # Ukuran citra
        h, w = gray_img.shape

        # Proses Minimum Filtering
        for i in range(3, h-3):
            for j in range(3, w-3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray_img[i + k, j + l]
                        neighbors.append(a)
                # Mencari nilai minimum dari tetangga
                min_value = min(neighbors)
                # Menetapkan nilai minimum ke piksel output
                img_out.itemset((i, j), min_value)

        # Menampilkan citra hasil Minimum Filtering
        self.Image = img_out
        self.displayImage(self.citra_label)  
    
    # -------------------------------------- DFT --------------------------
    def lpf(self):
        # Membuat array nilai x
        x = np.arange(256)
        # Membuat array nilai y sebagai sinus dengan frekuensi 1/32 dari x
        y = np.sin(2 * np.pi * x / 32)
        # Menambahkan nilai maksimum dari y ke semua elemen y
        y += max(y)
        # Membuat citra dengan nilai piksel yang dihasilkan dari nilai y    
        Img = np.array([[y[j]*127 for j in range(256)]for i in range(256)], dtype=np.uint8)
        # Menampilkan citra menggunakan matplotlib
        plt.imshow(Img)
        # Membaca citra dalam skala abu-abu
        Img = cv2.imread('img-noise.jpg',0)
        # Melakukan transformasi Fourier diskrit (DFT) pada citra menggunakan OpenCV
        dft = cv2.dft(np.float32(Img),flags = cv2.DFT_COMPLEX_OUTPUT)
        # Menggeser frekuensi nol ke tengah
        dft_shift = np.fft.fftshift(dft)
        # Menghitung magnitudo spektrum frekuensi
        maginitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
        # Mendapatkan dimensi citra
        rows, cols = Img.shape
        # Mendapatkan koordinat pusat citra
        crow, ccol = int(rows / 2), int(cols / 2)
        # Membuat mask untuk filtering frekuensi
        mask = np.zeros((rows, cols , 2),np.uint8)
        # Menentukan radius mask
        r = 50
        # Menentukan pusat mask
        center = [crow , ccol]
        # Membuat grid x, y
        x, y = np.ogrid[:rows , :cols]
        # Menentukan area mask dengan jarak dari pusat kurang dari radius
        mask_area = (x - center[0])** 2 + (y - center[1])** 2 <= r*r 
        # Mengisi area mask dengan nilai 1
        mask[mask_area] = 1
        # Melakukan filtering pada spektrum frekuensi
        fshift = dft_shift * mask
        # Menghitung magnitudo spektrum frekuensi hasil filtering
        fshift_maxk_mag = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        # Melakukan inversi pergeseran frekuensi
        f_ishift = np.fft.ifftshift(fshift)
        # Melakukan inversi DFT untuk mendapatkan citra spasial
        img_back = cv2.idft(f_ishift)
        # Mengambil magnitudo citra balik
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        # Menampilkan citra-citra dan hasil prosesnya menggunakan matplotlib
        fig = plt.figure(figsize = (12,12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(Img, cmap = 'gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(maginitude_spectrum, cmap = 'gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(fshift_maxk_mag, cmap = 'gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(img_back, cmap = 'gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()
    def hpf(self):
        # Membuat array nilai x
        x = np.arange(256)
        # Membuat array nilai y sebagai sinus dengan frekuensi 1/32 dari x
        y = np.sin(2 * np.pi * x / 32)
        # Menambahkan nilai maksimum dari y ke semua elemen y
        y += max(y)
        # Membuat citra dengan nilai piksel yang dihasilkan dari nilai y
        Img = np.array([[y[j]*127 for j in range(256)]for i in range(256)], dtype=np.uint8)
        # Menampilkan citra menggunakan matplotlib
        plt.imshow(Img)
        # Membaca citra 'img-noise1.jpg' dalam skala abu-abu
        Img = cv2.imread('img-noise.jpg',0)
        # Melakukan transformasi Fourier diskrit (DFT) pada citra menggunakan OpenCV
        dft = cv2.dft(np.float32(Img),flags = cv2.DFT_COMPLEX_OUTPUT)
        # Menggeser frekuensi nol ke tengah
        dft_shift = np.fft.fftshift(dft)
        # Menghitung magnitudo spektrum frekuensi
        maginitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
        # Mendapatkan dimensi citra
        rows, cols = Img.shape
        # Mendapatkan koordinat pusat citra
        crow, ccol = int(rows / 2), int(cols / 2)
        # Membuat mask untuk filtering frekuensi
        mask = np.ones((rows, cols , 2),np.uint8)
        # Menentukan radius mask
        r = 80
        # Menentukan pusat mask
        center = [crow , ccol]
        # Membuat grid x, y
        x, y = np.ogrid[:rows , :cols]
        # Menentukan area mask dengan jarak dari pusat lebih besar dari radius
        mask_area = (x - center[0])** 2 + (y - center[1])** 2 <= r*r 
        # Mengisi area mask dengan nilai 0
        mask[mask_area] = 0
        # Melakukan filtering pada spektrum frekuensi
        fshift = dft_shift * mask
        # Menghitung magnitudo spektrum frekuensi hasil filtering
        fshift_maxk_mag = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
        # Melakukan inversi pergeseran frekuensi
        f_ishift = np.fft.ifftshift(fshift)
        # Melakukan inversi DFT untuk mendapatkan citra spasial
        img_back = cv2.idft(f_ishift)
        # Mengambil magnitudo citra balik
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        # Menampilkan citra-citra dan hasil prosesnya menggunakan matplotlib
        fig = plt.figure(figsize = (12,12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(Img, cmap = 'gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(maginitude_spectrum, cmap = 'gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(fshift_maxk_mag, cmap = 'gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2,2,4)
        ax4.imshow(img_back, cmap = 'gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()
        
    # ---------------------------------------------------------- DETEKSI TEPI ----------------------------------------------------
    def sobel_edge_detection(self):
        # Convert RGB image to Grayscale
        gray_image = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        
        # Inisialisasi kernel Sobel
        sobel_kernel_x = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
        
        sobel_kernel_y = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])
        
        # Konvolusi gambar dengan kernel Sobel
        gradient_x = cv2.filter2D(gray_image, -1, sobel_kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, sobel_kernel_y)
        
        # Hitung magnitudo gradien
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisasi magnitudo gradien ke rentang 0-255
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
        
        # Konversi magnitudo gradien ke tipe data uint8
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Tampilkan gambar hasil
        cv2.imshow('Sobel Edge Detection', gradient_magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def prewitt_edge_detection(self):
        # Convert RGB image to Grayscale
        gray_image = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        
        # Inisialisasi kernel Prewitt
        prewitt_kernel_x = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
        
        prewitt_kernel_y = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])
        
        # Konvolusi gambar dengan kernel Prewitt
        gradient_x = cv2.filter2D(gray_image, -1, prewitt_kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, prewitt_kernel_y)
        
        # Hitung magnitudo gradien
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisasi magnitudo gradien ke rentang 0-255
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
        
        # Konversi magnitudo gradien ke tipe data uint8
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Tampilkan gambar hasil
        cv2.imshow('Prewitt Edge Detection', gradient_magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def roberts_edge_detection(self):
        # Convert RGB image to Grayscale
        gray_image = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        
        # Inisialisasi kernel Roberts
        roberts_kernel_x = np.array([[1, 0],
                                    [0, -1]])
        
        roberts_kernel_y = np.array([[0, 1],
                                    [-1, 0]])
        
        # Konvolusi gambar dengan kernel Roberts
        gradient_x = cv2.filter2D(gray_image, -1, roberts_kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, roberts_kernel_y)
        
        # Hitung magnitudo gradien
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisasi magnitudo gradien ke rentang 0-255
        gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
        
        # Konversi magnitudo gradien ke tipe data uint8
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Tampilkan gambar hasil
        cv2.imshow('Roberts Edge Detection', gradient_magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def canny_edge_detection(self):
        # 1. Baca Gambar dan Inisialisasi Variabel
        img = cv2.imread('lilypichu.jpg', 0)
        H, W = img.shape

        # 2. Reduksi Noise dengan Filter Gauss
        gauss = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
            [1, 3, 5, 3, 1],
            [2, 5, 9, 5, 2],
            [1, 3, 5, 3, 1],
            [0, 1, 2, 1, 0]])
        hasil = cv2.filter2D(img, -1, gauss)

        # 3. Hitung Gradien dengan Operator Sobel
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        Gx = cv2.filter2D(hasil, -1, sobel_x)
        Gy = cv2.filter2D(hasil, -1, sobel_y)

        # Hitung magnitudo dan arah gradien
        gradient_magnitude = np.sqrt((Gx ** 2) + (Gy ** 2))
        gradient_normalized = ((gradient_magnitude / np.max(gradient_magnitude)) * 255).astype(np.uint8)
        theta = np.arctan2(Gy, Gx)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        # 4. Non-Maximum Suppression
        img_out = np.zeros((H, W), dtype=np.float64)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = gradient_normalized[i, j + 1]
                        r = gradient_normalized[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = gradient_normalized[i + 1, j - 1]
                        r = gradient_normalized[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = gradient_normalized[i + 1, j]
                        r = gradient_normalized[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = gradient_normalized[i - 1, j - 1]
                        r = gradient_normalized[i + 1, j + 1]
                    if (gradient_normalized[i, j] >= q) and (gradient_normalized[i, j] >= r):
                        img_out[i, j] = gradient_normalized[i, j]
                    else:
                        img_out[i, j] = 0
                except IndexError as e:
                    pass
        img_N = img_out.astype("uint8")

        # 5. Hysteresis Thresholding
        weak = 30
        strong = 50
        for i in range(H):
            for j in range(W):
                a = img_N[i, j]
                if a > weak:
                    b = weak
                    if a > strong:
                        b = 255
                else:
                    b = 0
                img_N[i, j] = b

        img_H1 = img_N.copy()

        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_H1[i, j] == weak:
                    try:
                        if (img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or \
                                (img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or \
                                (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or \
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")

        # 6. Menampilkan Hasil dengan Matplotlib
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(hasil, cmap='gray')
        ax1.title.set_text("Noise Reduction")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(gradient_normalized, cmap='gray')
        ax2.title.set_text('Finding Gradien')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img_N, cmap='gray')
        ax3.title.set_text('Non Maximum Supression')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_H2, cmap='gray')
        ax4.title.set_text('Hysterisis Thresholding')
        plt.show()

     #  ------------------------------------ Morfologi ---------------------------------------------------- 
    def dilasi(self):
        # Mengubah citra ke citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Melakukan thresholding menggunakan metode Otsu
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Membuat elemen struktur untuk operasi morfologi dilasi
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        # Melakukan operasi dilasi pada citra menggunakan elemen struktur yang telah dibuat
        imgh = cv2.dilate(thresh, strel, iterations=1) 
        # Menampilkan citra hasil dilasi
        cv2.imshow('Dilasi', imgh)
        # Mencetak nilai piksel citra hasil dilasi
        print(imgh)
    
    def erosi(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Melakukan thresholding menggunakan metode Otsu untuk mendapatkan citra biner terbalik
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Membuat elemen struktur untuk operasi morfologi erosi (Cross-shaped)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        # Melakukan operasi erosi pada citra biner terbalik menggunakan elemen struktur yang telah dibuat
        imgh = cv2.erode(thresh, strel, iterations=1)
        # Menampilkan citra hasil erosi
        cv2.imshow('Erosi', imgh)
        # Mencetak nilai piksel citra hasil erosi
        print(imgh) 

    def opening(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Melakukan thresholding menggunakan metode Otsu untuk mendapatkan citra biner terbalik
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Membuat elemen struktur untuk operasi morfologi opening (Rectangular-shaped)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        # Melakukan operasi opening pada citra menggunakan elemen struktur yang telah dibuat
        imgh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, strel) 
        # Menampilkan citra hasil opening
        cv2.imshow('Opening', imgh)
        # Mencetak nilai piksel citra hasil opening
        print(imgh)

    def closing(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Melakukan thresholding menggunakan metode Otsu untuk mendapatkan citra biner terbalik
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Membuat elemen struktur untuk operasi morfologi closing (Rectangular-shaped)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) 
        # Melakukan operasi closing pada citra menggunakan elemen struktur yang telah dibuat
        imgh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strel, iterations=1) 
        # Menampilkan citra hasil closing
        cv2.imshow('Closing', imgh)
        # Mencetak nilai piksel citra hasil closing
        print(imgh)    

    # ----------------------------------- Global Thresholding ---------------------------------------------
    def binary(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Menetapkan ambang batas dan nilai maksimum
        T = 127 
        max = 255
        # Melakukan thresholding biner
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_BINARY)
        # Menampilkan citra hasil thresholding biner
        cv2.imshow('Binary', thresh)
        # Mencetak nilai piksel citra hasil thresholding
        print(thresh)

    
    def binaryinvers(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Menetapkan ambang batas dan nilai maksimum
        T = 127 
        max = 255
        # Melakukan thresholding biner terbalik
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_BINARY_INV)
        # Menampilkan citra hasil thresholding biner terbalik
        cv2.imshow('Binary Invers', thresh)
        # Mencetak nilai piksel citra hasil thresholding
        print(thresh)

   
    def turnc(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Menetapkan ambang batas dan nilai maksimum
        T = 127 
        max = 255
        # Melakukan thresholding Truncate
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TRUNC)
        # Menampilkan citra hasil thresholding Truncate
        cv2.imshow('Turnc', thresh)
        # Mencetak nilai piksel citra hasil thresholding
        print(thresh)

    
    def tozero(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Menetapkan ambang batas dan nilai maksimum
        T = 127 
        max = 255
        # Melakukan thresholding To Zero
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TOZERO)
        # Menampilkan citra hasil thresholding To Zero
        cv2.imshow('To Zero', thresh)
        # Mencetak nilai piksel citra hasil thresholding
        print(thresh)

    
    def tozeroinvers(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Menetapkan ambang batas dan nilai maksimum
        T = 127 
        max = 255
        # Melakukan thresholding To Zero Invers
        ret, thresh = cv2.threshold(img, T, max, cv2.THRESH_TOZERO_INV)
        # Menampilkan citra hasil thresholding To Zero Invers
        cv2.imshow('To Zero Invers', thresh)
        # Mencetak nilai piksel citra hasil thresholding
        print(thresh)


    # ------------------------------------------------ Local Thresholding ------------------------------------------------ 
    def meanthres(self):
        # Mengonversi citra input dari BGR ke grayscale
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Mencetak nilai piksel awal dari citra grayscale
        print('pixel awal', img)
        # Menerapkan thresholding adaptif menggunakan metode rata-rata
        # dan menampilkan citra hasilnya
        imgh = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('MEAN TRESHOLDING', imgh)
        # Mencetak citra hasil thresholding adaptif
        print('PIXEL MEAN THRESHOLDING')
        print(imgh) 


    def gaussianthres(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Mencetak nilai piksel citra awal
        print('pixel awal', img)
        # Melakukan thresholding adaptif Gaussian
        imgh = cv2.adaptiveThreshold(img, 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        # Menampilkan citra hasil Gaussian thresholding
        cv2.imshow('GAUSSSIAN TRESHOLDING', imgh)
        # Mencetak nilai piksel citra hasil Gaussian thresholding
        print('PIXEL GAUSSIAN THRESHOLDING')
        print(imgh)

    def otsuthres(self):
        # Mengubah citra menjadi citra keabuan (grayscale)
        img_gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        # Melakukan thresholding Otsu
        ret, imgh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Mencetak nilai ambang batas Otsu yang dihitung
        print('Nilai otsu', ret)
        # Menampilkan citra hasil Otsu thresholding
        cv2.imshow('OTSU THRESHOLDING', imgh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --------------------------------------- Contour ---------------------------------------
    # untuk mendeteksi dan menganalisis kontur (garis atau batas) dari objek-objek yang ada dalam citra
    def contour(self):
        # Membaca gambar shape.jpg
        image = cv2.imread('shapes.png')

        # Mengonversi gambar ke dalam skala abu-abu
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Membuat gambar biner menggunakan thresholding
        _, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Mencari kontur pada gambar biner
        contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # List untuk menyimpan poligon pendekatan
        approx_polygons = []

        # Iterasi melalui setiap kontur yang ditemukan
        for contour in contours:
            # Pendekatan kontur dengan epsilon 2% dari keliling kontur
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, False)
            approx_polygons.append(approx)

            # Pendekatan kedua dengan epsilon 1% dari keliling kontur
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Menggambar kotak pembatas di sekitar objek
            x, y, w, h = cv2.boundingRect(approx)#untuk mendapatkan kotak pembatas dari suatu kontur
            x_mid = int(x + (w / 3))
            y_mid = int(y + (h / 1.5))
            coords = (x_mid, y_mid)
            colour = (0, 0, 0)
            font = cv2.FONT_HERSHEY_DUPLEX

            # Mengidentifikasi bentuk geometri
            if len(approx) == 3:
                cv2.putText(image, "Triangle", coords, font, 1, colour, 1)  #  mengambil Teks segitiga
            elif len(approx) == 4:
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    cv2.putText(image, "Square", coords, font, 1, colour, 1)  # mengambil Teks persegi
                else:
                    cv2.putText(image, "Rectangle", coords, font, 1, colour, 1)  # mengambil Teks persegi panjang
            elif len(approx) == 5:
                cv2.putText(image, "Pentagon", coords, font, 1, colour, 1)  # mengambil Teks segi lima
            elif len(approx) == 6:
                cv2.putText(image, "Hexagon", coords, font, 1, colour, 1)  # mengambil teks segi enam
            else:
                cv2.putText(image, "Circle", coords, font, 1, colour, 1)  # mengambil teks lingkaran

        # Menggambar kontur dan poligon pendekatan pada gambar
        for i, contour in enumerate(approx_polygons):
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Menggambar kontur dengan warna hijau

        # Menampilkan gambar dengan hasil deteksi
        cv2.imshow("shapes_detected", image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
 

    # IMAGE DISPLAY FUNCTION            
    def displayImage(self, label):
        qformat = QImage.Format_Indexed8
        
        if len(self.Image.shape) == 3:
            if(self.Image.shape[2])== 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0],qformat)
        
        img = img.rgbSwapped() 
        
        label.setPixmap(QPixmap.fromImage(img))           

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 2')
window.show()
sys.exit(app.exec_())
