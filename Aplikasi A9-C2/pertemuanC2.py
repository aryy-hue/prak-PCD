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
        
        # SLIDER
        self.brightnessSlider.valueChanged.connect(self.brightnessChanged)
        self.contrastSlider.valueChanged.connect(self.contrastChanged)
        self.brightnessValue = 80
        self.contrastValue = 5.7
        
    def fungsi(self):
        self.Image = cv2.imread('image.jpg')
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
