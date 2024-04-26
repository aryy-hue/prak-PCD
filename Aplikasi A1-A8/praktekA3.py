import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui',self)
        self.Image = None
        self.btn_img.clicked.connect(self.fungsi)
        self.btn_gray.clicked.connect(self.grayscale)
        
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
        
    def displayImage(self, label):
        qformat = QImage.Format_Indexed8
        
        if len(self.Image.shape) == 3:
            if self.Image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)
        
        img = img.rgbSwapped() 
        
        label.setPixmap(QPixmap.fromImage(img))             


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 1')
window.show()
sys.exit(app.exec_())
