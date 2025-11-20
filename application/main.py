import sys
import csv
import design
import numpy as np
import os, shutil, glob
from PyQt5 import QtWidgets, QtGui, QtCore
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Application(QtWidgets.QMainWindow, design.Ui_Classificator):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model_chest = load_model('chest_CNN.h5')
        self.model_skin = load_model('FT_ResNet50.h5')
        self.model_leukemia = load_model('leukemia_CNN.h5')
        self.statusBar.showMessage('Ожидание снимка')
        self.diagnosted = False
        self.image_loaded = False
        self.btnDownload.clicked.connect(self.browse_image)
        self.btnStart.clicked.connect(self.classification)
        self.btnSaveTXT.clicked.connect(self.save_txt)
        self.btnSaveCSV.clicked.connect(self.save_csv)
    
    def browse_image(self):
        image = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите снимок", "*.jpg;; *.jpeg;; *.png;; *.bmp")
        if image[0]:
            self.textWidget.clear() 
            self.image_path = image[0]
            self.pixmap = QtGui.QPixmap(self.image_path)
            self.pixmap = self.pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(self.pixmap)
            self.statusBar.showMessage('Снимок загружен')
            self.image_loaded = True
            self.diagnosted = False
    
    def status_check(self):
        if not self.diagnosted:
            self.statusBar.showMessage('Диагностирование не пройдено')
            return False
        elif not self.lineName.text():
            self.statusBar.showMessage('Отсутствует имя пациента')
            return False
        else:
            return True
            
    def classification(self):
        if not self.image_loaded:
            self.statusBar.showMessage('Необходимо загрузить снимок')
            return
        self.textWidget.clear()
        if self.comboBox.currentText() == 'Заболевания легких':
            pred = predict(self.image_path, self.model_chest, 256)
            classes = ['Аденокарцинома', 'Крупноклеточная карцинома', 'Норма', 'Плоскоклеточная карцинома']
        elif self.comboBox.currentText() == 'Заболевания кожи':
            pred = predict(self.image_path, self.model_skin, 180)
            classes = ['Норма', 'Меланома']
        elif self.comboBox.currentText() == 'Заболевания крови':
            pred = predict(self.image_path, self.model_leukemia, 180)
            classes = ['Лейкемия', 'Норма']
        for i in range(len(classes)):
            self.textWidget.addItem(str(np.round(pred[:,i]*100,2)) + ' - ' + classes[i])
        self.diagnosted = True
        self.statusBar.showMessage('Диагностирование завершено')

    def save_txt(self):
        if not self.status_check():
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Сохранить файл")
        if directory:
            text = 'Пациент: ' + self.lineName.text() + '\n'
            text += '\n'.join(self.textWidget.item(i).text() for i in range(self.textWidget.count()))
            with open(directory + '/' + self.lineName.text() + '.txt', 'w') as txt_file:
                txt_file.write(text)
            self.statusBar.showMessage('Отчет сохранен в формате .txt')

    def save_csv(self):
        if not self.status_check():
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Сохранить файл")
        if directory:
            header = ['Вероятность', 'Заболевание']
            with open(directory + '/' + self.lineName.text() + '.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                for i in range(self.textWidget.count()):
                    text = self.textWidget.item(i).text()
                    text = text.replace(' ', '', 2).split('-')
                    writer.writerow(text)
            self.statusBar.showMessage('Отчет сохранен в формате .csv')

def predict(img_path, model, size): 
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    return prediction
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Application() 
    window.show() 
    app.exec_() 
    
if __name__ == '__main__': 
    main() 