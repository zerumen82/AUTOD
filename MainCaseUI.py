# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/MAIN_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(567, 224)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Fixedsys")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setFocusPolicy(QtCore.Qt.ClickFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/INdaHouse/.designer/icn/icons8-pill-64.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.fotImaBut = QtWidgets.QPushButton(self.centralwidget)
        self.fotImaBut.setGeometry(QtCore.QRect(40, 30, 161, 41))
        self.fotImaBut.setStyleSheet("background-color: rgb(0, 255, 0);\n"
"selection-color: rgb(0, 255, 127);\n"
"font: 75 12pt \"MS Shell Dlg 2\";\n"
"")
        self.fotImaBut.setObjectName("fotImaBut")
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionAyuda = QtWidgets.QAction(MainWindow)
        self.actionAyuda.setObjectName("actionAyuda")
        self.actionIm_genes = QtWidgets.QAction(MainWindow)
        self.actionIm_genes.setObjectName("actionIm_genes")
        self.actionVideo = QtWidgets.QAction(MainWindow)
        self.actionVideo.setObjectName("actionVideo")
        self.actionVoz = QtWidgets.QAction(MainWindow)
        self.actionVoz.setObjectName("actionVoz")
        self.actionGenerador = QtWidgets.QAction(MainWindow)
        self.actionGenerador.setObjectName("actionGenerador")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AUTO-DEEP"))
        self.fotImaBut.setText(_translate("MainWindow", "Fotos y Videos"))
        self.actionAyuda.setText(_translate("MainWindow", "Ayuda"))
        self.actionIm_genes.setText(_translate("MainWindow", "Fotos y Videos"))
        self.actionVideo.setText(_translate("MainWindow", "Video"))
        self.actionVoz.setText(_translate("MainWindow", "Voz"))
        self.actionGenerador.setText(_translate("MainWindow", "Generador"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())