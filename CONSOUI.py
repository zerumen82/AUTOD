# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/CONSO_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AUTODEEPCONSO(object):
    def setupUi(self, AUTODEEPCONSO):
        AUTODEEPCONSO.setObjectName("AUTODEEPCONSO")
        AUTODEEPCONSO.resize(658, 368)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AUTODEEPCONSO.sizePolicy().hasHeightForWidth())
        AUTODEEPCONSO.setSizePolicy(sizePolicy)
        self.CONSOLE = QtWidgets.QPlainTextEdit(AUTODEEPCONSO)
        self.CONSOLE.setGeometry(QtCore.QRect(20, 20, 611, 331))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CONSOLE.sizePolicy().hasHeightForWidth())
        self.CONSOLE.setSizePolicy(sizePolicy)
        self.CONSOLE.setObjectName("CONSOLE")

        self.retranslateUi(AUTODEEPCONSO)
        QtCore.QMetaObject.connectSlotsByName(AUTODEEPCONSO)

    def retranslateUi(self, AUTODEEPCONSO):
        _translate = QtCore.QCoreApplication.translate
        AUTODEEPCONSO.setWindowTitle(_translate("AUTODEEPCONSO", "Form"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    AUTODEEPCONSO = QtWidgets.QWidget()
    ui = Ui_AUTODEEPCONSO()
    ui.setupUi(AUTODEEPCONSO)
    AUTODEEPCONSO.show()
    sys.exit(app.exec_())
