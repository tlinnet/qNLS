/********************************************************************************
** Form generated from reading UI file 'poissonqtapp.ui'
**
** Created: Sun Jul 6 19:58:22 2014
**      by: Qt User Interface Compiler version 4.7.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_POISSONQTAPP_H
#define UI_POISSONQTAPP_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QProgressBar>
#include <QtGui/QPushButton>
#include <QtGui/QSpinBox>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_poissonqtapp
{
public:
    QDoubleSpinBox *doublespinnus;
    QSpinBox *spintd1;
    QCheckBox *checkct1;
    QDoubleSpinBox *doublespint2;
    QPushButton *pushok;
    QPushButton *pushcancel;
    QLabel *label_3;
    QLabel *label_4;
    QDoubleSpinBox *doublespinsw;
    QDoubleSpinBox *doublespinfield;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QLabel *label_8;
    QPushButton *pushbrowse;
    QLineEdit *lineeditpath;
    QLineEdit *lineeditout;
    QLabel *label_9;
    QPushButton *pushbrowse2;
    QProgressBar *progressBar;
    QLabel *label_2;
    QSpinBox *spinBox_seed;
    QSpinBox *spinBox_niter;
    QLabel *label_10;
    QLabel *done_label;
    QPushButton *pushcancel_2;

    void setupUi(QWidget *poissonqtapp)
    {
        if (poissonqtapp->objectName().isEmpty())
            poissonqtapp->setObjectName(QString::fromUtf8("poissonqtapp"));
        poissonqtapp->resize(592, 366);
        poissonqtapp->setMinimumSize(QSize(592, 366));
        doublespinnus = new QDoubleSpinBox(poissonqtapp);
        doublespinnus->setObjectName(QString::fromUtf8("doublespinnus"));
        doublespinnus->setGeometry(QRect(190, 210, 91, 25));
        doublespinnus->setMaximum(1);
        doublespinnus->setSingleStep(0.01);
        doublespinnus->setValue(0.6);
        spintd1 = new QSpinBox(poissonqtapp);
        spintd1->setObjectName(QString::fromUtf8("spintd1"));
        spintd1->setGeometry(QRect(190, 30, 91, 25));
        spintd1->setMaximum(9999);
        spintd1->setValue(256);
        checkct1 = new QCheckBox(poissonqtapp);
        checkct1->setObjectName(QString::fromUtf8("checkct1"));
        checkct1->setGeometry(QRect(190, 140, 121, 20));
        doublespint2 = new QDoubleSpinBox(poissonqtapp);
        doublespint2->setObjectName(QString::fromUtf8("doublespint2"));
        doublespint2->setGeometry(QRect(190, 170, 91, 25));
        doublespint2->setMaximum(99.99);
        doublespint2->setValue(50);
        pushok = new QPushButton(poissonqtapp);
        pushok->setObjectName(QString::fromUtf8("pushok"));
        pushok->setGeometry(QRect(450, 330, 114, 32));
        pushcancel = new QPushButton(poissonqtapp);
        pushcancel->setObjectName(QString::fromUtf8("pushcancel"));
        pushcancel->setGeometry(QRect(320, 330, 114, 32));
        label_3 = new QLabel(poissonqtapp);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(80, 170, 41, 31));
        label_4 = new QLabel(poissonqtapp);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(80, 210, 31, 16));
        doublespinsw = new QDoubleSpinBox(poissonqtapp);
        doublespinsw->setObjectName(QString::fromUtf8("doublespinsw"));
        doublespinsw->setGeometry(QRect(190, 60, 91, 25));
        doublespinsw->setValue(60);
        doublespinfield = new QDoubleSpinBox(poissonqtapp);
        doublespinfield->setObjectName(QString::fromUtf8("doublespinfield"));
        doublespinfield->setGeometry(QRect(190, 100, 91, 25));
        doublespinfield->setMaximum(999.99);
        doublespinfield->setValue(81);
        label_5 = new QLabel(poissonqtapp);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(80, 30, 21, 16));
        label_5->setLayoutDirection(Qt::LeftToRight);
        label_6 = new QLabel(poissonqtapp);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(80, 60, 41, 31));
        label_6->setLayoutDirection(Qt::LeftToRight);
        label_7 = new QLabel(poissonqtapp);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(80, 100, 62, 31));
        label_8 = new QLabel(poissonqtapp);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(80, 250, 121, 16));
        pushbrowse = new QPushButton(poissonqtapp);
        pushbrowse->setObjectName(QString::fromUtf8("pushbrowse"));
        pushbrowse->setGeometry(QRect(380, 250, 91, 32));
        lineeditpath = new QLineEdit(poissonqtapp);
        lineeditpath->setObjectName(QString::fromUtf8("lineeditpath"));
        lineeditpath->setGeometry(QRect(210, 250, 161, 22));
        lineeditpath->setContextMenuPolicy(Qt::DefaultContextMenu);
        lineeditout = new QLineEdit(poissonqtapp);
        lineeditout->setObjectName(QString::fromUtf8("lineeditout"));
        lineeditout->setGeometry(QRect(210, 290, 161, 22));
        lineeditout->setContextMenuPolicy(Qt::DefaultContextMenu);
        label_9 = new QLabel(poissonqtapp);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setGeometry(QRect(80, 290, 131, 16));
        pushbrowse2 = new QPushButton(poissonqtapp);
        pushbrowse2->setObjectName(QString::fromUtf8("pushbrowse2"));
        pushbrowse2->setGeometry(QRect(380, 290, 91, 32));
        progressBar = new QProgressBar(poissonqtapp);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setGeometry(QRect(80, 340, 211, 20));
        progressBar->setValue(0);
        label_2 = new QLabel(poissonqtapp);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(380, 100, 111, 16));
        spinBox_seed = new QSpinBox(poissonqtapp);
        spinBox_seed->setObjectName(QString::fromUtf8("spinBox_seed"));
        spinBox_seed->setGeometry(QRect(360, 120, 151, 24));
        spinBox_seed->setMaximum(999999999);
        spinBox_niter = new QSpinBox(poissonqtapp);
        spinBox_niter->setObjectName(QString::fromUtf8("spinBox_niter"));
        spinBox_niter->setGeometry(QRect(360, 60, 151, 24));
        spinBox_niter->setMinimum(1);
        spinBox_niter->setMaximum(100000);
        spinBox_niter->setValue(100);
        label_10 = new QLabel(poissonqtapp);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setGeometry(QRect(380, 40, 111, 16));
        done_label = new QLabel(poissonqtapp);
        done_label->setObjectName(QString::fromUtf8("done_label"));
        done_label->setEnabled(true);
        done_label->setGeometry(QRect(90, 320, 201, 20));
        done_label->setMinimumSize(QSize(171, 0));
        done_label->setTextFormat(Qt::PlainText);
        done_label->setScaledContents(false);
        pushcancel_2 = new QPushButton(poissonqtapp);
        pushcancel_2->setObjectName(QString::fromUtf8("pushcancel_2"));
        pushcancel_2->setGeometry(QRect(320, 330, 114, 32));
        pushcancel_2->raise();
        doublespinnus->raise();
        spintd1->raise();
        checkct1->raise();
        doublespint2->raise();
        pushok->raise();
        pushcancel->raise();
        label_3->raise();
        label_4->raise();
        doublespinsw->raise();
        doublespinfield->raise();
        label_5->raise();
        label_6->raise();
        label_7->raise();
        label_8->raise();
        pushbrowse->raise();
        lineeditpath->raise();
        lineeditout->raise();
        label_9->raise();
        pushbrowse2->raise();
        progressBar->raise();
        label_2->raise();
        spinBox_seed->raise();
        spinBox_niter->raise();
        label_10->raise();
        done_label->raise();

        retranslateUi(poissonqtapp);

        QMetaObject::connectSlotsByName(poissonqtapp);
    } // setupUi

    void retranslateUi(QWidget *poissonqtapp)
    {
        poissonqtapp->setWindowTitle(QApplication::translate("poissonqtapp", "NUSscore", 0, QApplication::UnicodeUTF8));
        checkct1->setText(QApplication::translate("poissonqtapp", "Constant time", 0, QApplication::UnicodeUTF8));
        pushok->setText(QApplication::translate("poissonqtapp", "OK", 0, QApplication::UnicodeUTF8));
        pushcancel->setText(QApplication::translate("poissonqtapp", "Cancel", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("poissonqtapp", "1/T2*\n"
"(s-1)", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("poissonqtapp", "NUS", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("poissonqtapp", "TD", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("poissonqtapp", "SW\n"
"(ppm)", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("poissonqtapp", "Obs Freq\n"
"(MHz)", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("poissonqtapp", "Path to dpoisson7:", 0, QApplication::UnicodeUTF8));
        pushbrowse->setText(QApplication::translate("poissonqtapp", "Browse", 0, QApplication::UnicodeUTF8));
        lineeditpath->setText(QApplication::translate("poissonqtapp", "dpoisson7", 0, QApplication::UnicodeUTF8));
        lineeditpath->setPlaceholderText(QApplication::translate("poissonqtapp", "dpoisson7", 0, QApplication::UnicodeUTF8));
        lineeditout->setText(QString());
        lineeditout->setPlaceholderText(QString());
        label_9->setText(QApplication::translate("poissonqtapp", "Full path to output:", 0, QApplication::UnicodeUTF8));
        pushbrowse2->setText(QApplication::translate("poissonqtapp", "Browse", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("poissonqtapp", "seed(0=random)", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("poissonqtapp", "# of schemes", 0, QApplication::UnicodeUTF8));
        done_label->setText(QApplication::translate("poissonqtapp", "nuslists generated and scored!", 0, QApplication::UnicodeUTF8));
        pushcancel_2->setText(QApplication::translate("poissonqtapp", "Stop", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class poissonqtapp: public Ui_poissonqtapp {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_POISSONQTAPP_H
