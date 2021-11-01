QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DeepSort/errmsg.cpp \
    DeepSort/feature/model.cpp \
    DeepSort/matching/kalmanfilter.cpp \
    DeepSort/matching/linear_assignment.cpp \
    DeepSort/matching/nn_matching.cpp \
    DeepSort/matching/track.cpp \
    DeepSort/matching/tracker.cpp \
    DeepSort/thirdPart/hungarianoper.cpp \
    DeepSort/thirdPart/munkres/adapters/adapter.cpp \
    DeepSort/thirdPart/munkres/adapters/boostmatrixadapter.cpp \
    DeepSort/thirdPart/munkres/munkres.cpp \
    Utilities/directorysetter.cpp \
    Utilities/filesetter.cpp \
    Utilities/waitbox.cpp \
    featuremodel.cpp \
    imageframe.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    DeepSort/errmsg.h \
    DeepSort/feature/dataType.h \
    DeepSort/feature/model.h \
    DeepSort/matching/kalmanfilter.h \
    DeepSort/matching/linear_assignment.h \
    DeepSort/matching/nn_matching.h \
    DeepSort/matching/track.h \
    DeepSort/matching/tracker.h \
    DeepSort/thirdPart/hungarianoper.h \
    DeepSort/thirdPart/munkres/adapters/adapter.h \
    DeepSort/thirdPart/munkres/adapters/boostmatrixadapter.h \
    DeepSort/thirdPart/munkres/matrix.h \
    DeepSort/thirdPart/munkres/munkres.h \
    Utilities/directorysetter.h \
    Utilities/filesetter.h \
    Utilities/waitbox.h \
    featuremodel.h \
    imageframe.h \
    mainwindow.h

INCLUDEPATH += $$(CONTRIB_PATH)/include \
               $$(BOOST_PATH)

LIBS += -L$$(CONTRIB_PATH)/lib \
        -lopencv_world451 \
        -ltensorflow

RESOURCES += resources/resources.qrc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
