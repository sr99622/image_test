#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QGridLayout>
#include <QDir>
#include <QRunnable>
#include <QThreadPool>

#include "featuremodel.h"
#include "DeepSort/feature/model.h"
#include "DeepSort/matching/tracker.h"
#include "Utilities/waitbox.h"

class Runner : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Runner(QMainWindow *parent);
    void run() override;
    bool running;

    QMainWindow *mainWindow;

signals:
    void frameNext();

};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    QString getMemoryUsage() const;
    void addCommas(QString *arg) const;
    void loadFilenames(const QString& dir);
    void loadDetections(const QString& filename);
    ~MainWindow();

    int current_index = -1;

    Runner *runner;
    cv::Mat image;

    QLabel *lblImage;
    QPushButton *btnLoad;
    QPushButton *btnNext;
    QPushButton *btnPlay;
    QPushButton *btnStop;
    QPushButton *btnClear;
    QLabel *lblMemUse;
    QLabel *lblFrame;
    FeatureModelLoader *loader;
    WaitBox *waitBox;

    FeatureModel *model;
    tracker *mytracker;

    QStringList image_filenames;
    std::map<int, std::vector<Bbox>> object_detections;
    QString image_pathname = "C:\\Users\\sr996\\Projects\\deep_sort\\MOT16\\test\\MOT16-06\\img1";
    QString saved_model_dir = "C:\\Users\\sr996\\Projects\\deep_sort\\saved_model";
    QString detections_filename = "C:\\Users\\sr996\\Projects\\deep_sort\\MOT16\\test\\MOT16-06\\det\\det.txt";

public slots:
    void load();
    void next();
    void play();
    void stop();
    void clear();

};
#endif // MAINWINDOW_H
