#ifndef FEATUREMODEL_H
#define FEATUREMODEL_H

#include <iostream>
#include <tensorflow/c/c_api.h>

#include <QMainWindow>
#include <QObject>
#include <QRunnable>
#include <QThreadPool>

#include "imageframe.h"

class FeatureModelLoader : public QObject, public QRunnable
{
    Q_OBJECT

public:
    FeatureModelLoader(QObject *featureModel);
    void run() override;

    QObject *featureModel;
    QString saved_model_dir;
    double pct_gpu_mem;

signals:
    void done(int);

};

class FeatureModel : public QObject
{
    Q_OBJECT

public:
    FeatureModel();
    ~FeatureModel();
    bool load(const QString& saved_model_dir, double pct_gpu_mem = 0.2);
    bool run(ImageFrame* frame);
    void clear();

    TF_SessionOptions* CreateSessionOptions(double perecentage);
    static void NoOpDeallocator(void *data, size_t a, void *b) { }

    bool initialized = false;
    const char *tags = "serve";
    TF_Graph *Graph = nullptr;
    TF_Status *Status = nullptr;
    TF_SessionOptions *SessionOpts = nullptr;
    TF_Buffer *RunOpts = nullptr;
    TF_Session *Session = nullptr;
    int ntags = 1;

    TF_Output *Input = nullptr;
    TF_Output *Output = nullptr;
    TF_Tensor **InputValues = nullptr;
    TF_Tensor **OutputValues = nullptr;
    int NumInputs = 1;
    int NumOutputs = 1;

};

#endif // FEATUREMODEL_H
