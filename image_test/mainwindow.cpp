#include <Windows.h>
#include "Psapi.h"

#include "mainwindow.h"

#define args_nn_budget 100
#define args_max_cosine_distance 0.2
#define args_min_confidence 0.3
#define args_nms_max_overlap 1.0

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    lblImage = new QLabel();
    lblImage->setMinimumWidth(640);
    lblImage->setMinimumHeight(480);
    btnLoad = new QPushButton("load");
    connect(btnLoad, SIGNAL(clicked()), this, SLOT(load()));
    btnNext = new QPushButton("next");
    connect(btnNext, SIGNAL(clicked()), this, SLOT(next()));
    btnPlay = new QPushButton("play");
    connect(btnPlay, SIGNAL(clicked()), this, SLOT(play()));
    btnStop = new QPushButton("stop");
    connect(btnStop, SIGNAL(clicked()), this, SLOT(stop()));
    btnClear = new QPushButton("clear");
    connect(btnClear, SIGNAL(clicked()), this, SLOT(clear()));
    lblMemUse = new QLabel();
    QLabel *lbl00 = new QLabel("Memory Usage (bytes): ");
    lblFrame = new QLabel();
    QLabel *lbl01 = new QLabel("Current Frame: ");
    QWidget *panel = new QWidget();
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(lblImage,   0, 0, 1, 6);
    layout->addWidget(lbl00,      5, 0, 1, 1);
    layout->addWidget(lblMemUse,  5, 1, 1, 1);
    layout->addWidget(lbl01,      5, 2, 1, 1);
    layout->addWidget(lblFrame,   5, 3, 1, 1);
    layout->addWidget(btnLoad,    6, 0, 1, 1);
    layout->addWidget(btnNext,    6, 1, 1, 1);
    layout->addWidget(btnPlay,    6, 2, 1, 1);
    layout->addWidget(btnStop,    6, 3, 1, 1);
    layout->addWidget(btnClear,   6, 4, 1, 1);
    panel->setLayout(layout);
    setCentralWidget(panel);

    runner = new Runner(this);
    connect(runner, SIGNAL(frameNext()), this, SLOT(next()));

    model = new FeatureModel();
    mytracker = new tracker((float)args_max_cosine_distance, args_nn_budget);
    loader = new FeatureModelLoader(model);
    loader->saved_model_dir = saved_model_dir;
    loader->pct_gpu_mem = 0.2;
    waitBox = new WaitBox(this);
    waitBox->setWindowTitle("Feature Model Loader");
    connect(loader, SIGNAL(done(int)), waitBox, SLOT(done(int)));

    lblMemUse->setText(getMemoryUsage());
}

void MainWindow::clear()
{
    std::cout << "MainWindow::clear" << std::endl;
    image_filenames.clear();
    std::map<int, std::vector<Bbox>>::iterator it = object_detections.begin();
    while(it != object_detections.end()) {
        it->second.clear();
        it++;
    }
    object_detections.clear();
    model->clear();
    current_index = -1;
    lblMemUse->setText(getMemoryUsage());
}

void MainWindow::loadDetections(const QString& filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        std::cout << "File open error: " << file.errorString().toStdString() << std::endl;
        return;
    }

    while (!file.atEnd()) {
        QByteArray line = file.readLine();
        QList<QByteArray> elements = line.split(',');
        bool ok;
        int index = (int)QString(elements[0]).toFloat(&ok);
        if (ok) {
            //std::cout << "index: " << index << " size: " << detections[index].size() << std::endl;
            Bbox box;
            box.x1 = QString(elements[2]).toFloat();
            box.y1 = QString(elements[3]).toFloat();
            box.w = QString(elements[4]).toFloat();
            box.h = QString(elements[5]).toFloat();
            box.confidence = QString(elements[6]).toFloat();
            box.setX2Y2();
            object_detections[index-1].push_back(box);
        }
        else {
            std::cout << "failed to get int: " << QString(elements[0]).toStdString() << std::endl;
        }
    }
}

void MainWindow::stop()
{
    runner->running = false;
}

void MainWindow::play()
{
    std::cout << "MainWindow::play" << std::endl;
    QThreadPool::globalInstance()->tryStart(runner);
}

void MainWindow::next()
{
    current_index++;
    if (current_index < image_filenames.size()) {
        std::cout << image_filenames[current_index].toStdString() << std::endl;
        lblFrame->setText(image_filenames[current_index]);
        QString str = image_pathname + "\\" + image_filenames[current_index];
        image = cv::imread(str.toStdString());

        ImageFrame frame;
        for (int i = 0; i < object_detections[current_index].size(); i++) {
            cv::Mat crop = frame.getCrop(image, &object_detections[current_index][i]);
            frame.crops.push_back(crop);
            frame.detections.push_back(object_detections[current_index][i]);
        }
        model->run(&frame);

        std::cout << "current_index: " << current_index << std::endl;
        std::cout << "detections size: " << frame.detections.size() << std::endl;
        std::cout << "crops size: " << frame.crops.size() << std::endl;
        std::cout << "features size: " << frame.features.size() << std::endl;

        DETECTIONS detections;
        for (int i = 0; i < frame.detections.size(); i++) {
            DETECTION_ROW tmpRow;
            float x = frame.detections[i].x1;
            float y = frame.detections[i].y1;
            float w = frame.detections[i].w;
            float h = frame.detections[i].h;
            tmpRow.tlwh = DETECTBOX(x, y, w, h);
            tmpRow.confidence = frame.detections[i].confidence;
            for (int j = 0; j < feature_size; j++) {
                tmpRow.feature[j] = frame.features[i][j];
            }
            detections.push_back(tmpRow);
        }

        ModelDetection::getInstance()->dataMoreConf((float)args_min_confidence, detections);
        ModelDetection::getInstance()->dataPreprocessing(args_nms_max_overlap, detections);

        mytracker->predict();
        mytracker->update(detections);

        std::vector<RESULT_DATA> result;
        for (Track& track : mytracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
        }

        for (int i = 0; i < result.size(); i++) {
            DETECTBOX box = result[i].second;
            cv::Rect rect = cv::Rect(box[0], box[1], box[2], box[3]);
            cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 1);
            cv::putText(image, QString::number(result[i].first).toStdString(), cv::Point(rect.x, rect.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }


        lblImage->setPixmap(QPixmap::fromImage(QImage(image.data, image.cols, image.rows, QImage::Format_BGR888)));
    }
    else {
        current_index = 0;
    }
    lblMemUse->setText(getMemoryUsage());
}

void MainWindow::loadFilenames(const QString& dir)
{
    QDir image_dir(dir);
    image_filenames = image_dir.entryList(QDir::Files, QDir::Name);
}

void MainWindow::load()
{
    loadFilenames(image_pathname);
    loadDetections(detections_filename);

    QThreadPool::globalInstance()->tryStart(loader);
    waitBox->exec();

    lblMemUse->setText(getMemoryUsage());
}

QString MainWindow::getMemoryUsage() const

{
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    QString memory = QString::number(pmc.WorkingSetSize);
    addCommas(&memory);
    return memory;
}

void MainWindow::addCommas(QString *arg) const
{
    int i = arg->length();
    if(i >0)
    {
       i -= 3;
       while(i > 0)
       {
         arg->insert(i, ',');
         i -= 3;
       }
    }
}

MainWindow::~MainWindow()
{
}

Runner::Runner(QMainWindow *parent)
{
    mainWindow = parent;
    setAutoDelete(false);
}

void Runner::run()
{
    running = true;
    MainWindow *mw = (MainWindow*)mainWindow;
    if (mw->current_index < 0)
        mw->current_index = 0;
    while (mw->current_index < mw->image_filenames.size() && running) {
        QThread::msleep(300);
        emit frameNext();
    }
}

































