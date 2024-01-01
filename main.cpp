#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

const std::string CURRENT_PATH = "C:\\Users\\ben\\Desktop\\cpp\\CV Projects\\ssd\\";
const std::string MODEL_PATH = CURRENT_PATH + "MobileNet\\MobileNetSSD_deploy.caffemodel";
const std::string TXT_PATH = CURRENT_PATH + "MobileNet\\MobileNetSSD_deploy.prototxt.txt";
const std::string IMAGES_PATH = CURRENT_PATH + "images";
const std::string OUTPUT_PATH = CURRENT_PATH + "outputs";

const std::vector<std::string> CLASSES = {
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
};

cv::dnn::Net loadModel(const std::string& txtPath, const std::string& modelPath) {
    return cv::dnn::readNetFromCaffe(txtPath, modelPath);
}

void drawBoundingBoxes(cv::Mat& img, const cv::Mat& detections, const std::vector<cv::Scalar>& colors) {
    int h = img.rows;
    int w = img.cols;

    for (int i = 0; i < detections.size[2]; ++i) {
        auto* data = detections.ptr<float>(0, 0, i);
        float confidence = data[2];

        if (confidence > 0.10) {
            int idx = static_cast<int>(data[1]);
            cv::Rect box(
                    static_cast<int>(data[3] * w),
                    static_cast<int>(data[4] * h),
                    static_cast<int>(data[5] * w),
                    static_cast<int>(data[6] * h)
            );

            cv::rectangle(img, box, colors[idx], 2);
            cv::putText(img, CLASSES[idx] + ": " + std::to_string(confidence), cv::Point(box.x, box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2);
        }
    }

    cv::imshow("ssd", img);
}

void processImage(cv::Mat& img, cv::dnn::Net& model, const std::vector<cv::Scalar>& colors) {
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(300, 300));

    cv::Mat blob = cv::dnn::blobFromImage(imgResized, 0.007843, cv::Size(300, 300), 127.5);

    model.setInput(blob);
    cv::Mat detections = model.forward();

    drawBoundingBoxes(img, detections, colors);
}

void detectObjects(const std::string& txtPath, const std::string& modelPath, bool isRealTime = false) {
    cv::dnn::Net model = loadModel(txtPath, modelPath);

    std::vector<cv::Scalar> colors(CLASSES.size());
    cv::RNG rng(12345);
    for (int i = 0; i < colors.size(); i++) {
        colors[i] = rng.uniform(0, 255);
    }

    if (isRealTime) {
        cv::VideoCapture cap(0);

        while (true) {
            cv::Mat frame;
            cap.read(frame);

            processImage(frame, model, colors);

            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } else {
        for (const auto& file : fs::directory_iterator(IMAGES_PATH)) {
            std::string imgPath = file.path().string();
            std::string imgName = file.path().filename().stem().string();

            cv::Mat img = cv::imread(imgPath);

            if (!img.empty()) {
                processImage(img, model, colors);
                std::string outputFileName = OUTPUT_PATH + "\\" + imgName + "_output.jpg";
                cv::imwrite(outputFileName, img);
            }

            if (cv::waitKey(0) == 'q') {
                continue;
            }
        }
    }
}

int main() {
    detectObjects(TXT_PATH, MODEL_PATH);
    return 0;
}
