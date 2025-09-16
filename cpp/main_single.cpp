#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

#include "FireDetector.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    // --- 单个文件测试配置 ---
    const std::string IMAGE_PATH = "/home/manu/nfs/visi_1757316682/visi_1757316682_01005.png";
    const std::string DETECTION_TXT_PATH = "/home/manu/tmp/detections_cache_v1/visi_1757316682_01005.txt";
    const std::string OUTPUT_FILE = "/home/manu/tmp/single_output_cpp.txt";

    // --- 从原始脚本保留的初始化代码 ---
    FireDetector detector;
    std::vector<cv::Point> std_pts_vec = {{572, 150},
                                          {648, 147},
                                          {568, 162},
                                          {651, 160}};
    const int W = 1280, H = 720;

    cv::Rect roi = cv::boundingRect(std_pts_vec);
    double expand_exclude_ratio = 0.5;
    double sx1 = std::max(0.0, roi.x - roi.width * expand_exclude_ratio);
    double sy1 = std::max(0.0, roi.y - roi.height * expand_exclude_ratio);
    double sw = std::min(roi.width * (1 + 2 * expand_exclude_ratio), W - sx1);
    double sh = std::min(roi.height * (1 + 2 * expand_exclude_ratio), H - sy1);
    RectD std_coord(sx1, sy1, sw, sh);

    // 初始化 detect_fire 函数所需的状态变量
    std::vector<Fire> pre_fire_boxes;

    // --- 针对单个文件的处理逻辑 ---
    int img_idx = 0; // 对于单张图片，索引可以设为0

    // 1. 读取图像文件
    cv::Mat img_bgr = cv::imread(IMAGE_PATH);
    if (img_bgr.empty()) {
        std::cerr << "错误: 无法在路径 " << IMAGE_PATH << " 找到或打开图像" << std::endl;
        return 1;
    }

    // 2. 从TXT文件加载检测结果
    std::vector<NmsBBoxInfo> results;
    fs::path detection_file(DETECTION_TXT_PATH);
    if (fs::exists(detection_file) && fs::file_size(detection_file) > 0) {
        std::ifstream ifs(detection_file);
        if (ifs.is_open()) {
            std::string line;
            while (std::getline(ifs, line)) {
                std::stringstream ss(line);
                std::vector<double> row;
                double val;
                while (ss >> val) row.push_back(val);
                if (row.size() >= 6) {
                    results.push_back({
                                              row[4],
                                              static_cast<int>(row[5]),
                                              {row[0], row[1], row[2] - row[0], row[3] - row[1]}
                                      });
                }
            }
        } else {
            std::cerr << "警告: 无法打开检测文件 " << DETECTION_TXT_PATH << std::endl;
        }
    } else if (!fs::exists(detection_file)) {
        std::cout << "警告: 未找到检测文件: " << DETECTION_TXT_PATH << std::endl;
    }

    cv::Mat img_rgb;
    cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    // 3. 调用核心的火焰检测逻辑
    DetectFireResult detection_output = detector.detect_fire_frame(results, img_rgb, pre_fire_boxes, std_coord,
                                                                   img_idx);

    // 4. 打印结果并写入输出文件
    std::cout << "---------" << std::endl;
    std::cout << "图像: " << fs::path(IMAGE_PATH).filename().string() << std::endl;

    // 从日志中提取最终结果摘要进行打印
    std::string result_summary = "No result summary found.";
    size_t summary_pos = detection_output.log_str.rfind("Found");
    if (summary_pos != std::string::npos) {
        size_t end_of_line = detection_output.log_str.find('\n', summary_pos);
        if (end_of_line != std::string::npos) {
            result_summary = detection_output.log_str.substr(summary_pos, end_of_line - summary_pos);
        }
    }
    std::cout << "检测结果: " << result_summary << std::endl;
    std::cout << "log_str --> " << detection_output.log_str << std::endl;
    std::cout << "---------" << std::endl;

    std::ofstream f_out(OUTPUT_FILE);
    if (f_out.is_open()) {
        f_out << img_idx << "\t" << fs::path(IMAGE_PATH).filename().string() << "\t" << detection_output.log_str;
        f_out.close();
        std::cout << "结果已成功保存到: " << OUTPUT_FILE << std::endl;
    } else {
        std::cerr << "错误: 无法打开输出文件 " << OUTPUT_FILE << std::endl;
        return 1;
    }

    return 0;
}