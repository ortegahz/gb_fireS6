#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <iomanip>

#include "FireDetector.hpp"

namespace fs = std::filesystem;

// 从文件名中提取数字用于排序
int extract_number(const std::string &filename) {
    std::regex re("_(\\d+)\\.");
    std::smatch match;
    if (std::regex_search(filename, match, re) && match.size() > 1) {
        try { return std::stoi(match.str(1)); }
        catch (...) { return 0; }
    }
    return 0;
}

// 格式化输出，与Python版本匹配
std::string format_warning_boxes(const std::vector<Fire> &warning_boxes) {
    if (warning_boxes.empty()) return "[]";
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "[";
    for (size_t i = 0; i < warning_boxes.size(); ++i) {
        const auto &fire = warning_boxes[i];
        ss << "<fire_box: (" << fire.fire_box.x << ", " << fire.fire_box.y << ", "
           << fire.fire_box.width << ", " << fire.fire_box.height << "), "
           << "score: " << fire.score << ", "
           << "center: (" << fire.center_point.x << ", " << fire.center_point.y << ")>";
        if (i < warning_boxes.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

int main() {
    // --- 配置 ---
    const std::string DETECTION_CACHE_DIR = "/home/manu/tmp/detections_cache";
    const std::string OUTPUT_FILE = "/home/manu/tmp/output_gb_s6.txt";
    const std::string IMG_FOLDER_PATH = "/home/manu/nfs/2p";

    // --- 初始化 ---
    FireDetector detector;

    // 初始化九点区域
    std::vector<cv::Point> std_pts_vec = {
            {828, 310},
            {885, 310},
            {945, 310},
            {826, 320},
            {886, 319},
            {946, 318},
            {826, 330},
            {886, 328},
            {950, 331}
    };
    const int W = 1920, H = 1080;
    cv::Rect roi = cv::boundingRect(std_pts_vec);
    cv::Rect2f std_coord;
    std_coord.x = std::max(0.0f, roi.x - roi.width * 0.5f);
    std_coord.y = std::max(0.0f, roi.y - roi.height * 0.5f);
    std_coord.width = std::min((float) roi.width * 2.0f, (float) W - std_coord.x);
    std_coord.height = std::min((float) roi.height * 2.0f, (float) H - std_coord.y);

    // 获取并排序文件列表
    std::vector<fs::path> file_list;
    if (fs::exists(IMG_FOLDER_PATH) && fs::is_directory(IMG_FOLDER_PATH)) {
        for (const auto &entry: fs::directory_iterator(IMG_FOLDER_PATH)) {
            if (entry.is_regular_file()) file_list.push_back(entry.path());
        }
    }
    std::sort(file_list.begin(), file_list.end(), [](const fs::path &a, const fs::path &b) {
        return extract_number(a.filename().string()) < extract_number(b.filename().string());
    });

    // --- 主循环 ---
    std::ofstream f_out(OUTPUT_FILE);
    int img_idx = 0;
    std::vector<Fire> pre_fire_boxes;

    for (const auto &image_path: file_list) {
        // 读取缓存的检测结果
        fs::path cache_file = fs::path(DETECTION_CACHE_DIR) / (image_path.stem().string() + ".txt");
        std::vector<NmsBBoxInfo> results;
        if (fs::exists(cache_file)) {
            std::ifstream ifs(cache_file);
            std::string line;
            while (std::getline(ifs, line)) {
                std::stringstream ss(line);
                std::vector<float> row;
                float val;
                while (ss >> val) row.push_back(val);
                if (row.size() >= 6) {
                    results.push_back(
                            {row[4], static_cast<int>(row[5]), {row[0], row[1], row[2] - row[0], row[3] - row[1]}});
                }
            }
        } else {
            std::cerr << "Warning: Cache file not found: " << cache_file << std::endl;
        }

        // 读取图像
        cv::Mat img_bgr = cv::imread(image_path.string());
        if (img_bgr.empty()) {
            std::cerr << "Warning: Could not read image: " << image_path << std::endl;
            continue;
        }
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        // 调用检测逻辑
        DetectFireResult detection_output = detector.detect_fire_frame(results, img_rgb, pre_fire_boxes, std_coord,
                                                                       img_idx);
        pre_fire_boxes = detection_output.pre_fire_boxes; // 更新状态

        // 输出结果
        std::string warning_str = format_warning_boxes(detection_output.warning_boxes);
        std::cout << img_idx << " --------- " << warning_str << std::endl;
        f_out << img_idx << "\t" << image_path.filename().string() << "\t" << warning_str << "\n";

        img_idx++;
    }

    f_out.close();
    return 0;
}