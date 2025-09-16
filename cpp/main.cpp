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

int extract_number(const std::string &filename) {
    std::regex re("_(\\d+)\\.");
    std::smatch match;
    if (std::regex_search(filename, match, re) && match.size() > 1) {
        try { return std::stoi(match.str(1)); }
        catch (...) { return 0; }
    }
    return 0;
}

std::string format_point_queue(const std::vector<std::pair<float, PointD>> &queue) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(5);
    ss << "[";
    for (size_t i = 0; i < queue.size(); ++i) {
        ss << "(" << queue[i].first << ", (" << queue[i].second.x << ", " << queue[i].second.y << "))";
        if (i < queue.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

std::string format_fire_object(const Fire &fire) {
    std::stringstream ss;
    ss << std::fixed;
    ss << "Fire(fire_box=" << std::setprecision(17) << "(" << fire.fire_box.x << ", " << fire.fire_box.y << ", "
       << fire.fire_box.width << ", " << fire.fire_box.height << ")"
       << ", score=" << std::setprecision(17) << fire.score
       // <-- 新增 fire_point 的输出
       << ", fire_point=(" << std::setprecision(1) << fire.fire_point.x << ", " << fire.fire_point.y << ")"
       << ", center_point=(" << std::setprecision(10) << fire.center_point.x << ", " << fire.center_point.y << ")"
       << ", matched=" << (fire.matched ? "True" : "False")
       << ", point_queue=" << format_point_queue(fire.point_queue)
       << ", queue_valid_flag=" << (fire.queue_valid_flag ? "True" : "False")
       << ", alarm_flag=" << (fire.alarm_flag ? "True" : "False")
       << ", non_zero_num=" << fire.non_zero_num
       << ", non_outlier_num=" << fire.non_outlier_num
       << ")";
    return ss.str();
}

std::string format_warning_boxes(const std::vector<Fire> &warning_boxes) {
    if (warning_boxes.empty()) return "[]";
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < warning_boxes.size(); ++i) {
        ss << format_fire_object(warning_boxes[i]);
        if (i < warning_boxes.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

int main(int argc, char *argv[]) {
    const std::string DETECTION_CACHE_DIR = "/home/manu/tmp/detections_cache_v0";
    const std::string OUTPUT_FILE = "/home/manu/tmp/output_gb_s6_cpp.txt";
    const std::string IMG_FOLDER_PATH = "/home/manu/nfs/visi_1757382127";

    FireDetector detector;

    std::vector<cv::Point> std_pts_vec = {{768, 291},
                                          {892, 308}};

    const int W = 1920, H = 1080;

    cv::Rect roi = cv::boundingRect(std_pts_vec);
    double expand_exclude_ratio = 0.5;
    double sx1 = std::max(0.0, roi.x - roi.width * expand_exclude_ratio);
    double sy1 = std::max(0.0, roi.y - roi.height * expand_exclude_ratio);
    double sw = std::min(roi.width * (1 + 2 * expand_exclude_ratio), W - sx1);
    double sh = std::min(roi.height * (1 + 2 * expand_exclude_ratio), H - sy1);
    RectD std_coord(sx1, sy1, sw, sh);

    std::vector<fs::path> file_list;
    if (fs::exists(IMG_FOLDER_PATH) && fs::is_directory(IMG_FOLDER_PATH)) {
        for (const auto &entry: fs::directory_iterator(IMG_FOLDER_PATH)) {
            if (entry.is_regular_file()) file_list.push_back(entry.path());
        }
    }
    std::sort(file_list.begin(), file_list.end(), [](const fs::path &a, const fs::path &b) {
        return extract_number(a.filename().string()) < extract_number(b.filename().string());
    });

    std::ofstream f_out(OUTPUT_FILE);
    int img_idx = 0;
    std::vector<Fire> pre_fire_boxes;

    for (const auto &image_path: file_list) {
        fs::path cache_file = fs::path(DETECTION_CACHE_DIR) / (image_path.stem().string() + ".txt");
        std::vector<NmsBBoxInfo> results;
        if (fs::exists(cache_file)) {
            std::ifstream ifs(cache_file);
            std::string line;
            while (std::getline(ifs, line)) {
                std::stringstream ss(line);
                std::vector<double> row;
                double val;
                while (ss >> val) row.push_back(val);
                if (row.size() >= 6) {
                    results.push_back(
                            {row[4], static_cast<int>(row[5]), {row[0], row[1], row[2] - row[0], row[3] - row[1]}});
                }
            }
        }

        cv::Mat img_bgr = cv::imread(image_path.string());
        if (img_bgr.empty()) {
            std::cerr << "Warning: Could not read image: " << image_path << std::endl;
            continue;
        }
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        DetectFireResult detection_output = detector.detect_fire_frame(results, img_rgb, pre_fire_boxes, std_coord,
                                                                       img_idx);
        pre_fire_boxes = detection_output.pre_fire_boxes;

        std::string warning_str = format_warning_boxes(detection_output.warning_boxes);
        f_out << img_idx << "\t" << image_path.filename().string() << "\t" << warning_str << "\n";

        img_idx++;
    }

    f_out.close();
    return 0;
}