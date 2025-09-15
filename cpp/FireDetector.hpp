#ifndef FIREDETECTOR_HPP
#define FIREDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <deque>

// 使用 double-precision 以匹配 Python/Numpy 行为
using RectD = cv::Rect2d;
using PointD = cv::Point2d;

// 对应 Python 中的 NmsBBoxInfo
struct NmsBBoxInfo {
    double score;
    int classID;
    RectD box;
};

// 对应 Python 中的 Fire 类
class Fire {
public:
    RectD fire_box;
    double score;
    bool matched;

    std::vector<std::pair<double, PointD>> point_queue;

    PointD center_point;
    bool queue_valid_flag;
    int non_zero_num;
    int non_outlier_num;
    bool alarm_flag;

    Fire(RectD box, double s);
};

// fire_locate 返回的结果
struct FireLocateResult {
    int shape_id;
    PointD coord;
    double weight;
    cv::Mat vis_img;
};

// outlier_filter 返回的结果
struct OutlierFilterResult {
    bool valid_flag;
    PointD weighted_avg;
    int non_zero_num;
    int non_outlier_num;
};

// detect_fire 的最终返回结果
struct DetectFireResult {
    std::vector<Fire> warning_boxes;
    std::vector<Fire> pre_fire_boxes;
};

// 主检测器类
class FireDetector {
public:
    FireDetector();

    DetectFireResult detect_fire_frame(
            const std::vector<NmsBBoxInfo> &results,
            const cv::Mat &img_rgb,
            const std::vector<Fire> &pre_fire_boxes_in,
            const RectD &std_coord,
            int path_idx = 0
    );

private:
    const int W = 1920, H = 1920;
    const int QUEUE_MAX_LEN = 10;
    const std::vector<double> SHAPE_SCORES = {0.72, 0.9, 0.7, 0.5, 0.3, 0.1};

    // --- Helper Functions ---
    std::pair<std::vector<NmsBBoxInfo>, std::vector<NmsBBoxInfo>>
    filter_firein_tungsten(const std::vector<NmsBBoxInfo> &detect_boxes);

    std::vector<Fire> filter_iou(std::vector<Fire> fire_list);

    FireLocateResult fire_locate(const cv::Mat &im, const RectD &bbox, const RectD &ext_xxyy, int path_idx);

    FireLocateResult shape_process(const std::vector<int> &span_list, const cv::Mat &im, const cv::Rect &xxyy,
                                   const std::vector<int> &left_zeros, const RectD &ext_xxyy, int path_idx);

    std::pair<int, int>
    refine_bbox(const cv::Mat &im, const cv::Rect &xyxy, int thresh, int up_tol = 3, int down_tol = 2);

    cv::Mat cal_rb(const cv::Mat &im);

    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
    count_high_rg_pixels_per_row(const cv::Mat &crop, int thresh);

    std::pair<double, int> is_circle(const std::vector<int> &lst);

    std::pair<double, int> is_short(const std::vector<int> &lst);

    std::tuple<double, int, double> is_funnel(const std::vector<int> &lst);

    std::vector<int> exponential_smoothing(const std::vector<int> &span_list, double alpha = 0.8);

    std::pair<int, double> find_most_significant_valley(const std::vector<int> &span_list);

    std::pair<double, int> is_diamond(const std::vector<int> &lst);

    std::pair<double, int> is_rectangle(const std::vector<int> &lst);

    OutlierFilterResult
    outlier_filter(const std::vector<std::pair<double, PointD>> &res, std::pair<int, int> min_valid_num,
                   double threshold = 2.5, int max_outlier_num = 3);

    std::vector<int> find_outliers(const std::vector<PointD> &points, double threshold, int max_outlier_num);

    static double calculate_iou(const RectD &box1, const RectD &box2);

    static std::vector<NmsBBoxInfo> merge_rects(const std::vector<NmsBBoxInfo> &boxes);
};

#endif // FIREDETECTOR_HPP