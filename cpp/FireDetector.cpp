#include "FireDetector.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace {
    // 辅助函数：将 RectD 格式化为 Python 元组样式
    std::string format_rect(const RectD &rect) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(17)
           << "(" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << ")";
        return ss.str();
    }

    // 辅助函数：将 PointD 格式化为 Python list 样式 (for outlier_filter)
    std::string format_point_as_list(const PointD &point) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << "[" << point.x << ". " << point.y << ".]";
        return ss.str();
    }
}

Fire::Fire(RectD box, float s)
        : fire_box(box), score(s), fire_point(0.0, 0.0), matched(true), center_point(0, 0),
          queue_valid_flag(false), non_zero_num(0), non_outlier_num(0), alarm_flag(false) {}

FireDetector::FireDetector() {}

DetectFireResult FireDetector::detect_fire_frame(const std::vector<NmsBBoxInfo> &results, const cv::Mat &img_rgb,
                                                 const std::vector<Fire> &pre_fire_boxes_in, const RectD &std_coord,
                                                 int path_idx) {
    std::stringstream log_ss;

    auto [filter_result_raw, tungsten_result] = filter_firein_tungsten(results);

    std::vector<NmsBBoxInfo> responding_labels;
    for (const auto &fr: filter_result_raw) {
        if (fr.classID == 0 && calculate_iou(fr.box, std_coord) > 1e-9) {
            responding_labels.push_back(fr);
        }
    }

    auto merged_labels = merge_rects(responding_labels);

    std::vector<NmsBBoxInfo> valid_boxes;
    for (auto &item: merged_labels) {
        RectD clipped_box = item.box & RectD(0, 0, W, H);
        if (clipped_box.width > 0 && clipped_box.height > 0) {
            item.box = clipped_box;
            valid_boxes.push_back(item);
        }
    }

    std::vector<Fire> pre_fire_boxes = pre_fire_boxes_in;
    for (auto &box: pre_fire_boxes) box.matched = false;

    std::vector<NmsBBoxInfo> cur_detections = valid_boxes;
    std::vector<bool> cur_matched_flags(cur_detections.size(), false);

    for (auto &pre_item: pre_fire_boxes) {
        int best_match_idx = -1;
        double best_iou = 0.0;
        for (size_t cur_idx = 0; cur_idx < cur_detections.size(); ++cur_idx) {
            if (cur_matched_flags[cur_idx]) continue;
            double iou = calculate_iou(pre_item.fire_box, cur_detections[cur_idx].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_match_idx = cur_idx;
            }
        }

        if (best_iou > 0.0) {
            const auto &cur_item = cur_detections[best_match_idx];
            pre_item.score = std::min(1.0, pre_item.score + std::max(0.18, (cur_item.score - 0.25) * 0.8));
            pre_item.fire_box = cur_item.box;
            pre_item.matched = true;
            cur_matched_flags[best_match_idx] = true;
        }
    }

    pre_fire_boxes.erase(std::remove_if(pre_fire_boxes.begin(), pre_fire_boxes.end(),
                                        [](const Fire &box) { return !box.matched && box.score - 0.05 < 1e-6; }),
                         pre_fire_boxes.end());
    for (auto &box: pre_fire_boxes) {
        if (!box.matched) {
            box.score -= 0.05;
        }
    }
    for (size_t i = 0; i < cur_detections.size(); ++i) {
        if (!cur_matched_flags[i]) {
            double score = std::max(0.15, (cur_detections[i].score - 0.25) / 2.0);
            pre_fire_boxes.emplace_back(cur_detections[i].box, score);
        }
    }

    log_ss << "--- Per-Fire Analysis (Total " << pre_fire_boxes.size() << " fires) ---\n";
    int fire_idx_counter = 0;
    for (auto &fire: pre_fire_boxes) {
        log_ss << "\n[Fire Object " << fire_idx_counter << "] Box: " << format_rect(fire.fire_box)
               << ", Score: " << std::fixed << std::setprecision(3) << fire.score
               << ", Matched: " << (fire.matched ? "(True,)" : "(False,)") << "\n";

        PointD coord_best(0.0, 0.0);
        double weight_best = 0.0;
        int shape_id_best = 5;

        if (fire.matched) {
            cv::Mat img_bgr;
            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            auto fire_loc_res = fire_locate(img_bgr, fire.fire_box, std_coord, path_idx);
            coord_best = fire_loc_res.coord;
            weight_best = fire_loc_res.weight;
            shape_id_best = fire_loc_res.shape_id;
            log_ss << "  - fire_locate (matched): coord="
                   << "[" << static_cast<int>(coord_best.x) << " " << static_cast<int>(coord_best.y) << "]"
                   << ", weight=" << std::fixed << std::setprecision(3) << weight_best
                   << ", shape_id=" << shape_id_best << "\n";

        } else {
            log_ss << "  - fire_locate (unmatched): placeholder coord=(0.0, 0.0)"
                   << ", weight=0.000"
                   << ", shape_id=None\n";
        }

        fire.point_queue.emplace_back(weight_best, coord_best);
        if (fire.point_queue.size() > (size_t) QUEUE_MAX_LEN) {
            fire.point_queue.erase(fire.point_queue.begin());
        }

        std::stringstream pq_ss;
        pq_ss << "[";
        for (size_t i = 0; i < fire.point_queue.size(); ++i) {
            pq_ss << "(" << std::fixed << std::setprecision(16) << fire.point_queue[i].first << ", ["
                  << static_cast<int>(fire.point_queue[i].second.x) << ", "
                  << static_cast<int>(fire.point_queue[i].second.y) << "])";
            if (i < fire.point_queue.size() - 1) pq_ss << ", ";
        }
        pq_ss << "]";

        log_ss << path_idx << "," << fire_idx_counter << "," << format_rect(fire.fire_box) << "," << pq_ss.str()
               << "\n";

        auto outlier_res = outlier_filter(fire.point_queue, {5, 4});
        fire.center_point = outlier_res.weighted_avg;
        fire.queue_valid_flag = outlier_res.valid_flag;
        fire.non_zero_num = outlier_res.non_zero_num;
        fire.non_outlier_num = outlier_res.non_outlier_num;
        log_ss << "  - outlier_filter result: valid_flag=" << (fire.queue_valid_flag ? "True" : "False")
               << ", center_point=" << format_point_as_list(fire.center_point)
               << ", non_zero_num=" << fire.non_zero_num
               << ", non_outlier_num=" << fire.non_outlier_num << "\n";

        fire.alarm_flag = fire.queue_valid_flag && fire.score > 0.5;
        bool score_check = fire.score > 0.5;
        log_ss << "  - Alarm Flag Calculation: queue_valid_flag(" << (fire.queue_valid_flag ? "True" : "False")
               << ") AND score(" << std::fixed << std::setprecision(3) << fire.score
               << ") > 0.5 (is " << (score_check ? "True" : "False")
               << ") ==> alarm_flag=" << (fire.alarm_flag ? "True" : "False") << "\n";

        fire_idx_counter++;
    }

    log_ss << "\n--- Final Adjustments ---\n[Tungsten Penalty]\n";
    for (auto &fire: pre_fire_boxes) {
        for (const auto &t_box: tungsten_result) {
            if (calculate_iou(fire.fire_box, t_box.box) >= 0.001) {
                float old_score = fire.score;
                fire.score -= 0.5;
                log_ss << "  - Fire at " << format_rect(fire.fire_box) << " penalized. Score "
                       << std::fixed << std::setprecision(3) << old_score
                       << " -> " << std::fixed << std::setprecision(3) << fire.score << ".\n";
                break;
            }
        }
    }

    log_ss << "[Final Filtering]\n";
    size_t count_before_score = pre_fire_boxes.size();
    log_ss << "  - Before score-based filtering: " << count_before_score << " fires.\n";
    pre_fire_boxes.erase(std::remove_if(pre_fire_boxes.begin(), pre_fire_boxes.end(),
                                        [](const Fire &box) { return box.score < 1e-6; }),
                         pre_fire_boxes.end());
    log_ss << "  - After score-based filtering (score >= 1e-6): " << pre_fire_boxes.size() << " fires remain.\n";

    size_t count_before_iou = pre_fire_boxes.size();
    log_ss << "  - Before IOU-based filtering: " << count_before_iou << " fires.\n";
    pre_fire_boxes = filter_iou(pre_fire_boxes);
    log_ss << "  - After IOU-based filtering: " << pre_fire_boxes.size() << " fires remain.\n";

    std::vector<Fire> warning_boxes;
    for (const auto &box: pre_fire_boxes) {
        if (box.alarm_flag) warning_boxes.push_back(box);
    }

    log_ss << "\n--- Result ---\nFound " << warning_boxes.size() << " warning boxes.\n";

    return {warning_boxes, pre_fire_boxes, log_ss.str()};
}

std::pair<std::vector<NmsBBoxInfo>, std::vector<NmsBBoxInfo>>
FireDetector::filter_firein_tungsten(const std::vector<NmsBBoxInfo> &detect_boxes) {
    std::vector<NmsBBoxInfo> fires, tungstens, filtered_fires;
    for (const auto &b: detect_boxes) {
        if (b.classID == 0) fires.push_back(b);
        else tungstens.push_back(b);
    }
    for (const auto &fire: fires) {
        bool is_inside = false;
        for (const auto &tungsten: tungstens) {
            if (calculate_iou(fire.box, tungsten.box) > 0.001) {
                is_inside = true;
                break;
            }
        }
        if (!is_inside) filtered_fires.push_back(fire);
    }
    return {filtered_fires, tungstens};
}

std::vector<Fire> FireDetector::filter_iou(std::vector<Fire> fire_list) {
    if (fire_list.empty()) return {};

    std::sort(fire_list.begin(), fire_list.end(), [](const Fire &a, const Fire &b) {
        return a.score > b.score;
    });

    std::vector<Fire> keep_list;
    std::vector<bool> suppressed(fire_list.size(), false);

    for (size_t i = 0; i < fire_list.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        keep_list.push_back(fire_list[i]);
        for (size_t j = i + 1; j < fire_list.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (calculate_iou(fire_list[i].fire_box, fire_list[j].fire_box) > 0.0) {
                suppressed[j] = true;
            }
        }
    }
    return keep_list;
}

OutlierFilterResult
FireDetector::outlier_filter(const std::vector<std::pair<float, PointD>> &res, std::pair<int, int> min_valid_num,
                             double threshold, int max_outlier_num) {
    OutlierFilterResult result = {false, {0.0, 0.0}, 0, 0};

    std::vector<std::pair<double, PointD>> res_valid;
    for (const auto &r: res) if (r.first != 0.0) res_valid.push_back(r);

    result.non_zero_num = res_valid.size();

    std::vector<PointD> points;
    for (const auto &r: res_valid) points.push_back(r.second);

    std::vector<int> outlier_indices = find_outliers(points, threshold, max_outlier_num);
    std::sort(outlier_indices.rbegin(), outlier_indices.rend());

    std::vector<std::pair<double, PointD>> cleaned_res = res_valid;
    for (int idx_to_remove: outlier_indices) {
        if (idx_to_remove < cleaned_res.size())
            cleaned_res.erase(cleaned_res.begin() + idx_to_remove);
    }
    result.non_outlier_num = cleaned_res.size();

    if (!cleaned_res.empty()) {
        PointD final_coord(0.0, 0.0);
        double total_conf = 0.0;
        for (const auto &cp: cleaned_res) {
            total_conf += cp.first;
            final_coord.x += cp.first * cp.second.x;
            final_coord.y += cp.first * cp.second.y;
        }

        if (total_conf > 1e-9) {
            result.weighted_avg = final_coord / total_conf;
        } else {
            result.weighted_avg = {0.0, 0.0};
        }
    } else {
        result.weighted_avg = {0.0, 0.0};
    }

    if ((int) result.non_zero_num >= min_valid_num.first && (int) result.non_outlier_num >= min_valid_num.second) {
        result.valid_flag = true;
    } else {
        result.valid_flag = false;
    }

    return result;
}

std::vector<int> FireDetector::find_outliers(const std::vector<PointD> &points, double threshold, int max_outlier_num) {
    if (points.empty()) return {};
    std::vector<int> all_outlier_indices;
    std::vector<PointD> remaining_points = points;
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);

    for (int k = 0; k < max_outlier_num && remaining_points.size() > 1; ++k) {
        cv::Scalar mean_scalar = cv::mean(remaining_points);
        PointD centroid(mean_scalar[0], mean_scalar[1]);

        double max_dist = -1.0;
        int outlier_idx_in_remaining = -1;
        for (size_t i = 0; i < remaining_points.size(); ++i) {
            double dist = cv::norm(remaining_points[i] - centroid);
            if (dist > max_dist) {
                max_dist = dist;
                outlier_idx_in_remaining = i;
            }
        }

        if (max_dist > threshold) {
            all_outlier_indices.push_back(original_indices[outlier_idx_in_remaining]);
            remaining_points.erase(remaining_points.begin() + outlier_idx_in_remaining);
            original_indices.erase(original_indices.begin() + outlier_idx_in_remaining);
        } else {
            break;
        }
    }
    return all_outlier_indices;
}

cv::Mat FireDetector::cal_rb(const cv::Mat &im) {
    if (im.empty() || im.channels() != 3) {
        return {};
    }
    cv::Mat result(im.rows, im.cols, CV_32F);
    for (int r = 0; r < im.rows; ++r) {
        const cv::Vec3b *p_im = im.ptr<cv::Vec3b>(r);
        float *p_res = result.ptr<float>(r);
        for (int c = 0; c < im.cols; ++c) {
            p_res[c] = (static_cast<float>(p_im[c][0]) + static_cast<float>(p_im[c][1])) * 0.5f;
        }
    }
    return result;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
FireDetector::count_high_rg_pixels_per_row(const cv::Mat &crop, int thresh) {
    if (crop.empty() || crop.cols == 0 || crop.rows == 0)
        return {{},
                {},
                {}};
    cv::Mat rb_avg = cal_rb(crop);
    cv::Mat mask = rb_avg >= (thresh == -1 ? 227 : thresh);

    std::vector<int> span_list, left_zeros, right_zeros;
    for (int i = 0; i < mask.rows; ++i) {
        cv::Mat row_mask = mask.row(i);
        std::vector<cv::Point> locations;
        cv::findNonZero(row_mask, locations);
        if (locations.empty()) {
            span_list.push_back(0);
            left_zeros.push_back(mask.cols);
            right_zeros.push_back(mask.cols);
        } else {
            int first = locations.front().x;
            int last = locations.back().x;
            for (const auto &pt: locations) {
                if (pt.x < first) first = pt.x;
                if (pt.x > last) last = pt.x;
            }
            span_list.push_back(last - first + 1);
            left_zeros.push_back(first);
            right_zeros.push_back(mask.cols - 1 - last);
        }
    }
    return {span_list, left_zeros, right_zeros};
}

std::pair<int, int>
FireDetector::refine_bbox(const cv::Mat &im, const cv::Rect &xyxy, int thresh, int up_tol, int down_tol) {
    int x1 = xyxy.x, y1 = xyxy.y, x2 = xyxy.x + xyxy.width, y2 = xyxy.y + xyxy.height;

    int y1_ex = 1, y1_ex_tol = 0;
    while (true) {
        if (y1 - y1_ex < 0 || y1_ex > 50) {
            y1_ex = y1_ex - 1 - y1_ex_tol;
            break;
        }
        cv::Rect roi_y1_ex_rect(x1, y1 - y1_ex, x2 - x1, y1_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y1_ex_rect), thresh);

        if (spans.empty() || spans.front() == 0) {
            if (++y1_ex_tol > up_tol) {
                y1_ex = y1_ex - 1 - y1_ex_tol;
                break;
            }
        } else {
            y1_ex_tol = 0;
        }
        y1_ex++;
    }

    int y2_ex = 1, y2_ex_tol = 0;
    while (true) {
        if (y2 + y2_ex >= im.rows || y2_ex > 50) {
            y2_ex = y2_ex - 1 - y2_ex_tol;
            break;
        }
        cv::Rect roi_y2_ex_rect(x1, y2, x2 - x1, y2_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y2_ex_rect), thresh);

        if (spans.empty() || spans.back() == 0) {
            if (++y2_ex_tol > down_tol) {
                y2_ex = y2_ex - 1 - y2_ex_tol;
                break;
            }
        } else {
            y2_ex_tol = 0;
        }
        y2_ex++;
    }
    return {y1_ex > 0 ? y1_ex : 0, y2_ex > 0 ? y2_ex : 0};
}

std::pair<double, int> FireDetector::is_circle(const std::vector<int> &lst) {
    if (lst.size() < 3) return {0.0, -1};
    int h = lst.size();
    if (h == 0) return {0.0, -1};
    int w = *std::max_element(lst.begin(), lst.end());
    if (w == 0 || (double) h / w > 1.5) return {0.0, -1};

    double lst_max = w;
    double lst_thresh = lst_max * 0.2;
    int cur_phase1 = lst.size() - 1;
    double last_max_value = -1.0;
    std::vector<double> err_phase1 = {0.0};
    for (size_t i = 0; i < lst.size(); ++i) {
        if (last_max_value < 0) {
            last_max_value = lst[i];
            err_phase1.push_back(0);
            continue;
        }
        if (lst[i] - last_max_value < -lst_thresh) {
            cur_phase1 = i - 1;
            break;
        }
        err_phase1.push_back(
                err_phase1.back() + (lst[i] < last_max_value ? std::abs((double) lst[i] - last_max_value) : 0));
        last_max_value = std::max(last_max_value, (double) lst[i]);
    }

    int cur_phase2 = lst.size() - 1;
    last_max_value = -1.0;
    std::vector<double> err_phase2 = {0.0};
    for (size_t i = 0; i < lst.size(); ++i) {
        int val = lst[lst.size() - 1 - i];
        if (last_max_value < 0) {
            last_max_value = val;
            err_phase2.push_back(0);
            continue;
        }
        if (val - last_max_value < -lst_thresh) {
            cur_phase2 = i - 1;
            break;
        }
        err_phase2.push_back(err_phase2.back() + (val < last_max_value ? std::abs((double) val - last_max_value) : 0));
        last_max_value = std::max(last_max_value, (double) val);
    }
    cur_phase2 = lst.size() - cur_phase2 - 1;

    if (cur_phase1 < cur_phase2 || cur_phase2 == 0) return {0.0, -1};

    double err_min = -1.0;
    int err_min_idx = -1;
    for (int i = cur_phase2; i <= cur_phase1; ++i) {
        double err_i = err_phase1[i] + err_phase2[lst.size() - 1 - i];
        if (err_min < 0 || err_i < err_min ||
            (err_i == err_min &&
             std::abs(i - (lst.size() - 1) / 2.0) <= std::abs(err_min_idx - (lst.size() - 1) / 2.0))) {
            err_min = err_i;
            err_min_idx = i;
        }
    }

    double center_bias = 1.0 - std::abs(err_min_idx - (lst.size() - 1.0) / 2.0) / lst.size();
    double err_score = std::max(0.5, std::min(1.0, 1.0 - err_min / std::max(lst_max, (double) lst.size())));
    return {err_score * center_bias, err_min_idx};
}

std::pair<double, int> FireDetector::is_short(const std::vector<int> &lst) {
    if (lst.size() < 3) return {0.0, -1};
    int h = lst.size();
    if (h == 0) return {0.0, -1};
    int w = *std::max_element(lst.begin(), lst.end());
    if (w == 0 || (double) h / w > 1.5) return {0.0, -1};

    int max_val = w;
    std::vector<int> max_indices;
    for (size_t i = 0; i < lst.size(); ++i) if (lst[i] == max_val) max_indices.push_back(i);
    int peak_index = max_indices[max_indices.size() / 2];

    auto monotonic_ratio = [](const std::vector<int> &seq, bool increasing) {
        if (seq.size() < 2) return 1.0;
        double good = 0;
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            if ((increasing && seq[i + 1] >= seq[i]) || (!increasing && seq[i + 1] <= seq[i])) {
                good++;
            }
        }
        return good / (double) (seq.size() - 1);
    };

    std::vector<int> left(lst.begin(), lst.begin() + peak_index + 1);
    std::vector<int> right(lst.begin() + peak_index, lst.end());

    double inc_ratio = monotonic_ratio(left, true);
    double dec_ratio = monotonic_ratio(right, false);

    double score_raw = (inc_ratio + dec_ratio) / 2.0;

    return {0.5 + 0.5 * score_raw, peak_index};
}

std::vector<int> FireDetector::exponential_smoothing(const std::vector<int> &span_list, double alpha) {
    if (span_list.empty()) return {};
    std::vector<double> smoothed_double(span_list.size());
    smoothed_double[0] = span_list[0];
    for (size_t i = 1; i < span_list.size(); ++i) {
        smoothed_double[i] = alpha * span_list[i] + (1 - alpha) * smoothed_double[i - 1];
    }
    std::vector<int> result;
    for (double val: smoothed_double) result.push_back(static_cast<int>(val));
    return result;
}

std::pair<int, double> FireDetector::find_most_significant_valley(const std::vector<int> &span_list) {
    int n = span_list.size();
    if (n < 3) return {-1, 0.0};

    auto smoothed_list = exponential_smoothing(span_list);

    std::vector<int> left_max_array(n, 0), right_max_array(n, 0);
    std::vector<int> left_max_smooth_arr(n, 0), right_max_smooth_arr(n, 0);
    std::vector<int> left_max_avgidx(n, 0), right_max_avgidx(n, 0);

    left_max_array[0] = span_list[0];
    left_max_smooth_arr[0] = smoothed_list[0];
    for (int i = 1; i < n; ++i) {
        left_max_array[i] = std::max(left_max_array[i - 1], span_list[i]);
        left_max_smooth_arr[i] = std::max(left_max_smooth_arr[i - 1], smoothed_list[i]);
        int best_idx = -1;
        for (int j = i; j >= 0; --j)
            if (smoothed_list[j] == left_max_smooth_arr[i]) {
                best_idx = j;
                break;
            }
        left_max_avgidx[i] = best_idx;
    }

    right_max_array[n - 1] = span_list[n - 1];
    right_max_smooth_arr[n - 1] = smoothed_list[n - 1];
    right_max_avgidx[n - 1] = n - 1;
    for (int i = n - 2; i >= 0; --i) {
        right_max_array[i] = std::max(right_max_array[i + 1], span_list[i]);
        right_max_smooth_arr[i] = std::max(right_max_smooth_arr[i + 1], smoothed_list[i]);
        int best_idx = -1;
        for (int j = i; j < n; ++j)
            if (smoothed_list[j] == right_max_smooth_arr[i]) {
                best_idx = j;
                break;
            }
        right_max_avgidx[i] = best_idx;
    }

    std::vector<int> valleys;
    for (int i = 1; i < n - 1; ++i)
        if (span_list[i] <= span_list[i - 1] && span_list[i] <= span_list[i + 1])
            valleys.push_back(i);

    if (valleys.empty()) return {-1, 0.0};

    int low_index = n / 4;
    int high_index = 3 * n / 4;
    std::vector<int> middle_valleys;
    for (int v: valleys) if (v >= low_index && v <= high_index) middle_valleys.push_back(v);
    if (middle_valleys.empty()) middle_valleys = valleys;

    int best_valley = -1;
    double best_depth = -1.0, best_depth0 = -1.0;
    double center = (n - 1.0) / 2.0;

    for (int i: middle_valleys) {
        double depth = std::min((double) left_max_array[i - 1] - span_list[i],
                                (double) right_max_array[i + 1] - span_list[i]);
        if (depth <= 0) continue;

        double l_slope = (left_max_array[i - 1] - span_list[i]) / std::sqrt(1.0 + std::abs(i - left_max_avgidx[i - 1]));
        double r_slope =
                (right_max_array[i + 1] - span_list[i]) / std::sqrt(1.0 + std::abs(i - right_max_avgidx[i + 1]));
        double depth0 = (l_slope + r_slope) / 2.0;

        if (best_valley == -1 || depth > best_depth ||
            (depth == best_depth && std::abs(i - center) < std::abs(best_valley - center))) {
            best_valley = i;
            best_depth = depth;
            best_depth0 = depth0;
        }
    }
    return {best_valley, best_depth0};
}

std::tuple<double, int, double> FireDetector::is_funnel(const std::vector<int> &lst) {
    if (lst.size() <= 3) return {0.0, -1, 0.0};

    auto [valley_idx, best_depth] = find_most_significant_valley(lst);
    if (valley_idx == -1) return {0.0, -1, 0.0};

    auto left_begin = lst.begin();
    auto left_end = lst.begin() + valley_idx;
    if (left_begin >= left_end) return {0.0, -1, 0.0};
    double left_peak = *std::max_element(left_begin, left_end);

    auto right_begin = lst.begin() + valley_idx + 1;
    auto right_end = lst.end();
    double right_peak = (right_begin < right_end) ? *std::max_element(right_begin, right_end) : 0;

    double peak = std::max(left_peak, right_peak);
    if (peak == 0 || lst[valley_idx] >= std::min(left_peak, right_peak)) return {0.0, -1, 0.0};

    double score = 1.0 - (double) lst[valley_idx] / peak;
    if (score < 0.2) return {0.0, -1, best_depth};

    score = 0.5 + 0.5 * score;
    double depth_score = best_depth / 2.0;
    if (depth_score > 1.0) depth_score = std::min(1.25, std::sqrt(depth_score));
    double len_score = 1.0;
    for (int i = lst.size(); i < 7; ++i) len_score *= 0.9;

    double val_score = 1.0;
    int max_lst_val = *std::max_element(lst.begin(), lst.end());
    for (int i = max_lst_val; i < 5; ++i) val_score *= 0.9;

    score = score * depth_score * len_score * val_score;
    if (score < 0.35) return {0.0, -1, best_depth};

    if (score < 0.5) return {0.0, -1, best_depth * len_score * val_score};

    return {score, valley_idx, best_depth * len_score * val_score};
}

std::pair<double, int> FireDetector::is_diamond(const std::vector<int> &lst) {
    if (lst.size() < 3) return {0.0, -1};

    int seq_id = std::max((int) lst.size() / 5, 1);
    if (lst.size() <= 2 * seq_id) return {0.0, -1};

    double pre_max = *std::max_element(lst.begin(), lst.begin() + seq_id);

    std::vector<int> mid_seq(lst.begin() + seq_id, lst.end() - seq_id);
    double mid_max = *std::max_element(mid_seq.begin(), mid_seq.end());

    std::vector<int> max_positions;
    for (size_t i = 0; i < mid_seq.size(); ++i) {
        if (mid_seq[i] == mid_max) max_positions.push_back(i);
    }

    double center_index = (mid_seq.size() - 1.0) / 2.0;
    int peak_idx = -1;
    double min_dist = -1.0;
    for (int pos: max_positions) {
        double dist = std::abs(pos - center_index);
        if (peak_idx == -1 || dist < min_dist) {
            min_dist = dist;
            peak_idx = pos;
        }
    }

    double after_max = *std::max_element(lst.end() - seq_id, lst.end());
    double edge_avg = std::max(pre_max, after_max);

    if (mid_max < 1e-9) return {0.0, -1};
    double ratio = edge_avg / mid_max;
    double score = 1.0 - ratio;

    if (score < 0.3) return {0.0, -1};

    score = std::max(0.0, std::min(1.0, score));
    score = 0.5 + 0.5 * score;

    return {score, peak_idx + seq_id};
}

std::pair<double, int> FireDetector::is_rectangle(const std::vector<int> &lst) {
    if (lst.size() < 3) return {0.0, -1};
    double h = lst.size();
    double w = *std::max_element(lst.begin(), lst.end());
    if (w < 1e-9 || h / w < 2.0) return {0.0, -1};
    return {1.0, -1};
}

FireLocateResult FireDetector::shape_process(const std::vector<int> &span_list, const cv::Mat &im, const cv::Rect &xxyy,
                                             const std::vector<int> &left_zeros, const RectD &ext_xxyy, int path_idx) {
    if (span_list.empty()) {
        PointD coord = {floor(xxyy.x + xxyy.width / 2.0), floor(xxyy.y + xxyy.height * 4.0 / 5.0)};
        return {5, coord, SHAPE_SCORES[5] * 0.0, {}};
    }

    int x1 = xxyy.x, y1 = xxyy.y;
    int x2 = x1 + xxyy.width, y2 = y1 + xxyy.height;
    PointD coord;

    auto [circle_score, circle_id] = is_circle(span_list);
    if (circle_score >= 0.5) {
        coord.x = floor((double) (x1 + x2) / 2.0);
        coord.y = floor((((double) y1 + circle_id) + ((double) y1 + (double) (y2 - y1) * 6.0 / 7.0)) / 2.0);
        return {0, coord, circle_score * SHAPE_SCORES[0], {}};
    }

    auto [funnel_score, funnel_id, funnel_depth] = is_funnel(span_list);
    if (funnel_score >= 0.5) {
        std::vector<int> upper_part(span_list.begin(), span_list.begin() + funnel_id + 1);
        int trim_end = 0;
        for (int i = upper_part.size() - 1; i >= 0; --i) { if (upper_part[i] == 0) trim_end++; else break; }
        if (trim_end > 0) upper_part.resize(upper_part.size() - trim_end);

        int coord_line_idx = std::max(1, (int) round(upper_part.size() * 4.0 / 5.0));
        int line_width = (coord_line_idx > 0 && (size_t) coord_line_idx - 1 < upper_part.size()) ? upper_part[
                coord_line_idx - 1] : 0;

        double coord_x_f;
        if (line_width <= 2) {
            coord_x_f = (*std::max_element(span_list.begin(), span_list.end()) - 1) / 2.0;
        } else {
            int edge_idx = -1;
            double edge_value_thresh = std::min(5.0,
                                                (double) *std::max_element(upper_part.begin(), upper_part.end()) * 0.2);
            for (int i = upper_part.size() - 1; i >= 0; --i) {
                if (upper_part[i] <= edge_value_thresh) edge_idx = i;
                else break;
            }
            if (edge_idx != -1 && edge_idx <= coord_line_idx - 1) {
                coord_line_idx = edge_idx;
                line_width = (coord_line_idx > 0 && (size_t) coord_line_idx - 1 < upper_part.size()) ? upper_part[
                        coord_line_idx - 1] : 0;
            }
            coord_x_f = left_zeros[coord_line_idx - 1] + line_width / 2.0;
            if (line_width % 2 == 0) {
                if ((*std::max_element(span_list.begin(), span_list.end()) - 1.0) / 2.0 < coord_x_f) {
                    coord_x_f -= 1;
                }
            }
        }
        coord.x = floor((double) x1 + coord_x_f);
        coord.y = floor((double) y1 + coord_line_idx - 1);
        return {1, coord, funnel_score * SHAPE_SCORES[1], {}};
    }

    auto [short_score, short_id] = is_short(span_list);
    if (short_score >= 0.5) {
        coord.x = floor((double) (x1 + x2) / 2.0);
        coord.y = floor((((double) y1 + short_id) + ((double) y1 + (double) (y2 - y1) * 6.0 / 7.0)) / 2.0);
        return {2, coord, short_score * SHAPE_SCORES[2], {}};
    }

    auto [diamond_score, diamond_id] = is_diamond(span_list);
    if (diamond_score >= 0.5) {
        coord.x = floor((double) (x1 + x2) / 2.0);
        coord.y = floor((((double) y1 + diamond_id) + ((double) y1 + (span_list.size() - 1.0) / 2.0)) / 2.0);
        return {3, coord, diamond_score * SHAPE_SCORES[3], {}};
    }

    auto [rectangle_score, rectangle_id] = is_rectangle(span_list);
    if (rectangle_score >= 0.5) {
        coord.x = floor((double) (x1 + x2) / 2.0);
        coord.y = floor((double) y1 + (double) (y2 - y1) * 3.0 / 5.0);
        return {4, coord, rectangle_score * SHAPE_SCORES[4], {}};
    }

    double len_score = span_list.empty() ? 0.0 : 1.0;
    coord.x = floor((double) (x1 + x2) / 2.0);
    coord.y = floor((double) y1 + (double) (y2 - y1) * 4.0 / 5.0);
    return {5, coord, len_score * SHAPE_SCORES[5], {}};
}

FireLocateResult
FireDetector::fire_locate(const cv::Mat &im, const RectD &bbox_d, const RectD &ext_xxyy, int path_idx) {
    cv::Rect bbox(bbox_d.x, bbox_d.y, bbox_d.width, bbox_d.height);
    std::vector<FireLocateResult> results;

    if (bbox.width <= 0 || bbox.height <= 0 || bbox.x + bbox.width > im.cols || bbox.y + bbox.height > im.rows ||
        bbox.x < 0 || bbox.y < 0) {
        return {5, {floor(bbox.x + bbox.width / 2.0), floor(bbox.y + bbox.height * 0.8)}, 0.0, {}};
    }
    cv::Mat roi;
    try {
        roi = im(bbox);
    } catch (const cv::Exception &e) {
        return {5, {floor(bbox.x + bbox.width / 2.0), floor(bbox.y + bbox.height * 0.8)}, 0.0, {}};
    }

    if (roi.empty()) {
        return {5, {floor(bbox.x + bbox.width / 2.0), floor(bbox.y + bbox.height * 0.8)}, 0.0, {}};
    }

    cv::Mat rb_avg_roi = cal_rb(roi);
    if (rb_avg_roi.empty()) {
        return {5, {floor(bbox.x + bbox.width / 2.0), floor(bbox.y + bbox.height * 0.8)}, 0.0, {}};
    }

    cv::Mat rb_flat = rb_avg_roi.reshape(1, rb_avg_roi.total());
    cv::sort(rb_flat, rb_flat, cv::SORT_ASCENDING);

    int start_idx = rb_flat.cols * 0.5;
    double adaptive_thresh_val = (start_idx < rb_flat.cols) ? cv::mean(rb_flat.colRange(start_idx, rb_flat.cols))[0]
                                                            : 0.0;
    int adaptive_thresh = std::max(140, (int) adaptive_thresh_val);

    std::vector<int> threshes = {-1, 200, adaptive_thresh};

    for (int thresh: threshes) {
        auto [y1_ex, y2_ex] = refine_bbox(im, bbox, thresh);
        int new_y1 = bbox.y - y1_ex;
        int new_y2 = bbox.y + bbox.height + y2_ex;
        if (new_y1 < 0) new_y1 = 0;
        if (new_y2 > im.rows) new_y2 = im.rows;
        if (new_y1 >= new_y2) continue;

        cv::Rect roi_ex_rect(bbox.x, new_y1, bbox.width, new_y2 - new_y1);
        if (roi_ex_rect.width <= 0 || roi_ex_rect.height <= 0 || roi_ex_rect.x < 0 || roi_ex_rect.y < 0 ||
            roi_ex_rect.x + roi_ex_rect.width > im.cols || roi_ex_rect.y + roi_ex_rect.height > im.rows)
            continue;

        cv::Mat roi_ex = im(roi_ex_rect);

        auto [spans, left_z, right_z] = count_high_rg_pixels_per_row(roi_ex, thresh);
        if (spans.empty()) continue;

        int x1_ex = left_z.empty() ? 0 : *std::min_element(left_z.begin(), left_z.end());
        int x2_ex = right_z.empty() ? 0 : *std::min_element(right_z.begin(), right_z.end());
        if (roi_ex_rect.width - x1_ex - x2_ex <= 0) continue;

        std::vector<int> final_left_z = left_z;
        for (size_t i = 0; i < final_left_z.size(); ++i) final_left_z[i] -= x1_ex;

        cv::Rect final_xxyy(roi_ex_rect.x + x1_ex, roi_ex_rect.y, roi_ex_rect.width - x1_ex - x2_ex,
                            roi_ex_rect.height);
        results.push_back(shape_process(spans, im, final_xxyy, final_left_z, ext_xxyy, path_idx));
    }
    if (results.empty()) return {5, {floor(bbox.x + bbox.width / 2.0), floor(bbox.y + bbox.height * 0.8)}, 0.1, {}};

    auto best_it = std::max_element(results.begin(), results.end(),
                                    [](const FireLocateResult &a, const FireLocateResult &b) {
                                        return a.weight < b.weight;
                                    });
    return *best_it;
}

double FireDetector::calculate_iou(const RectD &box1, const RectD &box2) {
    RectD inter_rect = box1 & box2;
    double union_area = box1.area() + box2.area() - inter_rect.area();
    return (union_area < 1e-9) ? 0.0 : inter_rect.area() / union_area;
}

std::vector<NmsBBoxInfo> FireDetector::merge_rects(std::vector<NmsBBoxInfo> &rects) {
    bool merged_flag = true;
    while (merged_flag) {
        merged_flag = false;
        if (rects.size() < 2) break;

        std::vector<NmsBBoxInfo> new_rects;
        std::vector<bool> merged_mask(rects.size(), false);

        for (size_t i = 0; i < rects.size(); ++i) {
            if (merged_mask[i]) continue;

            RectD current_rect = rects[i].box;
            double current_score = rects[i].score;

            for (size_t j = i + 1; j < rects.size(); ++j) {
                if (merged_mask[j]) continue;

                const RectD &other_rect = rects[j].box;

                bool should_merge = false;
                if (calculate_iou(current_rect, other_rect) > 0) {
                    should_merge = true;
                } else {
                    double proximity_y_dist = std::max({4.0, current_rect.height, other_rect.height});
                    double proximity_x_dist = std::max({4.0, current_rect.width, other_rect.width});

                    bool y_close =
                            (std::abs(other_rect.y - (current_rect.y + current_rect.height)) < proximity_y_dist) ||
                            (std::abs(current_rect.y - (other_rect.y + other_rect.height)) < proximity_y_dist);
                    bool x_close =
                            (std::abs(other_rect.x - (current_rect.x + current_rect.width)) < proximity_x_dist) ||
                            (std::abs(current_rect.x - (other_rect.x + other_rect.width)) < proximity_x_dist);

                    if (y_close && x_close) {
                        should_merge = true;
                    }
                }

                if (should_merge) {
                    current_rect |= other_rect;
                    current_score = std::max(current_score, rects[j].score);
                    merged_mask[j] = true;
                    merged_flag = true;
                }
            }
            new_rects.push_back({current_score, 0, current_rect});
            merged_mask[i] = true;
        }
        rects = new_rects;
    }
    return rects;
}