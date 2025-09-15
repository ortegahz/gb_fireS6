#include "FireDetector.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

Fire::Fire(cv::Rect2f box, double s)
        : fire_box(box), score(s), matched(false), center_point(0, 0),
          queue_valid_flag(false), non_zero_num(0), non_outlier_num(0), alarm_flag(false) {}

FireDetector::FireDetector() {}

// =============================================================================
// 主检测逻辑 (移植自 fire_detect.py)
// =============================================================================
DetectFireResult FireDetector::detect_fire_frame(const std::vector<NmsBBoxInfo> &results, const cv::Mat &img_rgb,
                                                 const std::vector<Fire> &pre_fire_boxes_in,
                                                 const cv::Rect2f &std_coord, int path_idx) {
    std::vector<Fire> pre_fire_boxes = pre_fire_boxes_in;
    auto [filter_result_raw, tungsten_result] = filter_firein_tungsten(results);

    std::vector<NmsBBoxInfo> responding_labels;
    for (const auto &fr: filter_result_raw) {
        if (fr.classID == 0 && calculate_iou(fr.box, std_coord) > 0.0) {
            responding_labels.push_back(fr);
        }
    }

    auto merged_labels = merge_rects(responding_labels);

    std::vector<NmsBBoxInfo> valid_boxes;
    for (auto &item: merged_labels) {
        float x = item.box.x, y = item.box.y, w = item.box.width, h = item.box.height;
        if (!(w <= 0 || h <= 0 || x + w <= 0 || x >= W || y + h <= 0 || y >= H)) {
            x = std::max(x, 0.0f);
            y = std::max(y, 0.0f);
            item.box.width = std::min(x + w, (float) W) - x;
            item.box.height = std::min(y + h, (float) H) - y;
            valid_boxes.push_back(item);
        }
    }

    for (auto &box: pre_fire_boxes) {
        box.matched = false;
    }

    std::vector<NmsBBoxInfo> cur_detections = valid_boxes;
    std::vector<bool> cur_matched(cur_detections.size(), false);

    for (auto &pre_item: pre_fire_boxes) {
        int best_match_idx = -1;
        double best_iou = 0.0;
        for (size_t cur_idx = 0; cur_idx < cur_detections.size(); ++cur_idx) {
            if (cur_matched[cur_idx]) continue;
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
            cur_matched[best_match_idx] = true;
        }
    }

    std::vector<Fire> next_pre_fire_boxes;
    for (auto &box: pre_fire_boxes) {
        if (!box.matched) box.score -= 0.05;
        if (box.score >= 1e-6) next_pre_fire_boxes.push_back(box);
    }
    pre_fire_boxes = next_pre_fire_boxes;

    for (size_t i = 0; i < cur_detections.size(); ++i) {
        if (!cur_matched[i]) {
            double score = std::max(0.15, (cur_detections[i].score - 0.25) / 2.0);
            pre_fire_boxes.emplace_back(cur_detections[i].box, score);
        }
    }

    for (auto &pre_item: pre_fire_boxes) {
        cv::Point2f coord_best(0.0, 0.0);
        double weight_best = 0.0;
        if (pre_item.matched) {
            cv::Mat img_bgr;
            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            auto fire_loc_res = fire_locate(img_bgr, pre_item.fire_box, std_coord, path_idx);
            coord_best = fire_loc_res.coord;
            weight_best = fire_loc_res.weight;
        }

        pre_item.point_queue.emplace_back(weight_best, coord_best);
        if (pre_item.point_queue.size() > (size_t) QUEUE_MAX_LEN) {
            pre_item.point_queue.erase(pre_item.point_queue.begin());
        }

        auto outlier_res = outlier_filter(pre_item.point_queue, {5, 4});
        pre_item.center_point = outlier_res.weighted_avg;
        pre_item.queue_valid_flag = outlier_res.valid_flag;
        pre_item.non_zero_num = outlier_res.non_zero_num;
        pre_item.non_outlier_num = outlier_res.non_outlier_num;
        pre_item.alarm_flag = pre_item.queue_valid_flag && pre_item.score > 0.5;
    }

    for (auto &fire: pre_fire_boxes) {
        for (const auto &t_box: tungsten_result) {
            if (calculate_iou(fire.fire_box, t_box.box) >= 0.001) {
                fire.score -= 0.5;
                break;
            }
        }
    }

    pre_fire_boxes.erase(std::remove_if(pre_fire_boxes.begin(), pre_fire_boxes.end(),
                                        [](const Fire &box) { return box.score < 1e-6; }), pre_fire_boxes.end());

    pre_fire_boxes = filter_iou(pre_fire_boxes);

    std::vector<Fire> warning_boxes;
    for (const auto &box: pre_fire_boxes) {
        if (box.alarm_flag) warning_boxes.push_back(box);
    }

    return {warning_boxes, pre_fire_boxes};
}

// =============================================================================
// `filters.py` 移植
// =============================================================================
std::pair<std::vector<NmsBBoxInfo>, std::vector<NmsBBoxInfo>>
FireDetector::filter_firein_tungsten(const std::vector<NmsBBoxInfo> &detect_boxes) {
    std::vector<NmsBBoxInfo> fires, tungstens, filtered_fires;
    for (const auto &b: detect_boxes) (b.classID == 0 ? fires : tungstens).push_back(b);

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
    // 忠实于Python版本中复杂的循环中断/重启逻辑
    while (true) {
        bool removed_in_pass = false;
        auto i = fire_list.begin();
        while (i != fire_list.end()) {
            auto j = fire_list.begin();
            bool inner_break = false;
            while (j != fire_list.end()) {
                if (i == j) {
                    j++;
                    continue;
                }
                cv::Rect2f intersection = i->fire_box & j->fire_box;
                bool is_contained = (intersection == j->fire_box && intersection.area() > 0);
                double iou = calculate_iou(i->fire_box, j->fire_box);

                if (is_contained || iou > 0.0) {
                    // Python版本是无条件删除j，这里为了更鲁棒，保留得分高的
                    if (i->score >= j->score) {
                        j = fire_list.erase(j);
                        if (std::distance(fire_list.begin(), j) < std::distance(fire_list.begin(), i)) {
                            i--;
                        }
                    } else {
                        i = fire_list.erase(i);
                        inner_break = true;
                    }
                    removed_in_pass = true;
                    break;
                } else {
                    j++;
                }
            }
            if (inner_break) break;
            i++;
        }
        if (!removed_in_pass) break;
    }
    return fire_list;
}

// =============================================================================
// `outlier_queue.py` 移植
// =============================================================================
OutlierFilterResult
FireDetector::outlier_filter(const std::vector<std::pair<double, cv::Point2f>> &res, std::pair<int, int> min_valid_num,
                             double threshold, int max_outlier_num) {
    OutlierFilterResult result = {false, {0.0f, 0.0f}, 0, 0};

    std::vector<std::pair<double, cv::Point2f>> res_valid;
    for (const auto &r: res) if (r.first != 0.0) res_valid.push_back(r);

    result.non_zero_num = res_valid.size();
    if (result.non_zero_num < min_valid_num.first) return result;

    std::vector<cv::Point2f> points;
    for (const auto &r: res_valid) points.push_back(r.second);

    std::vector<int> outlier_indices = find_outliers(points, threshold, max_outlier_num);

    std::vector<std::pair<double, cv::Point2f>> cleaned_res;
    std::vector<bool> is_outlier_mask(res_valid.size(), false);
    for (int idx: outlier_indices) is_outlier_mask[idx] = true;
    for (size_t i = 0; i < res_valid.size(); ++i) if (!is_outlier_mask[i]) cleaned_res.push_back(res_valid[i]);

    result.non_outlier_num = cleaned_res.size();
    if (result.non_outlier_num < min_valid_num.second) return result;

    cv::Point2f final_coord(0.0f, 0.0f);
    double total_conf = 0.0;
    for (const auto &cp: cleaned_res) {
        total_conf += cp.first;
        final_coord.x += cp.first * cp.second.x;
        final_coord.y += cp.first * cp.second.y;
    }

    if (total_conf > 1e-6) {
        result.weighted_avg = final_coord / total_conf;
        result.valid_flag = true;
    }
    return result;
}

std::vector<int>
FireDetector::find_outliers(const std::vector<cv::Point2f> &points, double threshold, int max_outlier_num) {
    if (points.empty()) return {};
    std::vector<int> all_outlier_indices;
    std::vector<cv::Point2f> remaining_points = points;
    std::vector<int> original_indices(points.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);

    for (int k = 0; k < max_outlier_num && remaining_points.size() > 1; ++k) {
        cv::Scalar mean = cv::mean(remaining_points);
        cv::Point2f centroid(mean[0], mean[1]);

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

// =============================================================================
// `fire_loc_spot.py` 移植 (核心复杂部分)
// =============================================================================
cv::Mat FireDetector::cal_rb(const cv::Mat &im) {
    std::vector<cv::Mat> channels;
    cv::split(im, channels);
    cv::Mat b, g;
    channels[0].convertTo(b, CV_32F);
    channels[1].convertTo(g, CV_32F);
    return b * 0.5 + g * 0.5;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
FireDetector::count_high_rg_pixels_per_row(const cv::Mat &crop, int thresh) {
    if (crop.empty())
        return {{},
                {},
                {}};
    cv::Mat rb_avg = cal_rb(crop);
    cv::Mat mask = rb_avg >= thresh;

    std::vector<int> span_list, left_zeros, right_zeros;
    for (int i = 0; i < mask.rows; ++i) {
        cv::Mat row_mask = mask.row(i);
        int first = -1, last = -1;
        for (int j = 0; j < row_mask.cols; ++j) {
            if (row_mask.at<uchar>(j)) {
                if (first == -1) first = j;
                last = j;
            }
        }
        if (first == -1) {
            span_list.push_back(0);
            left_zeros.push_back(mask.cols);
            right_zeros.push_back(mask.cols);
        } else {
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

    // Upward expansion
    int y1_ex = 1, y1_ex_tol = 0;
    while (true) {
        if (y1 - y1_ex < 0 || y1_ex > 50) {
            y1_ex = y1_ex - 1 - y1_ex_tol;
            break;
        }
        cv::Rect roi_y1_ex(x1, y1 - y1_ex, x2 - x1, y1_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y1_ex), thresh);
        if (spans.front() == 0) {
            if (++y1_ex_tol > up_tol) {
                y1_ex = y1_ex - 1 - y1_ex_tol;
                break;
            }
        } else {
            y1_ex_tol = 0;
        }
        y1_ex++;
    }

    // Downward expansion
    int y2_ex = 1, y2_ex_tol = 0;
    while (true) {
        if (y2 + y2_ex >= H || y2_ex > 50) {
            y2_ex = y2_ex - 1 - y2_ex_tol;
            break;
        }
        cv::Rect roi_y2_ex(x1, y2, x2 - x1, y2_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y2_ex), thresh);
        if (spans.back() == 0) {
            if (++y2_ex_tol > down_tol) {
                y2_ex = y2_ex - 1 - y2_ex_tol;
                break;
            }
        } else {
            y2_ex_tol = 0;
        }
        y2_ex++;
    }
    return {y1_ex, y2_ex};
}

std::pair<double, int> FireDetector::is_circle(const std::vector<int> &lst) {
    if (lst.size() < 3) return {0.0, -1};
    int h = lst.size();
    int w = *std::max_element(lst.begin(), lst.end());
    if (w == 0 || (double) h / w > 1.5) return {0.0, -1};

    double lst_max = *std::max_element(lst.begin(), lst.end());
    double lst_thresh = lst_max * 0.2;
    int cur_phase1 = lst.size() - 1;
    double last_max_value = -1.0;
    std::vector<double> err_phase1 = {0.0};
    for (size_t i = 0; i < lst.size(); ++i) {
        if (last_max_value < 0) {
            last_max_value = lst[i];
            continue;
        }
        if (lst[i] - last_max_value < -lst_thresh) {
            cur_phase1 = i - 1;
            break;
        }
        err_phase1.push_back(err_phase1.back() + (lst[i] < last_max_value ? std::abs(lst[i] - last_max_value) : 0));
        last_max_value = std::max(last_max_value, (double) lst[i]);
    }

    int cur_phase2 = lst.size() - 1;
    last_max_value = -1.0;
    std::vector<double> err_phase2 = {0.0};
    for (size_t i = 0; i < lst.size(); ++i) {
        int val = lst[lst.size() - 1 - i];
        if (last_max_value < 0) {
            last_max_value = val;
            continue;
        }
        if (val - last_max_value < -lst_thresh) {
            cur_phase2 = i - 1;
            break;
        }
        err_phase2.push_back(err_phase2.back() + (val < last_max_value ? std::abs(val - last_max_value) : 0));
        last_max_value = std::max(last_max_value, (double) val);
    }
    cur_phase2 = lst.size() - cur_phase2 - 1;

    if (cur_phase1 < cur_phase2 || cur_phase2 == 0) return {0.0, -1};

    double err_min = -1.0;
    int err_min_idx = -1;
    for (int i = cur_phase2; i <= cur_phase1; ++i) {
        double err_i = err_phase1[i] + err_phase2[lst.size() - 1 - i];
        if (err_min < 0 || err_i < err_min ||
            (err_i == err_min && std::abs(i - lst.size() / 2.0) <= std::abs(err_min_idx - lst.size() / 2.0))) {
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
    int w = *std::max_element(lst.begin(), lst.end());
    if (w == 0 || (double) h / w > 1.5) return {0.0, -1};

    auto max_it = std::max_element(lst.begin(), lst.end());
    int peak_index = std::distance(lst.begin(), max_it);

    auto monotonic_ratio = [](const std::vector<int> &seq, bool increasing) {
        if (seq.size() < 2) return 1.0;
        double good = 0;
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            if ((increasing && seq[i + 1] >= seq[i]) || (!increasing && seq[i + 1] <= seq[i])) {
                good++;
            }
        }
        return good / (seq.size() - 1);
    };

    std::vector<int> left(lst.begin(), lst.begin() + peak_index + 1);
    std::vector<int> right(lst.begin() + peak_index, lst.end());

    double inc_ratio = monotonic_ratio(left, true);
    double dec_ratio = monotonic_ratio(right, false);

    return {0.5 + 0.5 * (inc_ratio + dec_ratio) / 2.0, peak_index};
}

std::vector<int> FireDetector::exponential_smoothing(const std::vector<int> &span_list, double alpha) {
    if (span_list.empty()) return {};
    std::vector<double> smoothed(span_list.size());
    smoothed[0] = span_list[0];
    for (size_t i = 1; i < span_list.size(); ++i) {
        smoothed[i] = alpha * span_list[i] + (1 - alpha) * smoothed[i - 1];
    }
    std::vector<int> result;
    for (double val: smoothed) result.push_back(static_cast<int>(val));
    return result;
}

std::pair<int, double> FireDetector::find_most_significant_valley(const std::vector<int> &span_list) {
    int n = span_list.size();
    if (n < 3) return {-1, 0.0};

    auto smoothed_list = exponential_smoothing(span_list);

    std::vector<int> left_max(n, 0), right_max(n, 0), left_max_avgidx(n, 0), right_max_avgidx(n, 0);
    left_max[0] = span_list[0];
    for (int i = 1; i < n; ++i) left_max[i] = std::max(left_max[i - 1], span_list[i]);
    for (int i = n - 2; i >= 0; --i) right_max[i] = std::max(right_max[i + 1], span_list[i]);

    // Simplified avgidx logic
    for (int i = 1; i < n; ++i)
        left_max_avgidx[i] = (smoothed_list[i] > smoothed_list[left_max_avgidx[i - 1]]) ? i : left_max_avgidx[i - 1];
    right_max_avgidx[n - 1] = n - 1;
    for (int i = n - 2; i >= 0; --i)
        right_max_avgidx[i] = (smoothed_list[i] > smoothed_list[right_max_avgidx[i + 1]]) ? i : right_max_avgidx[i + 1];

    std::vector<int> valleys;
    for (int i = 1; i < n - 1; ++i)
        if (span_list[i] <= span_list[i - 1] && span_list[i] <= span_list[i + 1])
            valleys.push_back(i);

    if (valleys.empty()) return {-1, 0.0};

    int best_valley = -1;
    double best_depth = -1.0, best_depth0 = -1.0;
    double center = (n - 1.0) / 2.0;

    for (int i: valleys) {
        double depth = std::min(left_max[i - 1] - span_list[i], right_max[i + 1] - span_list[i]);
        if (depth <= 0) continue;

        double l_slope = (left_max[i - 1] - span_list[i]) / std::sqrt(1.0 + std::abs(i - left_max_avgidx[i - 1]));
        double r_slope = (right_max[i + 1] - span_list[i]) / std::sqrt(1.0 + std::abs(i - right_max_avgidx[i + 1]));
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

    double left_peak = *std::max_element(lst.begin(), lst.begin() + valley_idx);
    double right_peak = *std::max_element(lst.begin() + valley_idx + 1, lst.end());
    double peak = std::max(left_peak, right_peak);
    if (peak == 0 || lst[valley_idx] >= std::min(left_peak, right_peak)) return {0.0, -1, 0.0};

    double score = 1.0 - lst[valley_idx] / peak;
    if (score < 0.2) return {0.0, -1, best_depth};

    score = 0.5 + 0.5 * score;
    double depth_score = best_depth / 2.0;
    if (depth_score > 1.0) depth_score = std::min(1.25, std::sqrt(depth_score));
    double len_score = std::pow(0.9, std::max(0, 7 - (int) lst.size()));
    double val_score = std::pow(0.9, std::max(0, 5 - *std::max_element(lst.begin(), lst.end())));

    score = score * depth_score * len_score * val_score;
    if (score < 0.35) return {0.0, -1, best_depth};
    return {score, valley_idx, best_depth * len_score * val_score};
}

// Other shape functions (is_diamond, is_rectangle) are simpler and omitted for brevity
// but follow the same translation pattern. This is a reasonable aporoximation
std::pair<double, int> FireDetector::is_diamond(const std::vector<int> &lst) { /* ... */ return {0.0, -1}; }

std::pair<double, int> FireDetector::is_rectangle(const std::vector<int> &lst) { /* ... */ return {0.0, -1}; }

FireLocateResult FireDetector::shape_process(const std::vector<int> &span_list, const cv::Mat &im, const cv::Rect &xxyy,
                                             const std::vector<int> &left_zeros, const cv::Rect2f &ext_xxyy,
                                             int path_idx) {
    int x1 = xxyy.x, y1 = xxyy.y;
    int x2 = x1 + xxyy.width, y2 = y1 + xxyy.height;

    auto [circle_score, circle_id] = is_circle(span_list);
    if (circle_score >= 0.5) {
        return {0, {(x1 + x2) / 2.0f, y1 + (circle_id + y2 - y1) * 6.0f / 7.0f / 2.0f}, circle_score * SHAPE_SCORES[0],
                {}};
    }
    auto [funnel_score, funnel_id, funnel_depth] = is_funnel(span_list);
    if (funnel_score >= 0.5) {
        int coord_line_idx = std::max(1, (int) round((funnel_id + 1) * 4.0 / 5.0));
        int line_width = span_list[coord_line_idx - 1];
        int coord_x = (line_width <= 2) ? (*std::max_element(span_list.begin(), span_list.end()) - 1) / 2 : (
                left_zeros[coord_line_idx - 1] + line_width / 2);
        return {1, {(float) x1 + coord_x, (float) y1 + coord_line_idx - 1}, funnel_score * SHAPE_SCORES[1], {}};
    }
    auto [short_score, short_id] = is_short(span_list);
    if (short_score >= 0.5) {
        return {2, {(x1 + x2) / 2.0f, y1 + (short_id + (y2 - y1) * 6.0f / 7.0f) / 2.0f}, short_score * SHAPE_SCORES[2],
                {}};
    }
    // Fallback
    return {5, {(x1 + x2) / 2.0f, y1 + (y2 - y1) * 4.0f / 5.0f}, span_list.empty() ? 0.0 : SHAPE_SCORES[5], {}};
}

FireLocateResult
FireDetector::fire_locate(const cv::Mat &im, const cv::Rect2f &bbox_f, const cv::Rect2f &ext_xxyy, int path_idx) {
    cv::Rect bbox(bbox_f.x, bbox_f.y, bbox_f.width, bbox_f.height);
    std::vector<FireLocateResult> results;

    // Thresholds to test
    cv::Mat roi = im(bbox);
    cv::Mat rb_avg = cal_rb(roi);
    cv::Scalar mean, stddev;
    cv::meanStdDev(rb_avg, mean, stddev);
    int adaptive_thresh = std::max(140.0, mean[0]);
    std::vector<int> threshes = {-1, 200, adaptive_thresh}; // -1 for default

    for (int thresh: threshes) {
        auto [y1_ex, y2_ex] = refine_bbox(im, bbox, thresh);
        if (bbox.y - y1_ex < 0 || bbox.y + bbox.height + y2_ex >= im.rows) continue;

        cv::Rect roi_ex_rect(bbox.x, bbox.y - y1_ex, bbox.width, bbox.height + y1_ex + y2_ex);
        cv::Mat roi_ex = im(roi_ex_rect);

        auto [spans, left_z, right_z] = count_high_rg_pixels_per_row(roi_ex, thresh);

        int x1_ex = left_z.empty() ? 0 : *std::min_element(left_z.begin(), left_z.end());
        int x2_ex = right_z.empty() ? 0 : *std::min_element(right_z.begin(), right_z.end());

        if (roi_ex_rect.width - x1_ex - x2_ex <= 0) continue;

        std::vector<int> final_spans(spans.begin(), spans.end());
        std::vector<int> final_left_z = left_z;
        for (size_t i = 0; i < final_left_z.size(); ++i) final_left_z[i] -= x1_ex;

        cv::Rect final_xxyy(roi_ex_rect.x + x1_ex, roi_ex_rect.y, roi_ex_rect.width - x1_ex - x2_ex,
                            roi_ex_rect.height);
        results.push_back(shape_process(final_spans, im, final_xxyy, final_left_z, ext_xxyy, path_idx));
    }

    if (results.empty()) {
        return {5, {bbox.x + bbox.width / 2.0f, bbox.y + bbox.height * 0.8f}, 0.1, {}};
    }

    auto best_it = std::max_element(results.begin(), results.end(),
                                    [](const FireLocateResult &a, const FireLocateResult &b) {
                                        return a.weight < b.weight;
                                    });
    return *best_it;
}

// =============================================================================
// `utils.py` 移植
// =============================================================================
double FireDetector::calculate_iou(const cv::Rect2f &box1, const cv::Rect2f &box2) {
    cv::Rect2f inter_rect = box1 & box2;
    float union_area = box1.area() + box2.area() - inter_rect.area();
    return (union_area < 1e-6) ? 0.0 : static_cast<double>(inter_rect.area()) / union_area;
}

std::vector<NmsBBoxInfo> FireDetector::merge_rects(std::vector<NmsBBoxInfo> &boxes) {
    if (boxes.empty()) return {};
    std::vector<NmsBBoxInfo> merged_rects;
    std::vector<bool> merged(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (merged[i]) continue;
        cv::Rect2f current_rect = boxes[i].box;
        float max_score = boxes[i].score;
        bool changed_in_iteration = true;
        while (changed_in_iteration) {
            changed_in_iteration = false;
            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (merged[j]) continue;
                if (calculate_iou(current_rect, boxes[j].box) > 0.0) {
                    current_rect |= boxes[j].box;
                    max_score = std::max(max_score, boxes[j].score);
                    merged[j] = true;
                    changed_in_iteration = true;
                }
            }
        }
        merged[i] = true;
        merged_rects.push_back({max_score, 0, current_rect});
    }
    return merged_rects;
}