#include "FireDetector.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>

Fire::Fire(RectD box, double s)
        : fire_box(box), score(s), matched(false), center_point(0, 0),
          queue_valid_flag(false), non_zero_num(0), non_outlier_num(0), alarm_flag(false) {}

FireDetector::FireDetector() {}

DetectFireResult FireDetector::detect_fire_frame(const std::vector<NmsBBoxInfo> &results, const cv::Mat &img_rgb,
                                                 const std::vector<Fire> &pre_fire_boxes_in, const RectD &std_coord,
                                                 int path_idx) {
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
        item.box = item.box & RectD(0, 0, W, H);
        if (item.box.width > 0 && item.box.height > 0) {
            valid_boxes.push_back(item);
        }
    }

    std::vector<Fire> pre_fire_boxes = pre_fire_boxes_in;
    for (auto &box: pre_fire_boxes) box.matched = false;

    std::vector<NmsBBoxInfo> cur_detections = valid_boxes;
    std::vector<bool> cur_matched_flags(cur_detections.size(), false);

    // 1. Match existing tracks
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

    // 2. Combine old and new tracks into a single list for this frame
    std::vector<Fire> current_frame_fires;
    for (auto &box: pre_fire_boxes) {
        if (!box.matched) box.score -= 0.05;
        if (box.score >= 1e-6) current_frame_fires.push_back(box);
    }
    for (size_t i = 0; i < cur_detections.size(); ++i) {
        if (!cur_matched_flags[i]) {
            double score = std::max(0.15, (cur_detections[i].score - 0.25) / 2.0);
            Fire new_fire(cur_detections[i].box, score);
            new_fire.matched = true; // Mark as "matched" to a detection in its birth frame
            current_frame_fires.push_back(new_fire);
        }
    }

    // 3. Process all tracks for the current frame
    for (auto &fire: current_frame_fires) {
        PointD coord_best(0.0, 0.0);
        double weight_best = 0.0;
        if (fire.matched) {
            cv::Mat img_bgr;
            cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
            auto fire_loc_res = fire_locate(img_bgr, fire.fire_box, std_coord, path_idx);
            coord_best = fire_loc_res.coord;
            weight_best = fire_loc_res.weight;
        }

        fire.point_queue.emplace_back(weight_best, coord_best);
        if (fire.point_queue.size() > (size_t) QUEUE_MAX_LEN) {
            fire.point_queue.erase(fire.point_queue.begin());
        }

        auto outlier_res = outlier_filter(fire.point_queue, {5, 4});
        fire.center_point = outlier_res.weighted_avg;
        fire.queue_valid_flag = outlier_res.valid_flag;
        fire.non_zero_num = outlier_res.non_zero_num;
        fire.non_outlier_num = outlier_res.non_outlier_num;
        fire.alarm_flag = fire.queue_valid_flag && fire.score > 0.5;
    }

    // 4. Post-processing
    for (auto &fire: current_frame_fires) {
        for (const auto &t_box: tungsten_result) {
            if (calculate_iou(fire.fire_box, t_box.box) >= 0.001) {
                fire.score -= 0.5;
                break;
            }
        }
    }
    current_frame_fires.erase(std::remove_if(current_frame_fires.begin(), current_frame_fires.end(),
                                             [](const Fire &box) { return box.score < 1e-6; }),
                              current_frame_fires.end());

    current_frame_fires = filter_iou(current_frame_fires);

    std::vector<Fire> warning_boxes;
    for (const auto &box: current_frame_fires) {
        if (box.alarm_flag) warning_boxes.push_back(box);
    }

    return {warning_boxes, current_frame_fires};
}

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
    if (fire_list.size() < 2) return fire_list;
    while (true) {
        bool removed_in_pass = false;
        for (size_t i = 0; i < fire_list.size();) {
            bool i_was_removed = false;
            for (size_t j = 0; j < fire_list.size();) {
                if (i == j) {
                    ++j;
                    continue;
                }

                RectD intersection = fire_list[i].fire_box & fire_list[j].fire_box;
                bool is_contained = (intersection == fire_list[j].fire_box && intersection.area() > 1e-9);
                double iou = calculate_iou(fire_list[i].fire_box, fire_list[j].fire_box);

                if (is_contained || iou > 1e-9) {
                    removed_in_pass = true;
                    // Python logic is complex. This should approximate it: remove the one with lower index (j) if scores are equal.
                    // This is still an approximation, but closer to the original's weirdness.
                    if (fire_list[i].score < fire_list[j].score) {
                        fire_list.erase(fire_list.begin() + i);
                        i_was_removed = true;
                        goto restart_outer_loop;
                    } else {
                        fire_list.erase(fire_list.begin() + j);
                        if (j < i) --i;
                    }
                } else {
                    ++j;
                }
            }
            if (!i_was_removed) ++i;
        }
        restart_outer_loop:;
        if (!removed_in_pass) break;
    }
    return fire_list;
}

OutlierFilterResult
FireDetector::outlier_filter(const std::vector<std::pair<double, PointD>> &res, std::pair<int, int> min_valid_num,
                             double threshold, int max_outlier_num) {
    OutlierFilterResult result = {false, {0.0, 0.0}, 0, 0};

    std::vector<std::pair<double, PointD>> res_valid;
    for (const auto &r: res) if (r.first != 0.0) res_valid.push_back(r);

    result.non_zero_num = res_valid.size();
    if ((int) result.non_zero_num < min_valid_num.first) return result;

    std::vector<PointD> points;
    for (const auto &r: res_valid) points.push_back(r.second);

    std::vector<int> outlier_indices = find_outliers(points, threshold, max_outlier_num);

    std::vector<std::pair<double, PointD>> cleaned_res;
    std::vector<bool> is_outlier_mask(res_valid.size(), false);
    for (int idx: outlier_indices) if (idx < is_outlier_mask.size()) is_outlier_mask[idx] = true;
    for (size_t i = 0; i < res_valid.size(); ++i) if (!is_outlier_mask[i]) cleaned_res.push_back(res_valid[i]);

    result.non_outlier_num = cleaned_res.size();
    if ((int) result.non_outlier_num < min_valid_num.second) return result;

    PointD final_coord(0.0, 0.0);
    double total_conf = 0.0;
    for (const auto &cp: cleaned_res) {
        total_conf += cp.first;
        final_coord.x += cp.first * cp.second.x;
        final_coord.y += cp.first * cp.second.y;
    }

    if (total_conf > 1e-9) {
        result.weighted_avg = final_coord / total_conf;
        result.valid_flag = true;
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
        cv::Scalar mean = cv::mean(remaining_points);
        PointD centroid(mean[0], mean[1]);

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
    /* ... same as before ... */
    std::vector<cv::Mat> channels;
    cv::split(im, channels);
    cv::Mat b, g;
    channels[0].convertTo(b, CV_32F);
    channels[1].convertTo(g, CV_32F);
    return b * 0.5 + g * 0.5;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
FireDetector::count_high_rg_pixels_per_row(const cv::Mat &crop, int thresh) {
    /* ... same as before, with thresh==-1 check ... */
    if (crop.empty() || crop.cols == 0 || crop.rows == 0)
        return {{},
                {},
                {}};
    cv::Mat rb_avg = cal_rb(crop);
    cv::Mat mask = rb_avg >= (thresh == -1 ? 227 : thresh);

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
    /* ... same as before ... */
    int x1 = xyxy.x, y1 = xyxy.y, x2 = xyxy.x + xyxy.width, y2 = xyxy.y + xyxy.height;

    int y1_ex = 1, y1_ex_tol = 0;
    while (true) {
        int current_y = y1 - y1_ex;
        if (current_y < 0 || y1_ex > 50) {
            y1_ex = y1_ex - 1 - y1_ex_tol;
            break;
        }
        cv::Rect roi_y1_ex(x1, current_y, x2 - x1, y1_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y1_ex), thresh);
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
        if (y2 + y2_ex >= H || y2_ex > 50) {
            y2_ex = y2_ex - 1 - y2_ex_tol;
            break;
        }
        cv::Rect roi_y2_ex(x1, y2, x2 - x1, y2_ex);
        auto [spans, l, r] = count_high_rg_pixels_per_row(im(roi_y2_ex), thresh);
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
    return {y1_ex, y2_ex};
}

std::pair<double, int> FireDetector::is_circle(const std::vector<int> &lst) {
    /* ... same as before ... */
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
    /* ... same as before ... */
    if (lst.size() < 3) return {0.0, -1};
    int h = lst.size();
    if (h == 0) return {0.0, -1};
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
        return good / (double) (seq.size() - 1);
    };

    std::vector<int> left(lst.begin(), lst.begin() + peak_index + 1);
    std::vector<int> right(lst.begin() + peak_index, lst.end());

    double inc_ratio = monotonic_ratio(left, true);
    double dec_ratio = monotonic_ratio(right, false);

    return {0.5 + 0.5 * (inc_ratio + dec_ratio) / 2.0, peak_index};
}

std::vector<int> FireDetector::exponential_smoothing(const std::vector<int> &span_list, double alpha) {
    /* ... same as before ... */
    if (span_list.empty()) return {};
    std::vector<double> smoothed(span_list.size());
    smoothed[0] = span_list[0];
    for (size_t i = 1; i < span_list.size(); ++i) {
        smoothed[i] = alpha * span_list[i] + (1 - alpha) * smoothed[i - 1];
    }
    std::vector<int> result;
    for (double val: smoothed) result.push_back(static_cast<int>(round(val)));
    return result;
}

std::pair<int, double> FireDetector::find_most_significant_valley(const std::vector<int> &span_list) {
    /* ... same as before ... */
    int n = span_list.size();
    if (n < 3) return {-1, 0.0};

    auto smoothed_list = exponential_smoothing(span_list);

    std::vector<int> left_max(n, 0), left_max_smooth_arr(n, 0), right_max(n, 0), right_max_smooth_arr(n,
                                                                                                      0), left_max_avgidx(
            n, 0), right_max_avgidx(n, 0);
    left_max[0] = span_list[0];
    left_max_smooth_arr[0] = smoothed_list[0];
    for (int i = 1; i < n; ++i) {
        left_max[i] = std::max(left_max[i - 1], span_list[i]);
        left_max_smooth_arr[i] = std::max(left_max_smooth_arr[i - 1], smoothed_list[i]);
        int best_idx = -1;
        for (int j = i; j >= 0; --j)
            if (smoothed_list[j] == left_max_smooth_arr[i]) {
                best_idx = j;
                break;
            }
        left_max_avgidx[i] = best_idx;
    }
    right_max[n - 1] = span_list[n - 1];
    right_max_smooth_arr[n - 1] = smoothed_list[n - 1];
    right_max_avgidx[n - 1] = n - 1;
    for (int i = n - 2; i >= 0; --i) {
        right_max[i] = std::max(right_max[i + 1], span_list[i]);
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
    /* ... same as before ... */
    if (lst.size() <= 3) return {0.0, -1, 0.0};

    auto [valley_idx, best_depth] = find_most_significant_valley(lst);
    if (valley_idx == -1) return {0.0, -1, 0.0};

    double left_peak = *std::max_element(lst.begin(), lst.begin() + valley_idx);
    double right_peak = lst.size() > valley_idx + 1 ? *std::max_element(lst.begin() + valley_idx + 1, lst.end()) : 0;
    double peak = std::max(left_peak, right_peak);
    if (peak == 0 || lst[valley_idx] >= std::min(left_peak, right_peak)) return {0.0, -1, 0.0};

    double score = 1.0 - (double) lst[valley_idx] / peak;
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

std::pair<double, int> FireDetector::is_diamond(const std::vector<int> &lst) { return {0.0, -1}; }

std::pair<double, int> FireDetector::is_rectangle(const std::vector<int> &lst) { return {0.0, -1}; }

FireLocateResult FireDetector::shape_process(const std::vector<int> &span_list, const cv::Mat &im, const cv::Rect &xxyy,
                                             const std::vector<int> &left_zeros, const RectD &ext_xxyy, int path_idx) {
    /* ... same as before, with double casts ... */
    if (span_list.empty()) return {5, {xxyy.x + xxyy.width / 2.0, xxyy.y + xxyy.height * 4.0 / 5.0}, 0.0, {}};

    int x1 = xxyy.x, y1 = xxyy.y;
    int x2 = x1 + xxyy.width, y2 = y1 + xxyy.height;

    auto [circle_score, circle_id] = is_circle(span_list);
    if (circle_score >= 0.5) {
        return {0,
                {(double) (x1 + x2) / 2.0, (double) y1 + ((double) circle_id + (double) (y2 - y1) * 6.0 / 7.0) / 2.0},
                circle_score * SHAPE_SCORES[0], {}};
    }
    auto [funnel_score, funnel_id, funnel_depth] = is_funnel(span_list);
    if (funnel_score >= 0.5) {
        int coord_line_idx = std::max(1, (int) round((funnel_id + 1) * 4.0 / 5.0));
        int line_width = span_list[coord_line_idx - 1];
        int coord_x = (line_width <= 2) ? (*std::max_element(span_list.begin(), span_list.end()) - 1) / 2 : (
                left_zeros[coord_line_idx - 1] + line_width / 2);
        return {1, {(double) x1 + coord_x, (double) y1 + coord_line_idx - 1}, funnel_score * SHAPE_SCORES[1], {}};
    }
    auto [short_score, short_id] = is_short(span_list);
    if (short_score >= 0.5) {
        return {2, {(double) (x1 + x2) / 2.0, (double) y1 + ((double) short_id + (double) (y2 - y1) * 6.0 / 7.0) / 2.0},
                short_score * SHAPE_SCORES[2], {}};
    }
    // Fallback
    return {5, {(double) (x1 + x2) / 2.0, (double) y1 + (double) (y2 - y1) * 4.0 / 5.0},
            span_list.empty() ? 0.0 : SHAPE_SCORES[5], {}};
}

FireLocateResult
FireDetector::fire_locate(const cv::Mat &im, const RectD &bbox_d, const RectD &ext_xxyy, int path_idx) {
    /* ... same as before ... */
    cv::Rect bbox(bbox_d.x, bbox_d.y, bbox_d.width, bbox_d.height);
    std::vector<FireLocateResult> results;

    cv::Mat roi = im(bbox);
    cv::Mat rb_avg_roi = cal_rb(roi);
    cv::Mat rb_flat = rb_avg_roi.reshape(1, 1);
    cv::sort(rb_flat, rb_flat, cv::SORT_ASCENDING);
    int start_idx = rb_flat.cols * 0.5;
    double adaptive_thresh_val = cv::mean(rb_flat.colRange(start_idx, rb_flat.cols))[0];
    int adaptive_thresh = std::max(140.0, adaptive_thresh_val);

    std::vector<int> threshes = {-1, 200, adaptive_thresh};

    for (int thresh: threshes) {
        auto [y1_ex, y2_ex] = refine_bbox(im, bbox, thresh);
        int new_y1 = bbox.y - y1_ex, new_y2 = bbox.y + bbox.height + y2_ex;
        if (new_y1 < 0 || new_y2 >= im.rows || new_y1 >= new_y2) continue;

        cv::Rect roi_ex_rect(bbox.x, new_y1, bbox.width, new_y2 - new_y1);
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
    if (results.empty()) return {5, {bbox.x + bbox.width / 2.0, bbox.y + bbox.height * 0.8}, 0.1, {}};

    auto best_it = std::max_element(results.begin(), results.end(),
                                    [](const FireLocateResult &a, const FireLocateResult &b) {
                                        return a.weight < b.weight;
                                    });
    return *best_it;
}

double FireDetector::calculate_iou(const RectD &box1, const RectD &box2) {
    /* ... same as before ... */
    RectD inter_rect = box1 & box2;
    double union_area = box1.area() + box2.area() - inter_rect.area();
    return (union_area < 1e-9) ? 0.0 : inter_rect.area() / union_area;
}

std::vector<NmsBBoxInfo> FireDetector::merge_rects(const std::vector<NmsBBoxInfo> &boxes) {
    /* ... same as before ... */
    if (boxes.empty()) return {};
    std::vector<NmsBBoxInfo> merged_rects;
    std::vector<bool> merged(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (merged[i]) continue;
        RectD current_rect = boxes[i].box;
        double max_score = boxes[i].score;
        merged[i] = true;

        bool changed_in_iteration = true;
        while (changed_in_iteration) {
            changed_in_iteration = false;
            for (size_t j = 0; j < boxes.size(); ++j) {
                if (merged[j] || i == j) continue;
                if (calculate_iou(current_rect, boxes[j].box) > 1e-9) {
                    current_rect |= boxes[j].box;
                    max_score = std::max(max_score, boxes[j].score);
                    merged[j] = true;
                    changed_in_iteration = true;
                }
            }
        }
        merged_rects.push_back({max_score, 0, current_rect});
    }
    return merged_rects;
}