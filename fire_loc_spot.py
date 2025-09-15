import numpy as np
import cv2, os, glob
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# Shape names and their corresponding scores
SHAPE_NAMES = ['圆', '葫芦', '类圆', '菱形', '长条', '其他']
SHAPE_SCORES = [0.72, 0.9, 0.7, 0.5, 0.3, 0.1]

W, H = 1920, 1080


def cal_rb(im):
    b = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    rb_avg = b * 0.5 + g * 0.5
    return rb_avg


def count_high_rg_pixels_per_row(crop, thresh=None, ret_lr=False):
    """
    Compute per-row horizontal spans of high green/blue-average pixels.
    Optionally return left/right zero-run lengths for each row to help locate centers.

    Args:
        crop: HxWx3 BGR image ROI.
        thresh: Int threshold for (B+G)/2.
        ret_lr: If True, also return left and right zero counts for each row.

    Returns:
        span_list or (span_list, left_zeros, right_zeros)
    """
    rb_avg = cal_rb(crop)
    if thresh is None:
        thresh = 227
    mask = rb_avg >= thresh

    span_list = []
    left_zeros = []
    right_zeros = []
    for i in range(mask.shape[0]):
        row_mask = mask[i, :]
        true_indices = np.where(row_mask)[0]
        if len(true_indices) == 0:
            span_list.append(0)
            if ret_lr:
                left_zeros.append(len(row_mask))
                right_zeros.append(len(row_mask))
        else:
            span = true_indices[-1] - true_indices[0] + 1
            span_list.append(span)
            if ret_lr:
                left_zeros.append(true_indices[0])
                right_zeros.append(len(row_mask) - (true_indices[-1]) - 1)
    if ret_lr:
        return span_list, left_zeros, right_zeros
    return span_list


class FireShape:
    @staticmethod
    def is_circle(lst):
        """Heuristic check whether a vertical span profile resembles a circle."""
        if len(lst) < 3:
            return 0.0, None
        h = len(lst)
        w = max(lst)
        if h / w > 1.5 or len(lst) < 3:
            return 0.0, None
        lst_max = np.max(lst)
        lst_thresh = lst_max * 0.2
        cur_phase1 = len(lst) - 1
        last_max_value = None
        err_phase1 = [0]
        for l_idx, l_value in enumerate(lst):
            if last_max_value is None:
                last_max_value = l_value
                continue
            if l_value - last_max_value < -lst_thresh:
                cur_phase1 = l_idx - 1
                break
            if l_value < last_max_value:
                err_phase1.append(err_phase1[-1] + abs(l_value - last_max_value))
            else:
                err_phase1.append(err_phase1[-1])
            last_max_value = max(last_max_value, l_value)
        # print(cur_phase1)

        err_phase2 = [0]
        cur_phase2 = len(lst) - 1
        last_max_value = None
        for l_idx, l_value in enumerate(lst[::-1]):
            if last_max_value is None:
                last_max_value = l_value
                continue
            if l_value - last_max_value < -lst_thresh:
                cur_phase2 = l_idx - 1
                break
            if l_value < last_max_value:
                err_phase2.append(err_phase2[-1] + abs(l_value - last_max_value))
            else:
                err_phase2.append(err_phase2[-1])
            last_max_value = max(last_max_value, l_value)
        cur_phase2 = len(lst) - cur_phase2 - 1
        # print(cur_phase2)
        # print(lst)
        if cur_phase1 >= cur_phase2 and cur_phase2 != 0:
            pass
            # print(
            #     f'yes, it is circle, , cur1={cur_phase1}[{lst[cur_phase1]}], cur2={cur_phase2}[{lst[cur_phase2]}]')
        else:
            # print(f'no, cur1={cur_phase1}[{lst[cur_phase1]}], cur2={cur_phase2}[{lst[cur_phase2]}]')
            return 0.0, None

        err_min = None
        err_min_idx = None
        for cur_i in range(cur_phase2, cur_phase1 + 1):
            # print(f'cur_i={cur_i},err_phase1={err_phase1},err_phase2={err_phase2}')
            err_i = err_phase1[cur_i] + err_phase2[len(lst) - 1 - cur_i]
            if err_min is None:
                err_min = err_i
                err_min_idx = cur_i
                continue
            if err_min > err_i or (err_min == err_i and abs(cur_i - len(lst) // 2) <= abs(err_min_idx - len(lst) // 2)):
                err_min = err_i
                err_min_idx = cur_i
            pass

        peak_index = err_min_idx
        center_bias = abs(peak_index - (len(lst) - 1) / 2)
        center_bias = 1 - center_bias / (len(lst))
        # print(f'err_min={err_min}, peak_index={peak_index}')

        err_score = 1 - err_min / max(lst_max, len(lst))
        err_score = max(0.5, min(1.0, err_score))
        err_score *= center_bias
        return err_score, peak_index

    @staticmethod
    def is_short(lst):
        """Check for a short, mound-like peak (roughly round/short shape)."""
        if len(lst) < 3:
            return 0.0, None
        h = len(lst)
        w = max(lst)
        if h / w > 1.5 or len(lst) < 3:
            return 0.0, None

        max_val = max(lst)
        max_indices = [i for i, v in enumerate(lst) if v == max_val]
        peak_index = max_indices[len(max_indices) // 2]

        left = lst[:peak_index + 1]
        right = lst[peak_index:]

        def monotonic_ratio(seq, increasing=True):
            if len(seq) < 2:
                return 1.0
            good = 0
            for a, b in zip(seq, seq[1:]):
                if increasing:
                    if b >= a:
                        good += 1
                else:
                    if b <= a:
                        good += 1
            return good / (len(seq) - 1)

        inc_ratio = monotonic_ratio(left, increasing=True)
        dec_ratio = monotonic_ratio(right, increasing=False)

        score_raw = (inc_ratio + dec_ratio) / 2
        final_score = 0.5 + 0.5 * score_raw
        final_score = round(final_score, 3)
        return final_score, peak_index

    @staticmethod
    def is_funnel(lst, verbose=False):
        """Detect a funnel/figure-8 valley pattern within the span profile."""
        arr = np.array(lst, dtype=float)
        n = len(arr)
        if n <= 3:
            return 0.0, None, 0

        def exponential_smoothing(span_list, alpha=0.8):
            """
            Apply exponential smoothing to the input list
            Args:
                span_list: Input list to be smoothed
                alpha: Smoothing factor (0 < alpha < 1)
            Returns:
                Smoothed list
            """
            arr = np.array(span_list, dtype=float)
            smoothed = np.zeros_like(arr)
            smoothed[0] = arr[0]
            for i in range(1, len(arr)):
                smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
            return np.int_(smoothed.tolist())

        def find_most_significant_valley(span_list):
            n = len(span_list)
            if n < 3:
                return None, 0

            # Precompute left_max_array: left_max_array[i] is the max value from 0 to i
            smoothed_span_list = exponential_smoothing(span_list)
            left_max_array = [0] * n
            left_max_smooth_arr = [0] * n
            left_max_array[0] = span_list[0]
            left_max_smooth_arr[0] = smoothed_span_list[0]
            left_max_avgidx = [0]
            for i in range(1, n):
                left_max_array[i] = max(left_max_array[i - 1], span_list[i])
                left_max_smooth_arr[i] = max(left_max_smooth_arr[i - 1], smoothed_span_list[i])
                tmp_all_idxes = np.where(smoothed_span_list == left_max_smooth_arr[i])[0]
                left_max_avgidx.append(np.max(tmp_all_idxes[tmp_all_idxes <= i]))

            # Precompute right_max_array: right_max_array[i] is the max value from i to n-1
            right_max_array = [0] * n
            right_max_smooth_arr = [0] * n
            right_max_array[n - 1] = span_list[n - 1]
            right_max_smooth_arr[n - 1] = smoothed_span_list[n - 1]
            right_max_avgidx = [n - 1] * n
            for i in range(n - 2, -1, -1):
                right_max_array[i] = max(right_max_array[i + 1], span_list[i])
                right_max_smooth_arr[i] = max(right_max_smooth_arr[i + 1], smoothed_span_list[i])
                tmp_all_idxes = np.where(smoothed_span_list == right_max_smooth_arr[i])[0]
                right_max_avgidx[i] = np.min(tmp_all_idxes[tmp_all_idxes >= i])

            # Find all candidate valley points: internal point i is a valley if its value is less than or equal to its neighbors
            valleys = []
            for i in range(1, n - 1):
                if span_list[i] <= span_list[i - 1] and span_list[i] <= span_list[i + 1]:
                    valleys.append(i)

            if not valleys:
                return None, 0  # No candidate valleys

            # Define the middle region
            low_index = n // 4
            high_index = 3 * n // 4
            middle_valleys = [i for i in valleys if low_index <= i <= high_index]
            if not middle_valleys:
                middle_valleys = valleys  # If no middle region valleys, use all valleys

            # Calculate the center position of the list
            center = (n - 1) / 2.0
            best_valley = None
            best_depth = -1
            best_depth0 = -1
            depth_list = []

            for i in middle_valleys:
                # Get the max value on the left (0 to i-1) and right (i+1 to n-1)
                left_max = left_max_array[i - 1]
                right_max = right_max_array[i + 1]
                depth = min(left_max - span_list[i], right_max - span_list[i])
                l_slope = (left_max - span_list[i]) / np.sqrt(1 + abs(i - left_max_avgidx[i - 1]))
                r_slope = (right_max - span_list[i]) / np.sqrt(1 + abs(i - right_max_avgidx[i + 1]))
                depth0 = np.mean([l_slope, r_slope])
                depth_list.append([i, depth, l_slope, r_slope])
                if depth <= 0:
                    continue  # Ignore points with non-positive depth

                if best_valley is None or depth > best_depth:
                    best_valley = i
                    best_depth = depth
                    best_depth0 = depth0
                elif depth == best_depth:
                    # Depth is the same, choose the point closer to the center
                    current_dist = abs(i - center)
                    best_dist = abs(best_valley - center)
                    if current_dist < best_dist:
                        best_valley = i
                        best_depth0 = depth0
            # print(depth_list)
            return best_valley, best_depth0  # Return index, or None if no valid point

        valley_rel_idx, best_depth = find_most_significant_valley(arr)
        if verbose:
            arr2 = np.int_(arr)
            max_width = max(len(str(x)) for x in arr2 + list(range(len(arr2))))
            # Format the index line with proper spacing
            idx_str = ' '.join(f"{i:{max_width}}" for i in range(len(arr2)))
            print(f'funnel_idx=[{idx_str}]')
            # Format the array line with the same spacing
            arr_str = ' '.join(f"{x:{max_width}}" for x in arr2)
            print(f'funnel_arr=[{arr_str}]')

            print(f'find_most_valy={valley_rel_idx}, {arr2[valley_rel_idx]}')

        if valley_rel_idx is None:
            return 0.0, None, 0

        left_peak_idx = int(np.argmax(arr[:valley_rel_idx]))
        left_peak = arr[left_peak_idx]
        right_half = arr[valley_rel_idx + 1:]

        right_peak_rel_idx = int(np.argmax(right_half))
        right_peak_idx = valley_rel_idx + 1 + right_peak_rel_idx
        right_peak = arr[right_peak_idx]

        peak = max(left_peak, right_peak)
        if peak == 0:
            return 0.0, None, 0

        valley = arr[valley_rel_idx]
        if valley >= min(left_peak, right_peak):
            return 0.0, None, 0

        score = 1 - valley / peak
        score = max(0.0, min(1.0, score))

        if score < 0.2:
            return 0.0, None, best_depth
        score = 0.5 + 0.5 * score

        depth_score = best_depth / 2
        if depth_score > 1:
            depth_score = min(1.25, np.sqrt(depth_score))
        len_score = 1.0
        len_i = len(arr)
        while len_i < 7:
            len_score *= 0.9
            len_i += 1
        value_score = 1.0
        value_i = max(arr)
        while value_i < 5:
            value_score *= 0.9
            value_i += 1
        score = score * depth_score * len_score * value_score
        if score < 0.35:
            return 0.0, None, best_depth
        # print(f'best_valley={valley_rel_idx},best_depth={best_depth}, len_score={len_score}, value_score={value_score}')

        return score, valley_rel_idx, best_depth * len_score * value_score

    @staticmethod
    def is_diamond(lst):
        """Detect a diamond-like shape with mid bulge and thinner edges."""
        arr = np.array(lst, dtype=float)
        if len(arr) < 3:
            return 0.0, None

        seq_id = max(len(arr) // 5, 1)

        pre_seq = arr[0:seq_id]
        mid_seq = arr[seq_id:-seq_id]
        after_seq = arr[-seq_id:]
        pre_max = max(pre_seq)

        mid_max = max(mid_seq)
        max_positions = np.where(mid_seq == mid_max)[0]
        center_index = (len(mid_seq) - 1) / 2
        peak_idx = min(max_positions, key=lambda x: abs(x - center_index))

        after_max = max(after_seq)
        edge_avg = max(pre_max, after_max)

        ratio = edge_avg / mid_max

        score = 1 - ratio
        if score < 0.3:
            return 0.0, None
        score = max(0.0, min(1.0, score))
        score = 0.5 + 0.5 * score

        return score, peak_idx + seq_id

    @staticmethod
    def is_rectangle(lst):
        """Detect a tall, rectangular-like profile (long vertical extent)."""
        if len(lst) < 3:
            return 0.0, None
        arr = np.array(lst, dtype=float)
        h = len(lst)
        w = max(lst)
        if h / w < 2 or len(lst) < 3:
            return 0.0, None

        return 1, None


def calculate_iou(box1, box2):
    """Intersection-over-Union between two boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def shape_process(span_list, im, xxyy, left_zeros, global_offset_x, global_offset_y, ext_xxyy=None, std_pts=None, path_idx=0):
    """
    Given a span profile and ROI, classify shape and compute a representative coordinate.
    Returns visualization mosaic for inspection alongside score and shape id.
    """
    x1, x2, y1, y2 = xxyy
    sx1, sx2, sy1, sy2 = ext_xxyy
    x1_show, x2_show, y1_show, y2_show = max(0, x1 - 2), min(W, x2 + 2), max(0, y1 - 2), min(H, y2 + 2)
    roi = im[y1_show:y2_show, x1_show:x2_show]
    roi = cv2.resize(roi, np.int_(3 * np.array(roi.shape[:2][::-1])))
    im2 = im.copy()
    if std_pts is not None:
        for pt in std_pts:
            cv2.circle(im2, pt, 2, (0, 0, 255), -1)
    cv2.rectangle(im2, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 200, 0), 1)
    is_circle_score, is_circle_id = FireShape.is_circle(span_list)
    if is_circle_score >= 0.5:
        shape_id = 0
        coord = np.int_(((x1 + x2) // 2 + global_offset_x,
                         int(((y1 + is_circle_id) + (y1 + (y2 - y1) * 6 / 7)) / 2) + global_offset_y))
        cv2.circle(im2, coord, 1, (0, 255, 0), -1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
        cv2.putText(im3,
                    f"[{path_idx}]ball({SHAPE_SCORES[shape_id]})*sim({is_circle_score:.1f})={is_circle_score * SHAPE_SCORES[shape_id]:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im3, f"{span_list}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
        im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
        im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
        return coord, is_circle_score * SHAPE_SCORES[shape_id], shape_id, im4
    is_funnel_score, is_funnel_id, best_depth = FireShape.is_funnel(span_list)
    if is_funnel_score >= 0.5:
        shape_id = 1
        upper_part = span_list[:is_funnel_id + 1]
        upper_part = span_list[:len(upper_part) - next((i for i, x in enumerate(reversed(upper_part)) if x != 0), 0)]
        upper_l_zeros = left_zeros[:is_funnel_id + 1]
        # upper_r_zeros = right_zeros[:is_funnel_id+1]
        coord_line_idx = max(1, round(len(upper_part) * 4 / 5))
        line_width = upper_part[coord_line_idx - 1]
        if line_width <= 2:
            total_width = max(span_list)
            coord_x = int((total_width - 1) / 2)
        else:
            edge_value_thresh = min(5, max(upper_part) * 0.2)
            edge_idx = None
            for c_idx in range(0, len(upper_part))[::-1]:
                if upper_part[c_idx] <= edge_value_thresh:
                    edge_idx = c_idx
                else:
                    break
            if edge_idx is not None and edge_idx <= coord_line_idx - 1:
                coord_line_idx = edge_idx
                line_width = upper_part[coord_line_idx - 1]
            coord_x = upper_l_zeros[coord_line_idx - 1] + line_width // 2
            if line_width % 2 == 0:
                total_width = max(span_list)
                if (total_width - 1) / 2 < coord_x:
                    coord_x -= 1
        coord_x += x1 + global_offset_x
        coord_y = coord_line_idx + y1 + global_offset_y - 1
        coord = (coord_x, coord_y)
        cv2.circle(im2, coord, 1, (0, 255, 0), -1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
        cv2.putText(im3,
                    f"[{path_idx}]8({SHAPE_SCORES[shape_id]})*sim({is_funnel_score:.1f})={is_funnel_score * SHAPE_SCORES[shape_id]:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im3, f"{span_list}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(im3, f"d={best_depth:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
        im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
        im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
        return coord, is_funnel_score * SHAPE_SCORES[shape_id], shape_id, im4
    is_short_score, is_short_id = FireShape.is_short(span_list)
    if is_short_score >= 0.5:
        shape_id = 2
        coord = np.int_(((x1 + x2) // 2 + global_offset_x, int(((y1 + is_short_id) + (
                y1 + (y2 - y1) * 6 / 7)) / 2) + global_offset_y))
        cv2.circle(im2, coord, 1, (0, 255, 0), -1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
        cv2.putText(im3,
                    f"[{path_idx}]short({SHAPE_SCORES[shape_id]})*sim({is_short_score:.1f})={SHAPE_SCORES[shape_id] * is_short_score:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im3, f"{span_list}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
        im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
        im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
        return coord, is_short_score * SHAPE_SCORES[shape_id], shape_id, im4
    is_diamond_score, is_diamond_id = FireShape.is_diamond(span_list)
    if is_diamond_score >= 0.5:
        shape_id = 3
        coord = np.int_(((x1 + x2) // 2 + global_offset_x,
                         int(((y1 + is_diamond_id) + (y1 + (len(span_list) - 1) / 2)) / 2) + global_offset_y))
        cv2.circle(im2, coord, 1, (0, 255, 0), -1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
        cv2.putText(im3,
                    f"[{path_idx}]diamond({SHAPE_SCORES[shape_id]})*sim({is_diamond_score:.1f})={is_diamond_score * SHAPE_SCORES[shape_id]:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im3, f"{span_list}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
        im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
        im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
        return coord, is_diamond_score * SHAPE_SCORES[shape_id], shape_id, im4
    is_rectangle_score, is_rectangle_id = FireShape.is_rectangle(span_list)
    if is_rectangle_score >= 0.5:
        shape_id = 4
        coord = np.int_(((x1 + x2) // 2 + global_offset_x, int(y1 + (y2 - y1) * 3 / 5) + global_offset_y))
        cv2.circle(im2, coord, 1, (0, 255, 0), -1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
        cv2.putText(im3,
                    f"[{path_idx}]slim({SHAPE_SCORES[shape_id]})*sim({is_rectangle_score:.1f})={is_rectangle_score * SHAPE_SCORES[shape_id]:.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im3, f"{span_list}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
        im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
        im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
        return coord, is_rectangle_score * SHAPE_SCORES[shape_id], shape_id, im4
    # Fallback: other shape
    shape_id = 5
    coord = np.int_(((x1 + x2) // 2 + global_offset_x, int(y1 + (y2 - y1) * 4 / 5) + global_offset_y))
    cv2.circle(im2, coord, 1, (0, 255, 0), -1)
    im3 = im2[sy1:sy2, sx1:sx2].copy()
    im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))
    len_score = 1.0 if len(span_list) > 0 else 0.0
    cv2.putText(im3,
                f"[{path_idx}]others({SHAPE_SCORES[shape_id]})*sim({len_score})={SHAPE_SCORES[shape_id] * len_score}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(im3, f"{span_list}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    im4 = np.zeros((im3.shape[0] + roi.shape[0], max(im3.shape[1], roi.shape[1]), 3), dtype=np.uint8)
    im4[:im3.shape[0], :im3.shape[1], :] = im3.copy()
    im4[im3.shape[0]:, :roi.shape[1], :] = roi.copy()
    return coord, SHAPE_SCORES[shape_id] * len_score, shape_id, im4


def refine_bbox(im, xyxy, thresh=None, up_tol=3, down_tol=2):
    """Refine top/bottom of a bbox by scanning for contiguous high (B,G) rows."""
    x1, y1, x2, y2 = np.int_(xyxy)
    y1_ex, y2_ex = 1, 1
    y1_ex_tol, y2_ex_tol = 0, 0
    while True:
        roi_y1_ex = im[y1 - y1_ex:y1, x1:x2]
        y1_ex_span_list = count_high_rg_pixels_per_row(roi_y1_ex, thresh)
        if y1_ex_span_list[0] == 0:
            if y1_ex_tol + 1 > up_tol:
                y1_ex = y1_ex - 1 - y1_ex_tol
                break
            y1_ex_tol += 1
        else:
            y1_ex_tol = 0
        y1_ex += 1
        if y1_ex > 50 or y1 - y1_ex < 0:
            y1_ex = y1_ex - 1 - y1_ex_tol
            break
    if y1_ex == 0:
        y1_ex = -1
        while True:
            roi_y1_ex = im[y1: y1 - y1_ex, x1:x2]
            y1_ex_span_list = count_high_rg_pixels_per_row(roi_y1_ex, thresh)
            if y1_ex_span_list[-1] != 0:
                y1_ex = y1_ex + 1
                break
            y1_ex -= 1
            if y1_ex < -(y2 - y1):
                y1_ex += 1
                break
    while True:
        roi_y2_ex = im[y2:y2 + y2_ex, x1:x2]
        y2_ex_span_list = count_high_rg_pixels_per_row(roi_y2_ex, thresh)
        if y2_ex_span_list[-1] == 0:
            if y2_ex_tol + 1 > down_tol:
                y2_ex = y2_ex - 1 - y2_ex_tol
                break
            y2_ex_tol += 1
        else:
            y2_ex_tol = 0
        y2_ex += 1
        if y2_ex > 50 or y2 + y2_ex >= H:
            y2_ex = y2_ex - 1 - y2_ex_tol
            break
    if y2_ex == 0:
        y2_ex = -1
        while True:
            roi_y2_ex = im[y2 + y2_ex: y2, x1:x2]
            y2_ex_span_list = count_high_rg_pixels_per_row(roi_y2_ex, thresh)
            if y2_ex_span_list[0] != 0:
                y2_ex = y2_ex + 1
                break
            y2_ex -= 1
            if y2_ex < -50 or y2 + y2_ex <= y1 - y1_ex:
                y2_ex += 1
                break
    # print(f'expand pixels: up={y1_ex},down={y2_ex}')
    return y1_ex, y2_ex


def select_evenly_distributed(paths, max_element=100):
    """Pick up to max_element items spread evenly across the list order."""
    n = min(len(paths), max_element)
    step = len(paths) / n
    indices = [int(i * step) for i in range(n)]
    indices = [min(i, len(paths) - 1) for i in indices]
    selected = [paths[i] for i in indices]
    return selected



def merge_rects(rects):
    """Greedily merge overlapping or nearby rectangles into unions."""
    merged_flag = True
    merge_count = 0
    i_max = len(rects)
    while merged_flag:
        merge_count += 1
        merged_flag = False
        new_rects = []
        newly_update_rects = []
        merged_set = set()
        for i, rect1 in enumerate(rects):
            if i in merged_set:
                continue
            if i >= i_max:
                new_rects.append(rect1)
                continue
            merged = False
            for j, rect2 in enumerate(rects):
                if i < j and j not in merged_set and i not in merged_set:
                    xa1, ya1, xa2, ya2 = rect1
                    xb1, yb1, xb2, yb2 = rect2
                    box1 = (xa1, ya1, xa2 - xa1, ya2 - ya1)
                    box2 = (xb1, yb1, xb2 - xb1, yb2 - yb1)
                    if calculate_iou(box1, box2) != 0 or \
                            ((abs(yb1 - ya2) < max(4, (ya2 - ya1), yb2 - yb1) or abs(ya1 - yb2) < max(4, (ya2 - ya1),  yb2 - yb1)) and \
                             (abs(xb1 - xa2) < max(4, (xa2 - xa1), xb2 - xb1) or abs(xa1 - xb2) < max(4, (xa2 - xa1), xb2 - xb1))):
                        xcombine_1 = min(xa1, xb1)
                        xcombine_2 = max(xa2, xb2)
                        ycombine_1 = min(ya1, yb1)
                        ycombine_2 = max(ya2, yb2)
                        newly_update_rects.append((int(xcombine_1), int(ycombine_1), int(xcombine_2), int(ycombine_2)))
                        merged_set.add(i)
                        merged_set.add(j)
                        merged = True
                        merged_flag = True
                        continue
            if not merged:
                new_rects.append(rect1)
        rects = newly_update_rects
        i_max = len(rects)
        for r in new_rects:
            rects.append(r)
    return rects

def fire_shape_analysis(im, ori_labels, std_pts):
    conf = 0.0
    coord = (0, 0)

    # Compute inclusive ROI around the standard points, expanded by a ratio
    sx1, sy1 = np.min(std_pts, axis=0)
    sx2, sy2 = np.max(std_pts, axis=0)
    sw, sh = sx2 - sx1, sy2 - sy1
    expand_exclude_ratio = 0.5
    sx1 = int(max(0, sx1 - sw * expand_exclude_ratio))
    sy1 = int(max(0, sy1 - sh * expand_exclude_ratio))
    sw = int(min(sw * (1 + 2 * expand_exclude_ratio), W - sx1))
    sh = int(min(sh * (1 + 2 * expand_exclude_ratio), H - sy1))
    sx2 = sx1 + sw
    sy2 = sy1 + sh
    ext_xxyy = sx1, sx2, sy1, sy2

    responding_labels = []
    merged_labels = []

    if len(ori_labels) >= 1:
        imo = im.copy()
        imr = im.copy()
        imm = im.copy()
        for l in ori_labels:
            x1, y1, x2, y2 = np.int_(l)
            cv2.rectangle(imo, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (0, 200, 0), 1)
            if calculate_iou((x1, y1, x2 - x1, y2 - y1), (sx1, sy1, sx2 - sx1, sy2 - sy1)) == 0:
                continue
            responding_labels.append([x1, y1, x2, y2])
            cv2.rectangle(imr, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (0, 0, 200), 1)
        merged_labels = merge_rects(responding_labels)
        for ml in merged_labels:
            x1, y1, x2, y2 = np.int_(ml)
            cv2.rectangle(imm, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (200, 0, 0), 1)
        imo = imo[sy1:sy2, sx1:sx2].copy()
        imo = cv2.resize(imo, np.int_(3 * np.array(imo.shape[:2][::-1])))
        imr = imr[sy1:sy2, sx1:sx2].copy()
        imr = cv2.resize(imr, np.int_(3 * np.array(imr.shape[:2][::-1])))
        imm = imm[sy1:sy2, sx1:sx2].copy()
        imm = cv2.resize(imm, np.int_(3 * np.array(imm.shape[:2][::-1])))
        cv2.putText(imo, f"{len(ori_labels)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imr, f"{len(responding_labels)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imm, f"{len(merged_labels)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("imo", imo)
        cv2.imshow("imr", imr)
        cv2.imshow("imm", imm)

    if len(merged_labels) > 1:
        print(f'Error: len={len(merged_labels)}')
        return conf, coord

    if len(merged_labels) == 0 and len(responding_labels) != 0:
        print(f'Error: len={len(merged_labels)}/{len(responding_labels)}')
        cv2.waitKey()
        return conf, coord
    # print(merged_labels)
    if len(merged_labels) == 1:
        global_offset_x, global_offset_y = 0, 0
        x1, y1, x2, y2 = merged_labels[0]
        # if calculate_iou((x1, y1, x2 - x1, y2 - y1), (sx1, sy1, sx2 - sx1, sy2 - sy1)) == 0:
        #     return conf, coord
        # Run shape inference under default threshold

        shape_ids, coords, weights, im_list = fire_locate(im, bbox=(x1,y1,x2,y2), global_offset_x=global_offset_x, global_offset_y=global_offset_y, ext_xxyy=ext_xxyy, path_idx=path_idx)
        shape_id_default, shape_id_200, shape_id_auto = shape_ids
        coord_default, coord_200, coord_auto, coord_best = coords
        weight_default, weight_200, weight_auto, weight_best = weights
        im_vis_default, im_vis_200, im_vis_auto, im_best = im_list

        print(f"{path_idx}:{path}")
        if not shape_id_default == shape_id_200 == shape_id_auto:
            print(
                f'**ex:{SHAPE_NAMES[shape_id_default]}, 200:{SHAPE_NAMES[shape_id_200]}, a:{SHAPE_NAMES[shape_id_auto]}')
        else:
            print(f'{SHAPE_NAMES[shape_id_default]}')
        if coord_200[0] == coord_default[0] == coord_auto[0] and coord_200[1] == coord_default[1] == coord_auto[1]:
            print(f'all clear:{coord_default}')
        else:
            print(f'coord_def =\t{coord_default}\ncoord_200 =\t{coord_200}\ncoord_auto=\t{coord_auto}')
            print(f'coord_best=\t{coord_best}')
        print("==" * 20)

        im2 = im.copy()
        for pt in std_pts:
            cv2.circle(im2, pt, 2, (0, 0, 255), -1)
        for l in ori_labels:
            x1, y1, x2, y2 = np.int_(l)
            if calculate_iou((x1, y1, x2 - x1, y2 - y1), (sx1, sy1, sx2 - sx1, sy2 - sy1)) == 0:
                continue
            cv2.rectangle(im2, (x1 - 1, y1 - 1), (x2 + 2, y2 + 2), (0, 200, 0), 1)
        im3 = im2[sy1:sy2, sx1:sx2].copy()
        im3 = cv2.resize(im3, np.int_(3 * np.array(im3.shape[:2][::-1])))

        cv2.imshow("im0", im3)
        cv2.imshow("im-default", im_vis_default)
        cv2.imshow("im-200", im_vis_200)
        cv2.imshow("im-auto", im_vis_auto)
        cv2.imshow("im-best", im_best)
        cv2.waitKey()
        conf, coord = weight_best, coord_best
    return conf, coord

def fire_locate(im, bbox, global_offset_x=0, global_offset_y=0, ext_xxyy=None, ret_all=True, path_idx=0):
    x1, y1, x2, y2 = np.int_(bbox)
    y1_ex, y2_ex = refine_bbox(im, (x1, y1, x2, y2))
    roi_ex = im[y1 - y1_ex:y2 + y2_ex, x1:x2]
    span_list_ex, left_zeros, right_zeros = count_high_rg_pixels_per_row(roi_ex, ret_lr=True)
    x1_ex = min(left_zeros) if len(left_zeros) > 0 else 0
    x2_ex = min(right_zeros) if len(right_zeros) > 0 else 0
    if x1_ex != 0:
        left_zeros = np.array(left_zeros) - x1_ex
    # if x2_ex != 0:
    #     right_zeros = np.array(right_zeros) - x2_ex

    xxyy = x1 + x1_ex, x2 - x2_ex, y1 - y1_ex, y2 + y2_ex
    result_default = shape_process(span_list_ex, im, xxyy, left_zeros, global_offset_x, global_offset_y, ext_xxyy, path_idx=path_idx)
    coord_default, weight_default, shape_id_default, im_vis_default = result_default

    # Run shape inference under threshold=200
    y1_ex_200, y2_ex_200 = refine_bbox(im, (x1, y1, x2, y2), 200)
    roi_ex_200 = im[y1 - y1_ex_200:y2 + y2_ex_200, x1:x2]
    span_list_200, left_zeros_200, right_zeros_200 = count_high_rg_pixels_per_row(roi_ex_200, 200, ret_lr=True)
    x1_ex_200 = min(left_zeros_200) if len(left_zeros_200) > 0 else 0
    x2_ex_200 = min(right_zeros_200) if len(right_zeros_200) > 0 else 0
    if x1_ex_200 != 0:
        left_zeros_200 = np.array(left_zeros_200) - x1_ex_200
    # if x2_ex_200 != 0:
    #     right_zeros_200 = np.array(right_zeros_200) - x2_ex_200

    xxyy = x1 + x1_ex_200, x2 - x2_ex_200, y1 - y1_ex_200, y2 + y2_ex_200
    # Run shape inference with fixed threshold=200
    result_200 = shape_process(span_list_200, im, xxyy, left_zeros_200, global_offset_x, global_offset_y, ext_xxyy, path_idx=path_idx)
    coord_200, weight_200, shape_id_200, im_vis_200 = result_200

    # Compute adaptive threshold from brighter half of RB average
    roi = im[y1:y2, x1:x2]
    rb_avg_img = cal_rb(roi).astype(np.uint8)
    rb_flat_sorted = rb_avg_img.flatten()
    rb_flat_sorted.sort()
    # adaptive_thresh = np.mean(rb_flat_sorted[-1 * int(len(rb_flat_sorted) * 0.5):])
    adaptive_thresh = max(140, np.mean(rb_flat_sorted[-1 * int(len(rb_flat_sorted) * 0.5):]))
    y1_ex_auto, y2_ex_auto = refine_bbox(im, (x1, y1, x2, y2), adaptive_thresh)
    roi_ex_auto = im[y1 - y1_ex_auto:y2 + y2_ex_auto, x1:x2]
    span_list_auto, left_zeros_auto, right_zeros_auto = count_high_rg_pixels_per_row(roi_ex_auto, adaptive_thresh,
                                                                                     ret_lr=True)
    x1_ex_auto = min(left_zeros_auto) if len(left_zeros_auto) > 0 else 0
    x2_ex_auto = min(right_zeros_auto) if len(right_zeros_auto) > 0 else 0
    if x1_ex_auto != 0:
        left_zeros_auto = np.array(left_zeros_auto) - x1_ex_auto
    # if x2_ex_auto != 0:
    #     right_zeros_auto = np.array(right_zeros_auto) - x2_ex_auto

    xxyy = x1 + x1_ex_auto, x2 - x2_ex_auto, y1 - y1_ex_auto, y2 + y2_ex_auto
    # Run shape inference with adaptive threshold
    result_auto = shape_process(span_list_auto, im, xxyy, left_zeros_auto, global_offset_x, global_offset_y, ext_xxyy, path_idx=path_idx)
    coord_auto, weight_auto, shape_id_auto, im_vis_auto = result_auto

    im_best = im_vis_default
    weight_best = weight_default
    coord_best = coord_default
    shape_id_best = shape_id_default
    if weight_200 > weight_best:
        weight_best = weight_200
        im_best = im_vis_200
        coord_best = coord_200
        shape_id_best = shape_id_200
    if weight_auto > weight_best:
        weight_best = weight_auto
        im_best = im_vis_auto
        coord_best = coord_auto
        shape_id_best = shape_id_auto
    if ret_all:
        shape_ids = shape_id_default, shape_id_200, shape_id_auto
        coords = coord_default, coord_200, coord_auto, coord_best
        weights = weight_default, weight_200, weight_auto, weight_best
        im_list = im_vis_default, im_vis_200, im_vis_auto, im_best
        return shape_ids, coords, weights, im_list
    return shape_id_best, coord_best, weight_best, im_best
    pass
if __name__ == "__main__":
    # Standard points marking a reference area of interest
    std_pts = np.array([
        (828, 310), (885, 310), (945, 310),
        (826, 320), (886, 319), (946, 318),
        (826, 330), (886, 328), (948, 329)
    ])
    # Load image paths for processing
    # paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\2p\*.jpg')
    # paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\3p\*.jpg')
    # label_dir = r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict75\labels'

    std_pts = np.array([
        (610, 258),
        (687, 258),
        (606, 271),
        (694, 272),
    ])
    H, W = 720, 1280
    # paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict72\2p\*.jpg')#0908
    paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict72\3p\*.jpg')#0908
    # std_pts = np.array([
    #     (586, 160),
    #     (661, 161),
    #     (582, 174),
    #     (665, 175),
    # ])
    # paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict72\4p\*.jpg')#0908
    # std_pts = np.array([
    #     (572, 150),
    #     (648, 147),
    #     (568, 162),
    #     (660, 160),
    # ])
    # paths = glob.glob(r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict72\5p\*.jpg')#0908
    label_dir = r'G:\Windows\PycharmProjects1\stdlfire\runs\detect\predict72\labels'

    skip_idx = 3
    paths = select_evenly_distributed(paths, 120)
    for path_idx0, path in enumerate(paths[skip_idx:]):
        path_idx = skip_idx + path_idx0
        basename = os.path.basename(path)[:-4]
        label_path = os.path.join(label_dir, basename + '.txt')
        ori_labels = []
        im = cv2.imread(path)
        if im is None:
            print(f'im is None:{path}')
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '0':
                    line = line.rstrip('\n')
                    l = line.split(' ')[1:5]
                    xc, yc, wc, hc = np.float_(l)
                    x1, x2, y1, y2 = int(W * (xc - wc / 2)), int(W * (xc + wc / 2)), int(H * (yc - hc / 2)), int(
                        H * (yc + hc / 2))
                    ori_labels.append([x1, y1, x2, y2])

        conf, coord = fire_shape_analysis(im, ori_labels, std_pts)
        print(f'Frame[{path_idx}]: Fire at {coord} with confidence={conf}')
    cv2.destroyAllWindows()
