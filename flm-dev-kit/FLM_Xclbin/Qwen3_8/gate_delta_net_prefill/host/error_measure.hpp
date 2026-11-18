// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __error_measure_cpu__
#define __error_measure_cpu__

#include <type_traits> // Required for std::is_same_v
#include <stdfloat>
#include <algorithm> // For std::min
template<typename Ta,typename Tb >
float get_relativeL2(Ta* y, Tb* y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded){
    static_assert(std::is_same_v<Ta, float> || std::is_same_v<Ta, std::bfloat16_t>, 
                    "Error: T must be either float or std::bfloat16_t.");
    float rmse = 0;
    float ref_sum = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                ref_sum += (float)y_ref[y_ref_idx] * (float)y_ref[y_ref_idx];
                rmse += ((float)y[y_idx] - (float)y_ref[y_ref_idx]) * ((float)y[y_idx] - (float)y_ref[y_ref_idx]);
            }
        }
    }
    int y_size = batch_size * seq_len * hidden_size;
    if (ref_sum == 0) {
        return (rmse == 0) ? 0.0f : std::numeric_limits<float>::infinity();
    }
    return sqrt(rmse / y_size) / sqrt(ref_sum / y_size);
}

// y is [batch_size, seq_len_padded, hidden_size_padded]
// y_ref is [batch_size, seq_len, hidden_size]
template<typename Ta,typename Tb >
float get_relativeL1(Ta* y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float l1 = 0;
    float ref_sum = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                ref_sum += abs((float)y_ref[y_ref_idx]);
                l1 += abs((float)y[y_idx] - (float)y_ref[y_ref_idx]);
            }
        }
    }
    if (ref_sum == 0) {
        return (l1 == 0) ? 0.0f : std::numeric_limits<float>::infinity();
    }
    return l1 / ref_sum;
}


template<typename Ta,typename Tb >
float get_rmse(Ta *y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float rmse = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                rmse += ((float)y[y_idx] - (float)y_ref[y_ref_idx]) * ((float)y[y_idx] - (float)y_ref[y_ref_idx]);
            }
        }
    }
    int y_size = batch_size * seq_len * hidden_size;
    return sqrt(rmse / y_size);
}

template<typename T>
float cal_rms_value(T* a, int a_size) {
    float sum_sq = 0.0f;
    for (int i = 0; i < a_size; ++i) {
        sum_sq += static_cast<float>(a[i]) * static_cast<float>(a[i]);
    }
    return std::sqrt(sum_sq / a_size);
}
template<typename Ta,typename Tb >
float get_cosine_similarity(Ta *y, Tb *y_ref, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float dot_product = 0;
    float norm_y = 0;
    float norm_y_ref = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)b * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)b * seq_len * hidden_size + (long long)s * hidden_size + h;
                dot_product += (float)y[y_idx] * (float)y_ref[y_ref_idx];
                norm_y += (float)y[y_idx] * (float)y[y_idx];
                norm_y_ref += (float)y_ref[y_ref_idx] * (float)y_ref[y_ref_idx];
            }
        }
    }
    if (norm_y_ref == 0) {
        return (norm_y == 0) ? 1.0f : 0.0f;
    }
    if (norm_y == 0) {
        return 0.0f;
    }
    return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
}

template<typename Ta,typename Tb >
int max_absolute_different_index(Ta *a, Tb *b, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded)
{
    float max_diff = 0.0f;
    int max_diff_index = -1;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < hidden_size; ++h) {
                long long y_idx = (long long)batch_idx * seq_len_padded * hidden_size_padded + (long long)s * hidden_size_padded + h;
                long long y_ref_idx = (long long)batch_idx * seq_len * hidden_size + (long long)s * hidden_size + h;
                float diff = abs((float)a[y_idx] - (float)b[y_ref_idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_diff_index = y_ref_idx;
                }
            }
        }
    }
    return max_diff_index;
}

template<typename T_a, typename T_b>
void print_error_metrics(T_a*a, T_b*b, int batch_size, int seq_len, int hidden_size, int seq_len_padded, int hidden_size_padded){

  
    float rmse = get_rmse(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float relativeL1 = get_relativeL1(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float relativeL2 = get_relativeL2(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float cosine_similarity = get_cosine_similarity(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);

    // find the largest absoulte valie between a, b, and the index, print it out
    int max_diff_index = max_absolute_different_index(a, b, batch_size, seq_len, hidden_size, seq_len_padded, hidden_size_padded);
    float max_abs_difference = abs((float)a[max_diff_index] - (float)b[max_diff_index]);
    header_print("info", "Max absolute difference: " << max_abs_difference << " at index " << max_diff_index);


    header_print("info", "Relative L1: " << relativeL1);
    header_print("info", "Relative L2: " << relativeL2);
    header_print("info", "Cosine similarity: " << cosine_similarity);
    header_print("info", "RMSE: " << rmse);

}
#endif