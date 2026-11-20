// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include "typedef.hpp"
#include "utils/debug_utils.hpp"
#include <chrono>


float get_relativeL2(buffer<dtype_out>& y, buffer<dtype_out>& y_ref){
    float rmse = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += y_ref[i] * y_ref[i];
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
}

float get_relativeL1(buffer<dtype_out>& y, buffer<dtype_out>& y_ref){
    float l1 = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += abs(y_ref[i]);
        l1 += abs(y[i] - y_ref[i]);
    }
    return l1 / ref_sum;
}

float get_rmse(buffer<dtype_out>& y, buffer<dtype_out>& y_ref){
    float rmse = 0;
    for (int i = 0; i < y.size(); i++){
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size());
}

float get_cosine_similarity(buffer<dtype_out>& y, buffer<dtype_out>& y_ref){
    float dot_product = 0;
    float norm_y = 0;
    float norm_y_ref = 0;
    for (int i = 0; i < y.size(); i++){
        dot_product += y[i] * y_ref[i];
        norm_y += y[i] * y[i];
        norm_y_ref += y_ref[i] * y_ref[i];
    }
    return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
}

namespace time_utils {

typedef std::chrono::high_resolution_clock::time_point time_point;
typedef std::pair<float, std::string> time_with_unit;

time_point now(){
    return std::chrono::high_resolution_clock::now();
}

time_with_unit duration_us(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count(), "us");
}

time_with_unit duration_ms(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(), "ms");
}

time_with_unit duration_s(time_point start, time_point stop){
    return std::make_pair(std::chrono::duration_cast<std::chrono::seconds>(stop - start).count(), "s");
}

time_with_unit cast_to_us(time_with_unit time){
    if (time.second == "us"){
        return time;
    }
    if (time.second == "ms"){
        return std::make_pair(time.first * 1000, "us");
    }
    if (time.second == "s"){
        return std::make_pair(time.first * 1000000, "us");
    }
    else{
        return time;
    }
}

time_with_unit cast_to_ms(time_with_unit time){
    if (time.second == "ms"){
        return time;
    }
    if (time.second == "us"){
        return std::make_pair(time.first / 1000, "ms");
    }
    if (time.second == "s"){
        return std::make_pair(time.first / 1000000, "ms");
    }
    else{
        return time;
    }
}   

time_with_unit cast_to_s(time_with_unit time){
    if (time.second == "s"){
        return time;
    }
    if (time.second == "us"){
        return std::make_pair(time.first / 1000000, "s");
    }
    if (time.second == "ms"){
        return std::make_pair(time.first / 1000, "s");
    }
    else{
        return time;
    }
}   

time_with_unit re_unit(time_with_unit time){
    time = cast_to_us(time);
    float time_us = time.first;
    std::string time_unit = time.second;
    if (time_us > 1000){
        time_us /= 1000;
        time_unit = "ms";
    }
    if (time_us > 1000000){
        time_us /= 1000000;
        time_unit = "s";
    }
    return std::make_pair(time_us, time_unit);
}


}

namespace utils {
float getRand(float min = 0, float max = 1){
    return (float)(rand()) / (float)(RAND_MAX) * (max - min) + min;
}

int getRandInt(int min = 0, int max = 6){
    return (int)(rand()) % (max - min) + min;
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
    os  << "\r" << std::string((int)(progress * len), '|')
        << std::string(len - (int)(progress * len), ' ') << std::setw(4)
        << std::fixed << std::setprecision(0) << progress * 100 << "%"
        << "\r";
}

void check_arg_file_exists(std::string name) {
    // Attempt to open the file
    std::ifstream file(name);

    // Check if the file was successfully opened
    if (!file) {
        throw std::runtime_error("Error: File '" + name + "' does not exist or cannot be opened.");
    }

    // Optionally, close the file (not strictly necessary, as ifstream will close on destruction)
    file.close();
}

///@brief compare two vectors, Templated for different types
///@param y the vector to compare
///@param y_ref the reference vector
///@param print_errors the number of errors to print, default is -1, which means print all errors
///@param abs_tol the absolute tolerance, default is 1e-1
///@note The function will compare the two vectors and print the errors
///@note The function will print the errors if the number of errors is less than print_errors
///@note The function will print the total errors
///@return the number of errors
template <typename T>
int compare_vectors(buffer<T>& y, buffer<T>& y_ref, int print_errors = -1, float abs_tol = 1e-1, float rel_tol = 1e-1){
    int total_errors = 0;
    assert(y.size() == y_ref.size()); // you should check the size of the two vectors before calling this function
    for (int i = 0; i < y.size(); i++){
        bool close_enough = true;
        if ((std::abs(y[i] - y_ref[i]) > abs_tol)){
            close_enough = false;
        }
        if (close_enough && y_ref[i] != 0 && (std::abs(y[i] - y_ref[i]) / std::abs(y_ref[i]) > rel_tol)){
            close_enough = false;
        }
        if (!close_enough){
            if (total_errors < print_errors){
                header_print("err ", "Error: y[" << i << "] = " << y[i] << " != y_ref[" << i << "] = " << y_ref[i]);
            }
            total_errors++;
        }
    }
    if (total_errors > 0){
        header_print("err ", "Total errors: " << total_errors);
    }
    return total_errors;
}

///@brief print the matrix, Templated for different types
///@param matrix the matrix to print
///@param n_cols the number of columns of the matrix
///@param n_printable_rows the number of rows to print, default is 10
///@param n_printable_cols the number of columns to print, default is 10
///@param ostream the output stream, default is std::cout
///@param col_sep the column separator, default is "  "
///@param elide_sym the elide symbol, default is " ... "
///@param w width of the number being printed, default is 6
///@note The function will print the matrix in a readable format
template <typename T>
void print_matrix(const buffer<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 20,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = 6) {
  assert(matrix.size() % n_cols == 0);

  int n_rows = matrix.size() / n_cols;
  DO_VERBOSE(1, {
    header_print("info", "Matrix size: " << "[" << n_rows << ", " << n_cols << "]");
  });

  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = n_printable_rows < n_rows;
  const bool elide_cols = n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(2);

#define print_row(what)                                                        \
  for (int col = 0; col < (n_printable_cols + 1) / 2; col++) {                 \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int i = 0; i < n_printable_cols / 2; i++) {                             \
    int col = n_cols - n_printable_cols / 2 + i;                               \
    ostream << std::right << std::setw(w) << std::scientific << std::setprecision(2) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int i = 0; i < n_printable_rows / 2; i++) {
    int row = n_rows - n_printable_rows / 2 + i;
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }

#undef print_row
}

///@brief print the npu profile
///@param npu_time the npu time
///@param op the number of operations
///@param n_iter the number of iterations
///@note The function will print the npu time, speed and the number of errors
void print_npu_profile(time_utils::time_with_unit npu_time, float op, int n_iter = 1){
    npu_time.first /= n_iter;
    time_utils::time_with_unit time_united = time_utils::re_unit(npu_time);
    time_united = time_utils::cast_to_s(time_united);


    float ops = op / (time_united.first);
    float speed = ops / 1000000;
    std::string ops_unit = "Mops";
    std::string speed_unit = "Mops/s";
    if (speed > 1000){
        speed /= 1000;
        speed_unit = "Gops/s";
    }
    if (speed > 1000000){
        speed /= 1000000;
        speed_unit = "Tops/s";
    }

    time_united = time_utils::re_unit(npu_time);
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time : " << time_united.first << " " << time_united.second);
    MSG_BOX_LINE(40, "NPU speed: " << speed << " " << speed_unit);
    MSG_BONDLINE(40);
}
}
#endif
