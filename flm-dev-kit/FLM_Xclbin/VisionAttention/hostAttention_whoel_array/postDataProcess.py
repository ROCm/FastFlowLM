# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.




# # 1. first load all column in resuots.csv
# # also return a list of all the headers
# def load_results(file_path):
#     results = []
#     headers = []
#     with open(file_path, 'r') as f:
#         for line_num, line in enumerate(f):
            
#             if line_num == 0:
#                 headers = line.strip().split(',')
#             else:
#                 values = line.strip().split(',')
                
#                 val_arr = []
#                 for v in values:
#                     try:
#                         val_arr.append(float(v))
#                     except ValueError:
#                         # skip non-numeric entries
#                         val_arr.append(v)
            
#                 results.append(val_arr)  # skip non-numeric entries
#     return results, headers


# # 2. then process the data into dict of {"string": list[values]}
# def process_data_to_dict(raw_data):
#     results, headers = raw_data
#     data_dict = {header: [] for header in headers}
    
#     for row in results:
#         for header, value in zip(headers, row):
#             data_dict[header].append(value)
    
#     return data_dict





# def visualize_data(data_dict:dict):
    
#     pass



# def re_group_data(data_dict:dict):
#     #NOTE: assume all data have same Devoce. NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN
    
#     # different DH,LQ_PER_CT,LK_PER_CT, LQ_internal, lk_lv_internal forms a group
#     # assume all keys have same length
#     data_length = len(next(iter(data_dict.values())))
    
    
#     regrouped_data_dict:dict[str, dict[str, dict]] = {}
    
#     for data_idx in range(data_length):
#         cur_data_r = int(data_dict["r"][data_idx])
#         cur_data_s = int(data_dict["s"][data_idx])
#         cur_data_t = int(data_dict["t"][data_idx])
#         cur_data_L_SEQ = int(data_dict["L_SEQ"][data_idx])
#         cur_data_S_SEQ = int(data_dict["S_SEQ"][data_idx])    
#         cur_data_D_H =      int(data_dict["D_H"][data_idx])
#         cur_data_NUM_HEAD = int(data_dict["num_head"][data_idx])
#         cur_data_LQ_PER_CT = int(data_dict["LQ_per_CT"][data_idx])
#         cur_data_LK_PER_CT = int(data_dict["LK_per_CT"][data_idx])
#         cur_data_LQ_internal = int(data_dict["lq_internal"][data_idx])
#         cur_data_lk_lv_internal = int(data_dict["lk_lv_internal"][data_idx])
        

#         main_header_keys=f"{cur_data_r}_{cur_data_s}_{cur_data_t}_{cur_data_L_SEQ}_{cur_data_S_SEQ}_{cur_data_D_H}_{cur_data_NUM_HEAD}_{cur_data_LQ_PER_CT}_{cur_data_LK_PER_CT}_{cur_data_LQ_internal}_{cur_data_lk_lv_internal}"

        
        
#         if main_header_keys not in regrouped_data_dict:
#             regrouped_data_dict[main_header_keys] = {}
            
#         # sub group as CU_MODE
#         sub_header_keys = f"{data_dict['CU_MODE'][data_idx]}"

#         if sub_header_keys not in regrouped_data_dict[main_header_keys]:
#             regrouped_data_dict[main_header_keys][sub_header_keys] = {}

        
#         # another sub group of REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH and RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION
#         sub_sub_header_keys = f"{data_dict['REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH'][data_idx]}_{data_dict['RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION'][data_idx]}"
#         if sub_sub_header_keys not in regrouped_data_dict[main_header_keys][sub_header_keys]:
#             regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys] = {}

#         if int(data_dict["LOW_PRECISION"][data_idx]) == 1: # mans use low precision
#             # store the average time
#             regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["Average_Time_us_low_precison"] = data_dict["Average_Time_us"][data_idx]
#         else:
#             regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["Average_Time_us_high_precison"] = data_dict["Average_Time_us"][data_idx]
#     return regrouped_data_dict
# __main__ = "__main__"
# if __name__ == __main__:
#     file_path = '/path/to/gpt-oss-mega-kernel/Template/hostAttention_whoel_array/data/DH_72_NUM_HEAD_16.csv'
#     raw_data = load_results(file_path)
#     data_dict = process_data_to_dict(raw_data)
#     # print out the keys
#     print("Keys in data_dict:")
    
#     re_group_dataed = re_group_data(data_dict=data_dict)
#     k = 1
import matplotlib.pyplot as plt
import numpy as np
import os

# --- BANDWIDTH CALCULATION ---
def calculate_input_memory_bandwidth(
    CU_MODE:int,
    L_SEQ:int,
    S_SEQ:int,
    D_H:int,
    NUM_HEAD:int,
    LQ_PER_CT:int,
    LK_PER_CT:int,
    LQ_internal:int,
    lk_lv_internal:int,
    average_time_us:float
):
    total_head_dim = D_H * NUM_HEAD
    # Check CU Mode mapping based on your hardware config
    if CU_MODE == 0:
        LQ_chunk_per_CU = 4 * LQ_PER_CT
        NUM_OF_CU = 8
    elif CU_MODE == 1:
        LQ_chunk_per_CU = 32 * LQ_PER_CT
        NUM_OF_CU = 1
    elif CU_MODE == 2:
        LQ_chunk_per_CU = 16 * LQ_PER_CT
        NUM_OF_CU = 2
    elif CU_MODE == 3:
        LQ_chunk_per_CU = 8 * LQ_PER_CT
        NUM_OF_CU = 4
    else:
        # Default fallback if unknown CU_MODE, to prevent crash
        LQ_chunk_per_CU = 4 * LQ_PER_CT
        NUM_OF_CU = 1 
        
    if LQ_PER_CT == LQ_internal:
        dataSize = total_head_dim * L_SEQ
        dataSize += NUM_OF_CU * (L_SEQ // LQ_chunk_per_CU) * (2 * total_head_dim * S_SEQ)
        byte_Size = dataSize * 2  # bf16 = 2 bytes
        memory_bandwidth = (byte_Size / 1e9) / (average_time_us / 1e6)
        return memory_bandwidth
    else:
        dataSize = total_head_dim * L_SEQ * (NUM_OF_CU * (S_SEQ // LK_PER_CT))
        dataSize += NUM_OF_CU * (L_SEQ // LQ_chunk_per_CU) * (2 * total_head_dim * S_SEQ)
        byte_Size = dataSize * 2  # bf16
        memory_bandwidth = (byte_Size / 1e9) / (average_time_us / 1e6)
        return memory_bandwidth

# 1. Load Data (Robust Parsing)
def load_results(file_path):
    results = []
    headers = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line: continue # Skip empty lines
                
                # Split and strip spaces from each element
                parts = [p.strip() for p in line.split(',')]
                
                if line_num == 0:
                    headers = parts
                else:
                    # Parse numeric values safely
                    val_arr = []
                    for v in parts:
                        try:
                            val_arr.append(float(v))
                        except ValueError:
                            val_arr.append(v)
                    results.append(val_arr)
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []
        
    return results, headers

# 2. Process to Dict
def process_data_to_dict(raw_data):
    results, headers = raw_data
    if not headers: return {}
    
    data_dict = {header: [] for header in headers}
    for row in results:
        for i, header in enumerate(headers):
            if i < len(row):
                data_dict[header].append(row[i])
            else:
                data_dict[header].append(None)
    return data_dict

# 3. Regroup Data
def re_group_data(data_dict: dict):
    if not data_dict or not data_dict.values():
        return {}
        
    data_length = len(next(iter(data_dict.values())))
    regrouped_data_dict = {}
    
    # Keys to construct the unique X-axis ID
    keys_to_extract = ["r", "s", "t", "L_SEQ", "S_SEQ", "D_H", "num_head", 
                       "LQ_per_CT", "LK_per_CT", "lq_internal", "lk_lv_internal"]
    display_names = ["r", "s", "t", "L", "S", "DH", "H", 
                     "LQ", "LK", "LQi", "LKi"]

    for data_idx in range(data_length):
        # Build Main Key (X-Axis ID)
        key_parts = []
        for raw_key, display_name in zip(keys_to_extract, display_names):
            val = data_dict.get(raw_key, [0]*data_length)[data_idx]
            clean_val = str(int(float(val))) if val is not None else "0"
            key_parts.append(f"{display_name}:{clean_val}")
            
        main_header_keys = " ".join(key_parts)
        
        # Meta Params for File Splitting
        current_lk_per_ct = int(float(data_dict.get("LK_per_CT", [0]*data_length)[data_idx]))
        current_l_seq = int(float(data_dict.get("L_SEQ", [0]*data_length)[data_idx]))
        current_s_seq = int(float(data_dict.get("S_SEQ", [0]*data_length)[data_idx]))
        
        if main_header_keys not in regrouped_data_dict:
            regrouped_data_dict[main_header_keys] = {
                "meta_LK_per_CT": current_lk_per_ct,
                "meta_L_SEQ": current_l_seq,
                "meta_S_SEQ": current_s_seq
            }
            
        # CU Mode Grouping
        raw_cu = data_dict.get('CU_MODE', [0]*data_length)[data_idx]
        sub_header_keys = str(int(float(raw_cu))) if raw_cu is not None else "0"
        
        if sub_header_keys not in regrouped_data_dict[main_header_keys]:
            regrouped_data_dict[main_header_keys][sub_header_keys] = {}

        # Reorder/Runtime Grouping
        raw_reorder = data_dict.get('REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH', [0]*data_length)[data_idx]
        raw_runtime = data_dict.get('RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION', [0]*data_length)[data_idx]
        reorder = int(float(raw_reorder)) if raw_reorder is not None else 0
        runtime = int(float(raw_runtime)) if raw_runtime is not None else 0
        
        # Shortened key for legend readability
        sub_sub_header_keys = f"Reorder:{reorder} | Runtime:{runtime}"
        
        if sub_sub_header_keys not in regrouped_data_dict[main_header_keys][sub_header_keys]:
            regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys] = {}

        # Values
        raw_time = data_dict.get("Average_Time_us", [0]*data_length)[data_idx]
        time_ms = float(raw_time) / 1000.0 if raw_time is not None else 0.0
        
        raw_prec = data_dict.get("LOW_PRECISION", [0]*data_length)[data_idx]
        is_low_prec = int(float(raw_prec)) == 1 if raw_prec is not None else False

        # Bandwidth Calc
        bw_val = calculate_input_memory_bandwidth(
            CU_MODE= int(float(raw_cu)),
            L_SEQ= current_l_seq,
            S_SEQ= current_s_seq,
            D_H= int(float(data_dict.get('D_H', [0]*data_length)[data_idx])),
            NUM_HEAD= int(float(data_dict.get('num_head', [0]*data_length)[data_idx])),
            LQ_PER_CT= int(float(data_dict.get('LQ_per_CT', [0]*data_length)[data_idx])),
            LK_PER_CT= current_lk_per_ct,
            LQ_internal= int(float(data_dict.get('lq_internal', [0]*data_length)[data_idx])),
            lk_lv_internal= int(float(data_dict.get('lk_lv_internal', [0]*data_length)[data_idx])),
            average_time_us= float(raw_time) if raw_time is not None else 1.0
        )

        if is_low_prec:
            regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["Low_Prec"] = time_ms
            regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["Low_Prec_BW"] = bw_val
        else:
            regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["High_Prec"] = time_ms
            regrouped_data_dict[main_header_keys][sub_header_keys][sub_sub_header_keys]["High_Prec_BW"] = bw_val
            
    return regrouped_data_dict

# 4. Plot Single Group
def plot_and_save_group(plot_items, lk_value, l_seq, s_seq, save_dir="."):
    unique_x_keys = sorted(list(set(d['x'] for d in plot_items)))
    x_map = {key: i for i, key in enumerate(unique_x_keys)}
    
    # Create Table Data
    table_data = [[i, key] for i, key in enumerate(unique_x_keys)]

    unique_cu_modes = sorted(list(set(d['cu_mode'] for d in plot_items)))
    unique_configs = sorted(list(set(d['config'] for d in plot_items)))
    
    num_rows = len(unique_cu_modes)
    if num_rows == 0: return

    # --- DYNAMIC SIZE CALCULATION ---
    # Calculate required height based on number of table rows (x_keys)
    # Each table row needs ~0.5 inches roughly
    table_height_needed = max(4, len(unique_x_keys) * 0.4) 
    chart_height_needed = 5 * num_rows
    total_height = max(chart_height_needed, table_height_needed)
    
    # Increase width to give bandwidth text breathing room
    fig = plt.figure(figsize=(26, total_height))
    
    # Adjust width ratios to give table enough space
    gs = fig.add_gridspec(num_rows, 2, width_ratios=[0.65, 0.35], wspace=0.15)

    num_configs = len(unique_configs)
    config_offsets = {}
    # Slightly wider offsets to prevent text overlap
    if num_configs > 1:
        offsets = np.linspace(-0.25, 0.25, num_configs)
        for i, conf in enumerate(unique_configs):
            config_offsets[conf] = offsets[i]
    else:
        config_offsets = {unique_configs[0]: 0}

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_configs)))
    color_map = {key: colors[i] for i, key in enumerate(unique_configs)}

    # Plot Charts (Left Column)
    for row_idx, current_cu_mode in enumerate(unique_cu_modes):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.set_title(f"CU_MODE: {current_cu_mode} (L:{l_seq}, S:{s_seq}, LK:{lk_value})", fontsize=16, pad=20, loc='left')
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel("Time (ms)", fontsize=12)
        
        subset = [d for d in plot_items if d['cu_mode'] == current_cu_mode]
        
        for item in subset:
            base_x = x_map[item['x']]
            offset = config_offsets[item['config']]
            final_x = base_x + offset
            color = color_map[item['config']]
            
            y_high = item['high']
            y_low = item['low']
            bw_high = item.get('high_bw')
            bw_low = item.get('low_bw')
            
            # Draw Line (Range)
            if y_high is not None and y_low is not None:
                ax.plot([final_x, final_x], [y_low, y_high], color=color, linestyle='-', linewidth=2, alpha=0.8)
                
            # Plot High Prec
            if y_high is not None:
                ax.plot(final_x, y_high, marker='_', markersize=14, markeredgewidth=3, color=color)
                ax.plot(final_x, y_high, marker='o', markersize=5, color=color)
                if bw_high is not None:
                    # Text ABOVE
                    label = f"{y_high:.2f}ms | {bw_high:.0f}GB/s"
                    ax.text(final_x, y_high, label, fontsize=9, ha='center', va='bottom', 
                            rotation=90, color='black', weight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

            # Plot Low Prec
            if y_low is not None:
                ax.plot(final_x, y_low, marker='_', markersize=14, markeredgewidth=3, color=color)
                ax.plot(final_x, y_low, marker='o', markersize=5, color=color)
                if bw_low is not None:
                    # Text BELOW
                    label = f"{y_low:.2f}ms | {bw_low:.0f}GB/s"
                    ax.text(final_x, y_low, label, fontsize=9, ha='center', va='top', 
                            rotation=90, color='black', weight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Increase margins to ensure vertical text fits
        ax.margins(y=0.3)
        
        ax.set_xticks(range(len(unique_x_keys)))
        if row_idx == num_rows - 1:
            ax.set_xticklabels(range(len(unique_x_keys)), fontsize=12, fontweight='bold')
            ax.set_xlabel("Configuration ID (See Table)", fontsize=14, labelpad=15)
        else:
            ax.set_xticklabels([])

    # Table & Legend (Right Column) - Spans all rows
    ax_right = fig.add_subplot(gs[:, 1])
    ax_right.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='w', label='-- Configurations --')]
    for config_name, color in color_map.items():
        legend_elements.append(Line2D([0], [0], color=color, label=config_name, linewidth=2, marker='_'))
        
    legend_elements.extend([
        Line2D([0], [0], color='w', label=' '),
        Line2D([0], [0], color='black', label='Text: Time | Bandwidth', marker=None, linestyle='None'),
        Line2D([0], [0], color='black', label='Top Bar: High Prec', marker='_', markersize=10, linestyle='None'),
        Line2D([0], [0], color='black', label='Bot Bar: Low Prec', marker='_', markersize=10, linestyle='None')
    ])
    
    # Place Legend at Top
    ax_right.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.0), frameon=True, fontsize=10)

    # Table - Placed below legend
    # bbox=[left, bottom, width, height] relative to ax_right
    the_table = ax_right.table(cellText=table_data, 
                               colLabels=["ID", "Full Configuration Key"], 
                               loc='center', 
                               colWidths=[0.1, 0.9], 
                               cellLoc='left',
                               bbox=[0, 0, 1, 0.85]) # Take up bottom 85% of right panel
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9) 
    
    plt.tight_layout()

    # --- SAVE FILES ---
    base_filename = f"plot_L{l_seq}_S{s_seq}_LK{lk_value}"
    
    # Standard Res
    save_path_reg = os.path.join(save_dir, f"{base_filename}_regular.png")
    plt.savefig(save_path_reg, dpi=100)
    print(f"Saved: {save_path_reg}")

    # 4K Res
    save_path_4k = os.path.join(save_dir, f"{base_filename}_4k.png")
    plt.savefig(save_path_4k, dpi=300)
    print(f"Saved: {save_path_4k}")

    plt.close(fig)

# 5. Main Viz Driver
def visualize_data(regrouped_data: dict):
    all_items = []
    
    # Flatten Data
    for main_key, sub_dict in regrouped_data.items():
        lk_val = sub_dict.get("meta_LK_per_CT", 0)
        l_seq = sub_dict.get("meta_L_SEQ", 0)
        s_seq = sub_dict.get("meta_S_SEQ", 0)
        
        for key, val in sub_dict.items():
            if key.startswith("meta_"): continue
            cu_mode = key
            
            for config_key, times in val.items():
                high_val = times.get("High_Prec", None)
                low_val = times.get("Low_Prec", None)
                high_bw = times.get("High_Prec_BW", None)
                low_bw = times.get("Low_Prec_BW", None)
                
                if high_val is not None or low_val is not None:
                    all_items.append({
                        "lk_val": lk_val,
                        "l_seq": l_seq,
                        "s_seq": s_seq,
                        "x": main_key,
                        "cu_mode": cu_mode,
                        "config": config_key,
                        "high": high_val,
                        "low": low_val,
                        "high_bw": high_bw,
                        "low_bw": low_bw
                    })

    if not all_items:
        print("No data found to plot. Check CSV format or path.")
        return

    # Group by (L, S, LK)
    unique_groups = sorted(list(set(
        (d['l_seq'], d['s_seq'], d['lk_val']) for d in all_items
    )))
    
    print(f"Found {len(unique_groups)} groups to plot:")
    for g in unique_groups:
        print(f"  - L:{g[0]}, S:{g[1]}, LK:{g[2]}")
    
    for (l_seq, s_seq, lk_val) in unique_groups:
        group_items = [
            d for d in all_items 
            if d['l_seq'] == l_seq and d['s_seq'] == s_seq and d['lk_val'] == lk_val
        ]
        plot_and_save_group(group_items, lk_val, l_seq, s_seq)

if __name__ == "__main__":
    # Update this path to your actual CSV file location
    file_path = './data/DH_64_NUM_HEAD_16.csv'
    
    if os.path.exists(file_path):
        raw_data = load_results(file_path)
        if raw_data[0]: # Check if data was loaded
            data_dict = process_data_to_dict(raw_data)
            re_group_dataed = re_group_data(data_dict=data_dict)
            visualize_data(re_group_dataed)
        else:
            print("Failed to load data from CSV.")
    else:
        print(f"File not found: {file_path}")