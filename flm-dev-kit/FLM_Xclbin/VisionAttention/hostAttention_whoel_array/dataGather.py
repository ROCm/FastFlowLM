# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import subprocess
import os
from pathlib import Path
from time import sleep



def run_npu_attention_kernel(
    DEVICE:str,
    r:int, s:int, t:int, DH:int,
    LQ_PER_CT:int, LK_PER_CT:int,
    LQ_internal:int, lk_lv_internal:int,
    NUM_HEAD:int,
    SHIM_L_SEQ:int, SHIM_S_SEQ:int,
    NUM_OF_COLUMNS:int, NUM_OF_CT_PER_COLUMN:int,
    CU_MODE:int,
    LOW_PRECISION:int,
    RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION:int
):
    """
    Run the NPU attention kernel with specified parameters.
    Returns True if successful, False otherwise.
    """
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Create data directory if it doesn't exist
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create output filename based on parameters
    output_filename = (f"output_r{r}_s{s}_t{t}_DH{DH}_"
                      f"LQ{LQ_PER_CT}_LK{LK_PER_CT}_"
                      f"NH{NUM_HEAD}_COL{NUM_OF_COLUMNS}_CT{NUM_OF_CT_PER_COLUMN}_"
                      f"CU{CU_MODE}_LP{LOW_PRECISION}_RSLD{RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION}.txt")
    output_path = data_dir / output_filename
    
    # Build make parameters
    make_params = [
        f"DEVICE={DEVICE}",
        f"r={r}",
        f"s={s}",
        f"t={t}",
        f"DH={DH}",
        f"LQ_PER_CT={LQ_PER_CT}",
        f"LK_PER_CT={LK_PER_CT}",
        f"LQ_internal={LQ_internal}",
        f"lk_lv_internal={lk_lv_internal}",
        f"NUM_HEAD={NUM_HEAD}",
        f"SHIM_L_SEQ={SHIM_L_SEQ}",
        f"SHIM_S_SEQ={SHIM_S_SEQ}",
        f"NUM_OF_COLUMNS={NUM_OF_COLUMNS}",
        f"NUM_OF_CT_PER_COLUMN={NUM_OF_CT_PER_COLUMN}",
        f"CU_MODE={CU_MODE}",
        f"LOW_PRECISION={LOW_PRECISION}",
        f"RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION={RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION}"
    ]
    
    try:
        print(f"\n{'='*80}")
        print(f"Running configuration: r={r}, s={s}, t={t}, DH={DH}, CU_MODE={CU_MODE}, LOW_PRECISION={LOW_PRECISION}, RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION={RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION}")
        print(f"{'='*80}\n")
        
        # Step 1: make clean
        print("Step 1: Cleaning...")
        result = subprocess.run(
            ["make", "clean"],
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR during make clean: {result.stderr}")
            return False
        print("Clean completed successfully")
        
        # Step 2: make bitstream
        print("\nStep 2: Building bitstream...")
        result = subprocess.run(
            ["make", "bitstream"] + make_params,
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR during make bitstream: {result.stderr}")
            return False
        print("Bitstream built successfully")
        
        sleep(2)  # brief pause to ensure file system stability
        
        # Step 3: make instructions
        print("\nStep 3: Building instructions...")
        result = subprocess.run(
            ["make", "instructions"] + make_params,
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR during make instructions: {result.stderr}")
            return False
        print("Instructions built successfully")
        sleep(2)  # brief pause to ensure file system stability
        # Step 4: make run and capture output
        print(f"\nStep 4: Running and saving output to {output_filename}...")
        result = subprocess.run(
            ["make", "run"] + make_params,
            cwd=script_dir,
            capture_output=True,
            text=True
        )
        
        # Save output to file
        with open(output_path, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"WARNING: make run returned non-zero exit code: {result.returncode}")
            print(f"Output saved to {output_filename}")
            return False
        
        print(f"Run completed successfully. Output saved to {output_filename}")
        return True
        
    except Exception as e:
        print(f"ERROR: Exception occurred: {str(e)}")
        return False


def run_parameter_sweep(
    r:int, s:int, t:int, DH:int, NUM_HEAD:int, SHIM_L_SEQ:int, SHIM_S_SEQ:int,
    configs: list[tuple[int, int, int, int]],  # LQ_PER_CT, LK_PER_CT, LQ_internal, lk_lv_internal,
    CU_modes:list[int]
):
    """
    Example function to run multiple configurations.
    Customize the parameter ranges as needed.
    """


    

    # Define parameter combinations to test

    
    # let us fix r, s, t, DH, NUM_HEAD, SHIM_L_SEQ, SHIM_S_SEQ, NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN


    NUM_OF_COLUMNS = 8
    NUM_OF_CT_PER_COLUMN = 4
    
    results = []    
    for config in configs:
        
        for RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION in [0,1]:
            for LOW_PRECISION in range(2):

                for CU_MODE in CU_modes:
            
                    LQ_PER_CT, LK_PER_CT, LQ_internal, lk_lv_internal = config
                    
                    success = run_npu_attention_kernel(
                        DEVICE="npu2",
                        r=r, s=s, t=t, DH=DH,
                        LQ_PER_CT=LQ_PER_CT, LK_PER_CT=LK_PER_CT,
                        LQ_internal=LQ_internal, lk_lv_internal=lk_lv_internal,
                        NUM_HEAD=NUM_HEAD,
                        SHIM_L_SEQ=SHIM_L_SEQ, SHIM_S_SEQ=SHIM_S_SEQ,
                        NUM_OF_COLUMNS=NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN=NUM_OF_CT_PER_COLUMN,
                        CU_MODE=CU_MODE,
                        LOW_PRECISION=LOW_PRECISION,
                        RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION=RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION
                    )
                
        results.append((config, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for config, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: LQ_PER_CT={config[0]}, LK_PER_CT={config[1]}, LQ_internal={config[2]}, lk_lv_internal={config[3]}")
    

if __name__ == "__main__":
    # Run the parameter sweep
    
        
    # Delete "results.txt" if exists and create a new one with CSV headers
    script_dir = Path(__file__).parent.resolve()

    # remove ./data directory if exists
    data_dir = script_dir / "data"
    if data_dir.exists() and data_dir.is_dir():
        for file in data_dir.iterdir():
            file.unlink()
        data_dir.rmdir()
    # now, recreate data directory
    data_dir.mkdir()
    
    # configs_for_gemma = [
    #     # FORMAT: LQ_PER_CT, LK_PER_CT, LQ_internal, lk_lv_internal
    #     # (32, 256, 16, 16),
    #     # (32, 256, 32, 16),
    #     # (32, 256, 32, 32),
    #     (32, 512, 16, 16),
    #     (32, 512, 32, 16),
    #     (32, 512, 32, 32),
    #     #(64, 256, 32, 16),
    #     #(64, 256, 32, 32),
    #     (64, 512, 32, 16),
    #     #(128, 256, 16, 16),
    #     (128, 512, 16, 16)
    # ]
    r = 8
    s= 8
    t= 8
    # CU_modes = [0,1,2,3]    
    # # run the gemma
    # run_parameter_sweep(
    #     r=r, s=s, t=t, DH=72, NUM_HEAD=16, SHIM_L_SEQ=4096, SHIM_S_SEQ=4096,
    #     configs=configs_for_gemma,
    #     CU_modes=CU_modes
    # )
    
    
    
    # configs_for_qwen3vl = [
    #     (32,512,16,16),
    #     (32,512,32,16),
    #     (32,512,32,32),
    #     (64,512,32,16),
    #     (128,512,16,16)
    # ]
    # run_parameter_sweep(
    #     r=r, s=s, t=t, DH=64, NUM_HEAD=16, SHIM_L_SEQ=4096, SHIM_S_SEQ=4096,
    #     configs=configs_for_qwen3vl,
    #     CU_modes=CU_modes
    # )
    
    
    
    
    # for running with CU_mode of 1, 2 only
    CU_modes_reduced = [1,2]
    configs_for_qwen3vl_small_test = [
        (32,512,32,32),
        (64,512,32,16),
    ]    
    
    
    # run different size
    # l = 1024
        
    for l in [1024, 2048, 4096, 8192, 16384]:
        run_parameter_sweep(
            r=r, s=s, t=t, DH=64, NUM_HEAD=16, SHIM_L_SEQ=l, SHIM_S_SEQ=l,
            configs=configs_for_qwen3vl_small_test,
            CU_modes=CU_modes_reduced
        )
    