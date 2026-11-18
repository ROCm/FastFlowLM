// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file npu_utils.cpp
/// \brief NPU manager and utility implementation
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.1.0
#include "npu_utils/npu_utils.hpp"

/// \brief Constructor
/// \param device_id device id default to 0, almost always 0
npu_manager::npu_manager(npu_device device, unsigned int device_id){
    this->device = xrt::device(device_id);
    this->xclbin_descs.resize(max_xclbins);
    this->kernel_descs.resize(max_kernels);
    this->registered_xclbin_names.clear();
    this->kernel_desc_count = 0;
    this->xclbin_desc_count = 0;
    this->npu_gen = device;
}

///@brief register an accel_user_desc to the npu_manager
///@param user_desc accel_user_desc to be registered
///@return the app id of the registered accel_user_desc
///@note Different apps may have the same xclbin, but the sequence is unique.
///@note To avoid creating duplicated applications, the function checks if the xclbin is registered.
///@note If the xclbin is not registered, the function will register the xclbin and create a new application.
///@note If the xclbin is registered, the function will create a new application and load the instruction sequence.
npu_app npu_manager::create_app(npu_app_desc& desc){
    assert(desc.app_name != "");
    int xclbin_id = -1;
    for (size_t i = 0; i < this->registered_xclbin_names.size(); i++){
        if (this->registered_xclbin_names[i] == desc.xclbin_name){
            xclbin_id = i;
            break;
        }
    }
    LOG_VERBOSE_IF_ELSE(2, xclbin_id > -1, 
        "Found xclbin: " << desc.xclbin_name << "registered as id " << xclbin_id << "!",
        "Xclbin: " << desc.xclbin_name << " not registered yet!"
    );

    if (xclbin_id == -1){ // the xclbin is not registered yet
        if (this->xclbin_desc_count >= this->xclbin_descs.size()){
            throw std::runtime_error("Max number of xclbins reached");
        }
        if (_load_xclbin(desc.xclbin_name) != 0){
            std::cout<< "Load " << desc.xclbin_name << "ERROR!" << std::endl;
            exit(-1);
        }
        this->registered_xclbin_names.push_back(desc.xclbin_name);
        xclbin_id = this->registered_xclbin_names.size() - 1;
        LOG_VERBOSE(2, "Xclbin: " << desc.xclbin_name << " registered as id " << xclbin_id << "!");
        this->xclbin_desc_count++;
    }
    // register the instr
    int app_id = -1;
    for (size_t i = 0; i < this->kernel_descs.size(); i++){
        if (this->kernel_descs[i].app_name == desc.app_name){
            app_id = i;
            break;
        }
    }
    LOG_VERBOSE_IF_ELSE(2, app_id > -1, 
        "Found instruction: " << desc.app_name << "registered as id " << app_id << "!",
        "Instruction: " << desc.app_name << " not registered yet!"
    );
    if (app_id == -1){ // instr is not registered yet
        if (this->kernel_desc_count >= this->kernel_descs.size()){
            throw std::runtime_error("Max number of kernels reached");
        }
        this->kernel_descs[this->kernel_desc_count] = accel_kernel_desc{
            .app_name = desc.app_name,
            .xclbin_desc = std::make_unique<accel_xclbin_desc>(this->xclbin_descs[xclbin_id]),
            .instr_seq = npu_sequence(this->npu_gen, &this->device, &this->xclbin_descs[xclbin_id].kernel, desc.app_name)
        };
        LOG_VERBOSE(2, "Instruction: " << desc.app_name << " registered as id " << app_id << "!");
        this->kernel_desc_count++;
        app_id = this->kernel_desc_count - 1;
    }
    return npu_app(
        app_id, 
        &this->kernel_descs[app_id].instr_seq, 
        &this->kernel_descs[app_id].xclbin_desc->kernel,
        &this->device
    );
}

///@brief load the xclbin to the kernel descriptor
///@param xclbin_name name of the xclbin file
///@return 0 if successful
int npu_manager::_load_xclbin(std::string xclbin_name){
    LOG_VERBOSE(2, "Loading xclbin: " << xclbin_name);
    this->xclbin_descs[this->xclbin_desc_count].xclbin = xrt::xclbin(xclbin_name);
    // int verbosity = VERBOSE;
    std::string Node = "MLIR_AIE";
    auto xkernels = this->xclbin_descs[this->xclbin_desc_count].xclbin.get_kernels();
    auto xkernel = *std::find_if(
        xkernels.begin(), 
        xkernels.end(),
        [Node](xrt::xclbin::kernel &k) {
            auto name = k.get_name();
            return name.rfind(Node, 0) == 0;
        }
    );
    this->device.register_xclbin(this->xclbin_descs[this->xclbin_desc_count].xclbin);
    auto kernelName = xkernel.get_name();
    this->xclbin_descs[this->xclbin_desc_count].context = xrt::hw_context(this->device, this->xclbin_descs[this->xclbin_desc_count].xclbin.get_uuid());
    this->xclbin_descs[this->xclbin_desc_count].kernel = xrt::kernel(this->xclbin_descs[this->xclbin_desc_count].context, kernelName);
    LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " loaded successfully!");
    return 0;
}

///@brief create a xrt::runlist object
///@param app_id which application the runlist belongs to
///@return the runlist object
///@note The function will create a xrt::runlist object with the context of the application.
xrt::runlist npu_manager::create_runlist(npu_app& app){
    return xrt::runlist(this->kernel_descs[app.app_id].xclbin_desc->context);
}

// ///@brief destructor
// ///@note The function will destroy the kernel, buffer object and context.
// npu_manager::~npu_manager(){
//     // std::cout<<"clear bin!" << std::endl;
//     // this->kernel.~kernel();
//     // this->bo_instr.~bo();
//     // this->context.~hw_context();
// }

///@brief list the kernels and instruction sequences
void npu_manager::list_kernels(){
    std::cout << "Listing kernels: (Total: " << this->kernel_desc_count << ")" << std::endl;
    for (size_t i = 0; i < this->kernel_desc_count; i++){
        std::cout << "Instruction " << i << ": " << this->kernel_descs[i].app_name << std::endl;
    }
    std::cout << "Listing xclbins: (Total: " << this->xclbin_desc_count << ")" << std::endl;
    for (size_t i = 0; i < this->xclbin_desc_count; i++){
        std::cout << "Xclbin " << i << " at address: " <<  &this->xclbin_descs[i].xclbin << std::endl;
    }
}

///@brief write out the trace to a file
void npu_manager::write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  LOG_VERBOSE(1, "Writing out trace to: " << path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (size_t i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
  fout.close();
  LOG_VERBOSE(1, "Trace written successfully!");
}

#ifdef __LINUX__
///@brief print the npu information
///@note The function will print the npu version, clock frequency, column count, row count, core info, mem info, shim info.
///@note The information is read via the IOCTL interface.
void npu_manager::print_npu_info(){
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open amdgpu device");
        return;
    }
    amdxdna_drm_query_clock_metadata query_clock_metadata;
    amdxdna_drm_get_info get_info = {
        .param = DRM_AMDXDNA_QUERY_CLOCK_METADATA,
        .buffer_size = sizeof(amdxdna_drm_query_clock_metadata),
        .buffer = (unsigned long)&query_clock_metadata,
    };
    int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return;
    }

    amdxdna_drm_query_aie_metadata query_aie_metadata;
    get_info.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
    get_info.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
    get_info.buffer = (unsigned long)&query_aie_metadata;
    ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return;
    }

    close(fd);
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU version: " << query_aie_metadata.version.major << "." << query_aie_metadata.version.minor);
    MSG_BOX_LINE(40, "MP-NPU clock frequency: " << query_clock_metadata.mp_npu_clock.freq_mhz << " MHz");
    MSG_BOX_LINE(40, "H clock frequency: " << query_clock_metadata.h_clock.freq_mhz << " MHz");
    // What is the meaning of the column size?
    // std::cout << "NPU column size: " << query_aie_metadata.col_size << std::endl;
    MSG_BOX_LINE(40, "NPU column count: " << query_aie_metadata.cols);
    MSG_BOX_LINE(40, "NPU row count: " << query_aie_metadata.rows);
    MSG_BOX_LINE(40, "NPU core Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.core.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.core.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.core.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.core.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.core.event_reg_count);
    MSG_BOX_LINE(40, "NPU mem Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.mem.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.mem.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.mem.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.mem.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.mem.event_reg_count);
    MSG_BOX_LINE(40, "NPU shim Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.shim.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.shim.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.shim.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.shim.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.shim.event_reg_count);
    MSG_BONDLINE(40);
}

///@brief get the npu power consumption
///@param print whether to print the power consumption
///@return the power consumption, unit is Watt
float npu_manager::get_npu_power(bool print){
    // get the npu power consumption, unit is Watt
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open amdgpu device");
        return -1;
    }
    amdxdna_drm_query_sensor query_sensor;

    amdxdna_drm_get_info get_info = {
        .param = DRM_AMDXDNA_QUERY_SENSORS,
        .buffer_size = sizeof(amdxdna_drm_query_sensor),
        .buffer = (unsigned long)&query_sensor,
    };
    int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return -1;
    }
    if (print){
        MSG_BOX(40, "NPU power: " << query_sensor.input << " " << query_sensor.units);
    }
    close(fd);
    return (float)query_sensor.input * pow(10, query_sensor.unitm);
}
#endif