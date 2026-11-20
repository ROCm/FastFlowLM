// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __PACKET_CONTROL_GEN_H__
#define __PACKET_CONTROL_GEN_H__
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>




// Use a constexpr function for compile-time computation
constexpr uint32_t get_CT_bd_x_0_address(int32_t bd_id) {
    return 0x000001D000u + static_cast<uint32_t>(bd_id) * 0x20u;
}


constexpr uint32_t get_SHM_bd_x_0_address(int32_t bd_id){

    return 0x000001D000 + bd_id*0x20; 

}

constexpr uint32_t tm_start_addr = 0x80000;
static volatile uint32_t chess_storage(TM : tm_start_addr) addr_space_start;

void read_processor_bus(uint32_t *data, uint32_t addr, uint32_t size,
                        uint32_t stride=16) {
  for (uint32_t i = 0; i < size; i++) {
    
    #if defined(__chess__)
        uint32_t offset = addr + (i * stride);
        data[i] = *(&addr_space_start + (offset / 4));
    #elif defined(__AIECC__)
        uint32_t offset =addr  +  (i*(stride/4));    
        data[i] = read_tm(offset );
    #else
        static_assert(false, "Unexpected case here");   
    #endif
  }
}

void write_process_bus(uint32_t *data, uint32_t addr, uint32_t size, uint32_t stride=16){
  
    for (uint32_t i = 0; i < size; i++) {

        #if defined(__chess__)
            uint32_t offset = addr + (i * stride);
            *(&addr_space_start + (offset / 4)) =  data[i];
        #elif defined(__AIECC__)
            uint32_t offset =addr  +  (i*(stride/4));
            write_tm( data[i],  offset );
        #else
            static_assert(false, "Unexpected case here");   
        #endif    
    }
}

inline volatile void compiler_sync_barrier(){
    #if defined(__chess__)
        chess_separator_scheduler(4);
    #elif defined(__AIECC__)
          __builtin_aie2p_sched_barrier();    
        volatile int32_t k = 0;    
        k++;
        __builtin_aie2p_sched_barrier();
        k++;
        __builtin_aie2p_sched_barrier();     

    #else
        static_assert(false, "Unexpected case here");   
    #endif
}



bool even_parity(uint32_t n) {
    // Return true of even number of "1" bits, false otherwise
    uint32_t p = 0;
    while (n) {
        p += n & 1;
        n >>= 1;
    }
    return (p % 2) == 0;
}
uint32_t control_packet_gen(int32_t stream_id, int32_t operation, int32_t beats, int32_t address){
  //operation: 1 read, 0 write
  uint32_t control_packet =
        stream_id << 24 | operation << 22 | beats << 20 | address;
  control_packet |= (0x1 & even_parity(control_packet)) << 31;

  return control_packet;
}



inline uint32_t source_core_id_gen(uint32_t source_col, uint32_t source_row){
  return   (source_col <<5)|source_row;
}

// Generate packet flow header
uint32_t packet_flow_gen(uint32_t packet_type, uint32_t stream_id, uint32_t source_core_id =get_coreid()){

    //TODO: put a check on the packet_type? core=0, memtile=1, shimtile=2???
    uint32_t packet_header = (source_core_id << 16) | (packet_type<<12) | (stream_id);

    bool odd_parity  = (even_parity(packet_header)) ;
    packet_header |= (0x1 & odd_parity) << 31;

    return packet_header;
}





// Code for changing CT's register/behavior at runtime


// change the buffer len this bd receive/send at each transaction
inline void configure_BD_x_MM2S_y_dma_bd_len(
    
    const uint32_t CT_BD_id, // The CT dma's bd_id for configuration 
    
    const uint32_t MM2S_port_num,  uint32_t len, uint32_t bd_repeat_len=0 ){
    uint32_t len_mask = 0;
    uint32_t write_buffer[1];


    uint32_t BD_x_0_addr = get_CT_bd_x_0_address(CT_BD_id);

    uint32_t MM2S_CTRL = (MM2S_port_num==0)? 0x000001DE10 : 0x000001DE18;
    uint32_t MM2S_START_QUEUE  = (MM2S_port_num == 0) ? 0x000001DE14 : 0x000001DE1C;

    read_processor_bus( write_buffer, BD_x_0_addr, 1, 16 );// first read
    compiler_sync_barrier();  

    len_mask = 0x3FFF;
    *write_buffer =  ( (*write_buffer) & ~len_mask) | ( (len) & len_mask);         
    write_process_bus( write_buffer, BD_x_0_addr, 1, 16 );
    compiler_sync_barrier();  

    *(write_buffer) =(1<<1);// disable MM2S-1
    write_process_bus( (uint32_t*)(write_buffer),MM2S_CTRL, 1, 16 );  
    compiler_sync_barrier();  

    *(write_buffer) =(0<1);  // renable MM2S-1
    write_process_bus( (uint32_t*)(write_buffer),MM2S_CTRL, 1, 16  ); 
    compiler_sync_barrier();  
       
    *(write_buffer) =(bd_repeat_len<<16) | (CT_BD_id);
    write_process_bus( (uint32_t*)(write_buffer),MM2S_START_QUEUE, 1, 16  ); 
    compiler_sync_barrier();           




}






// CODE For control shimtile transfer by changing shimtile  BD_x_0 address and the corresponding queue
//THIS version assume the packet_id is already setup in CT's MM2S port

void control_Shimtile_transfer(
    uint32_t virtual_address_base,
    uint32_t length, uint32_t address_offset,
    uint32_t control_packet_out_prod_lock, uint32_t control_packet_out_con_lock,
    uint32_t * control_packet_out_buf,


    uint32_t CT_to_shimtile_control_bd_id,
    uint32_t CT_to_shimtile_control_bd_port,
    uint32_t SHIM_control_packet_id,

    uint32_t  SHM_MM2S_channel_idx,  uint32_t Shimtile_bd_id
){

    uint32_t Shimtile_BD_X_0_addr = get_SHM_bd_x_0_address(Shimtile_bd_id);

    configure_BD_x_MM2S_y_dma_bd_len(CT_to_shimtile_control_bd_id,CT_to_shimtile_control_bd_port, 3,0);//Configure MM2s-0 to send data
    // Write BD_00 and BD_01
    acquire_greater_equal(control_packet_out_prod_lock,1);
    *control_packet_out_buf = control_packet_gen(SHIM_control_packet_id , 0,1, Shimtile_BD_X_0_addr);
    *(control_packet_out_buf+1) = length; //bd_x_0
    *(control_packet_out_buf+2)  = (virtual_address_base + address_offset); ///bd_x_1         
    release(control_packet_out_con_lock, 1);


    uint32_t Shimtile_MM2S_X_TASK_QUEUE_addr = SHM_MM2S_channel_idx==0 ?0x000001D214 : 0x000001D21C;

    configure_BD_x_MM2S_y_dma_bd_len(CT_to_shimtile_control_bd_id,CT_to_shimtile_control_bd_port, 2,0);//Configure MM2s-0 to send data
    // Send write message
    acquire_greater_equal(control_packet_out_prod_lock,1);
    *control_packet_out_buf = control_packet_gen( SHIM_control_packet_id, 0,0, Shimtile_MM2S_X_TASK_QUEUE_addr);
    
    // assume not enable token issue
    // repeat_count = 1-0
    *(control_packet_out_buf+1) =  0<<16 | Shimtile_bd_id;
    release(control_packet_out_con_lock, 1);

}





// This is the version where the packet_flow is not being setup in MM2S port of CT
template<bool issue_token=false>
void control_Shimtile_transfer_with_packetflow_gen(
    uint32_t virtual_address_base,
    uint32_t length, uint32_t address_offset,
    uint32_t control_packet_out_prod_lock, uint32_t control_packet_out_con_lock,
    uint32_t * control_packet_out_buf,


    uint32_t CT_to_shimtile_control_bd_id,
    uint32_t CT_to_shimtile_control_bd_port,
    uint32_t SHIM_control_packet_id,

    uint32_t  SHM_MM2S_channel_idx,  uint32_t Shimtile_bd_id
){

    uint32_t Shimtile_BD_X_0_addr = get_SHM_bd_x_0_address(Shimtile_bd_id);

    configure_BD_x_MM2S_y_dma_bd_len(CT_to_shimtile_control_bd_id,CT_to_shimtile_control_bd_port, 4,0);//Configure MM2s-0 to send data
    // Write BD_00 and BD_01
    acquire_greater_equal(control_packet_out_prod_lock,1);
    *control_packet_out_buf = packet_flow_gen(0, // 0 since is from CT?
    SHIM_control_packet_id
    );
    *(control_packet_out_buf+1) = control_packet_gen(SHIM_control_packet_id , 0,1, Shimtile_BD_X_0_addr);
    *(control_packet_out_buf+2) = length; //bd_x_0
    *(control_packet_out_buf+3)  = (virtual_address_base + address_offset); ///bd_x_1         
    release(control_packet_out_con_lock, 1);


    uint32_t Shimtile_MM2S_X_TASK_QUEUE_addr = SHM_MM2S_channel_idx==0 ?0x000001D214 : 0x000001D21C;

    configure_BD_x_MM2S_y_dma_bd_len(CT_to_shimtile_control_bd_id,CT_to_shimtile_control_bd_port, 3,0);//Configure MM2s-0 to send data
    // Send write message
    acquire_greater_equal(control_packet_out_prod_lock,1);

    *control_packet_out_buf = packet_flow_gen(0, // 0 since is from CT?
    SHIM_control_packet_id);

    *(control_packet_out_buf+1) = control_packet_gen( SHIM_control_packet_id, 0,0, Shimtile_MM2S_X_TASK_QUEUE_addr);
    
    // assume not enable token issue
    // repeat_count = 1-0
    if(! issue_token){
        *(control_packet_out_buf+2) =  0<<16 | Shimtile_bd_id;
    }else{
        *(control_packet_out_buf+2) = 1<<31 |   0<<16 | Shimtile_bd_id;
    }

    release(control_packet_out_con_lock, 1);

}





#endif