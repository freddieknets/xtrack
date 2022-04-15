#include <math.h>

#include "xtrack.h"
#include "simconfig.h"

#include <stdio.h>
#include <stdlib.h>

int main(){

    FILE *sim_fid;

    // Get buffer size
    sim_fid = fopen("./sim.bin", "rb");
    int64_t sim_buffer_size;
    fread(&sim_buffer_size, sizeof(int64_t), 1, sim_fid);
    fclose(sim_fid);

    printf("sim buffer size: %d\n", (int) sim_buffer_size);

    // Get buffer
    sim_fid = fopen("./sim.bin", "rb");
    int8_t* sim_buffer = malloc(sim_buffer_size*sizeof(int8_t));
    fread(sim_buffer, sizeof(int8_t), sim_buffer_size, sim_fid);
    fclose(sim_fid);

    // Get sim config
    SimConfig sim_config = (SimConfig) sim_buffer;

    printf("num_turns: %d\n", (int) SimConfig_get_num_turns(sim_config));

    /*
    // This is what we want to call
    track_line(
          line_buffer, //    int8_t* buffer,
          line_ele_offsets, //    int64_t* ele_offsets,
          line_ele_typeids, //    int64_t* ele_typeids,
          part, //    ParticlesData particles,
          2, //    int num_turns,
          0, //    int ele_start,
          num_elements, //    int num_ele_track,
          0, //int flag_end_turn_actions,
          0, //int flag_reset_s_at_end_turn,
          0, //    int flag_monitor,
          NULL,//    int8_t* buffer_tbt_monitor,
          0//    int64_t offset_tbt_monitor
    );
    */

}