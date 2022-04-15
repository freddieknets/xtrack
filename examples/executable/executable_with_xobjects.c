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

    const int64_t num_turns = SimConfig_get_num_turns(sim_config);
    const int64_t num_elements = SimConfig_len_line_metadata_ele_offsets(sim_config);

    printf("num_turns: %d\n", (int) num_turns);
    printf("num_elements: %d\n", (int) num_elements);

    int64_t* line_ele_offsets = SimConfig_getp1_line_metadata_ele_offsets(sim_config, 0);
    int64_t* line_ele_typeids = SimConfig_getp1_line_metadata_ele_typeids(sim_config, 0);
    ParticlesData particles = SimConfig_getp_sim_state_particles(sim_config);

    // This is what we want to call
    track_line(
          sim_buffer, //    int8_t* buffer,
          line_ele_offsets, //    int64_t* ele_offsets,
          line_ele_typeids, //    int64_t* ele_typeids,
          particles, //    ParticlesData particles,
          (int) num_turns, //    int num_turns,
          0, //    int ele_start,
          (int) num_elements, //    int num_ele_track,
          0, //int flag_end_turn_actions,
          0, //int flag_reset_s_at_end_turn,
          0, //    int flag_monitor,
          NULL,//    int8_t* buffer_tbt_monitor,
          0//    int64_t offset_tbt_monitor
    );

    // Quick check
    for (int ii=0; ii<ParticlesData_get__capacity(particles); ii++){
        printf("s[%d] = %e\n", ii, ParticlesData_get_s(particles, (int64_t) ii));
    }

}