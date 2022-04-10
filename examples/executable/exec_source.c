#include <stdio.h>
#include <math.h>
#include "xtrack.h"

int main(){
    printf("Hello, world!\n");

    FILE *ptr;
    ptr = fopen("./part.bin", "rb");

    int8_t part_buffer[1216]; // Hardcoded size for now

    fread(part_buffer, sizeof(int8_t), 1216, ptr);

    printf("%d\n", part_buffer[0]);

    ParticlesData part = (ParticlesData) part_buffer;

    for (int ii=0; ii<ParticlesData_get__capacity(part); ii++){
        printf("x[%d] = %e\n", ii, ParticlesData_get_x(part, (int64_t) ii));
    }
    // This is what we want to call
    //track_line(
    //    int8_t* buffer,
    //    int64_t* ele_offsets,
    //    int64_t* ele_typeids,
    //    ParticlesData particles,
    //    int num_turns,
    //    int ele_start,
    //    int num_ele_track,
    //    int flag_end_turn_actions,
    //    int flag_reset_s_at_end_turn,
    //    int flag_monitor,
    //    int8_t* buffer_tbt_monitor,
    //    int64_t offset_tbt_monitor);


    return 0;
}