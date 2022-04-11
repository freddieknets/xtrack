#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "xtrack.h"

int main(){
    printf("Hello, world!\n");

    FILE *conf_fid;
    conf_fid = fopen("./conf.txt", "r");
    int part_buffer_size;
    fscanf(conf_fid,"%d",&part_buffer_size);
    int part_offset;
    fscanf(conf_fid,"%d",&part_offset);
    printf("part buffer size: %d\n", part_buffer_size);
    printf("part offset: %d\n", part_offset);


    FILE *ptr;
    ptr = fopen("./part.bin", "rb");
    int8_t* part_buffer = malloc(part_buffer_size*sizeof(double));
    fread(part_buffer, sizeof(int8_t), part_buffer_size, ptr);

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