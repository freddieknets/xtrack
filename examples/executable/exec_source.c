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
    int line_buffer_size;
    fscanf(conf_fid,"%d",&line_buffer_size);
    int num_elements;
    fscanf(conf_fid,"%d",&num_elements);

    printf("part buffer size: %d\n", part_buffer_size);
    printf("part offset: %d\n", part_offset);
    printf("line buffer size: %d\n", line_buffer_size);
    printf("num elements: %d\n", num_elements);

    FILE *part_fid;
    part_fid = fopen("./part.bin", "rb");
    int8_t* part_buffer = malloc(part_buffer_size*sizeof(int8_t));
    fread(part_buffer, sizeof(int8_t), part_buffer_size, part_fid);

    FILE *line_ele_offsets_fid;
    line_ele_offsets_fid = fopen("./line_ele_offsets.bin", "rb");
    int64_t* line_ele_offsets = malloc(num_elements*sizeof(int64_t));
    fread(line_ele_offsets, sizeof(int64_t), num_elements, line_ele_offsets_fid);
    for (int ii=0; ii<num_elements; ii++){
        printf("offs[%d] = %d\n", ii, (int)line_ele_offsets[ii]);
    }

    FILE *line_ele_typeids_fid;
    line_ele_typeids_fid = fopen("./line_ele_typeids.bin", "rb");
    int64_t* line_ele_typeids = malloc(num_elements*sizeof(int64_t));
    fread(line_ele_typeids, sizeof(int64_t), num_elements, line_ele_typeids_fid);
    for (int ii=0; ii<num_elements; ii++){
        printf("typeid[%d] = %d\n", ii, (int)line_ele_typeids[ii]);
    }

    printf("%d\n", part_buffer[0]);
    ParticlesData part = (ParticlesData) (part_buffer + part_offset);
    for (int ii=0; ii<ParticlesData_get__capacity(part); ii++){
        printf("x[%d] = %e\n", ii, ParticlesData_get_x(part, (int64_t) ii));
    }

    FILE *line_fid;
    line_fid = fopen("./line.bin", "rb");
    int8_t* line_buffer = malloc(line_buffer_size*sizeof(int8_t));
    fread(line_buffer, sizeof(int8_t), line_buffer_size, line_fid);

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


    for (int ii=0; ii<ParticlesData_get__capacity(part); ii++){
        printf("s[%d] = %e\n", ii, ParticlesData_get_s(part, (int64_t) ii));
    }

    FILE *part_out_fid;
    part_out_fid = fopen("./part_out.bin", "wb");
    fwrite(part_buffer, sizeof(int8_t), part_buffer_size, part_out_fid);
    return 0;
}