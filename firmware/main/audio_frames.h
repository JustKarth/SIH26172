#ifndef AUDIO_FRAMES_H
#define AUDIO_FRAMES_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#define AUDIO_FRAME_SIZE 400
#define AUDIO_FRAME_HOP  160

#define AUDIO_FRAME_DURATION_MS 25
#define AUDIO_FRAME_HOP_MS      10

typedef struct {
    uint64_t sample_index;
    uint32_t frame_number;
} audio_frame_info_t;

esp_err_t audio_frames_init(void);

size_t audio_frames_available(void);

esp_err_t audio_frames_get(
    int32_t *frame,
    audio_frame_info_t *info
);

uint32_t audio_frames_get_missed(void);

#endif