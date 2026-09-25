#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#define AUDIO_SAMPLE_RATE       16000
#define AUDIO_CHANNELS          1
#define AUDIO_RING_SAMPLES      16000
#define AUDIO_DMA_READ_SAMPLES  256

typedef struct {
    uint32_t samples_written;
    uint32_t samples_dropped;
    uint32_t available_samples;
} audio_stats_t;

esp_err_t audio_pipeline_init(void);

esp_err_t audio_pipeline_start(void);

size_t audio_pipeline_read(
    int32_t *destination,
    size_t samples
);

size_t audio_pipeline_available(void);

size_t audio_pipeline_peek(
    int32_t *destination,
    size_t offset,
    size_t count
);

/*
 * Total number of samples captured since boot.
 * This value only increases.
 */
uint64_t audio_pipeline_total_samples(void);

void audio_pipeline_get_stats(
    audio_stats_t *stats
);

#endif