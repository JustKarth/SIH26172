#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#define AUDIO_FEATURE_FFT_SIZE       512
#define AUDIO_FEATURE_NUM_MEL        40
#define AUDIO_FEATURE_SAMPLE_RATE    16000

/*
 * Process one 25 ms audio frame.
 *
 * Input:
 *     400 raw I2S samples
 *
 * Output:
 *     40 log-Mel features
 */
esp_err_t audio_features_init(void);

esp_err_t audio_features_process(
    const int32_t *raw_frame,
    float *features
);

#endif