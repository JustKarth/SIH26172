#include "audio_frames.h"
#include "audio_pipeline.h"

#include "esp_log.h"

static const char *TAG = "AUDIO_FRAMES";


/*
 * Absolute sample number corresponding to the
 * beginning of the next frame.
 */
static uint64_t next_frame_sample = 0;


/*
 * Sequential frame number.
 */
static uint32_t frame_number = 0;


/*
 * Number of times the frame extractor had to
 * skip forward because old samples were overwritten.
 */
static uint32_t missed_frames = 0;


static bool initialized = false;


/*
 * Determine the absolute sample number of the
 * oldest sample currently retained by the ring.
 *
 * Example:
 *
 * total = 20000
 * available = 15999
 *
 * oldest = 4001
 */
static uint64_t get_oldest_sample(void)
{
    uint64_t total =
        audio_pipeline_total_samples();

    size_t available =
        audio_pipeline_available();

    if (total < available) {
        return 0;
    }

    return total - available;
}


esp_err_t audio_frames_init(void)
{
    next_frame_sample = 0;

    frame_number = 0;

    missed_frames = 0;

    initialized = true;

    ESP_LOGI(
        TAG,
        "Frame extractor initialized"
    );

    ESP_LOGI(
        TAG,
        "Frame size : %d samples (%d ms)",
        AUDIO_FRAME_SIZE,
        AUDIO_FRAME_DURATION_MS
    );

    ESP_LOGI(
        TAG,
        "Frame hop  : %d samples (%d ms)",
        AUDIO_FRAME_HOP,
        AUDIO_FRAME_HOP_MS
    );

    return ESP_OK;
}


size_t audio_frames_available(void)
{
    if (!initialized) {
        return 0;
    }

    uint64_t oldest =
        get_oldest_sample();

    uint64_t total =
        audio_pipeline_total_samples();

    /*
     * We need 400 complete samples.
     */
    if (total <
        next_frame_sample + AUDIO_FRAME_SIZE) {

        return 0;
    }

    /*
     * The frame we wanted has already
     * been overwritten.
     */
    if (next_frame_sample < oldest) {
        return 0;
    }

    return 1;
}


esp_err_t audio_frames_get(
    int32_t *frame,
    audio_frame_info_t *info)
{
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (frame == NULL || info == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint64_t oldest =
        get_oldest_sample();

    uint64_t total =
        audio_pipeline_total_samples();


    /*
     * If our frame cursor has fallen behind
     * the rolling buffer, resynchronize it.
     */
    if (next_frame_sample < oldest) {

        ESP_LOGW(
            TAG,
            "Frame cursor behind ring. "
            "Resyncing from %llu to %llu",
            (unsigned long long)next_frame_sample,
            (unsigned long long)oldest
        );

        /*
         * Align to the next frame-hop boundary.
         *
         * This keeps subsequent frames spaced
         * exactly 160 samples apart.
         */
        uint64_t remainder =
            oldest % AUDIO_FRAME_HOP;

        if (remainder != 0) {
            oldest +=
                AUDIO_FRAME_HOP - remainder;
        }

        next_frame_sample = oldest;

        missed_frames++;
    }


    /*
     * Check that a complete 400-sample frame
     * is currently available.
     */
    if (total <
        next_frame_sample + AUDIO_FRAME_SIZE) {

        return ESP_ERR_INVALID_STATE;
    }


    /*
     * Calculate where the desired frame starts
     * relative to the oldest retained sample.
     */
    size_t offset =
        (size_t)(
            next_frame_sample - oldest
        );


    /*
     * Safety check.
     */
    if (offset + AUDIO_FRAME_SIZE >
        audio_pipeline_available()) {

        return ESP_ERR_INVALID_STATE;
    }


    /*
     * Copy the frame WITHOUT consuming it.
     */
    size_t copied =
        audio_pipeline_peek(
            frame,
            offset,
            AUDIO_FRAME_SIZE
        );

    if (copied != AUDIO_FRAME_SIZE) {
        return ESP_FAIL;
    }


    /*
     * Return frame metadata.
     */
    info->sample_index =
        next_frame_sample;

    info->frame_number =
        frame_number;


    /*
     * Advance by 160 samples.
     *
     * Therefore:
     *
     * Frame 0: 0   -> 399
     * Frame 1: 160 -> 559
     * Frame 2: 320 -> 719
     * Frame 3: 480 -> 879
     */
    next_frame_sample +=
        AUDIO_FRAME_HOP;

    frame_number++;

    return ESP_OK;
}


uint32_t audio_frames_get_missed(void)
{
    return missed_frames;
}