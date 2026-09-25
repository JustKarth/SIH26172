#include "audio_pipeline.h"
#include "audio_frames.h"
#include "audio_features.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#include "esp_log.h"
#include <stdint.h>
#include <math.h>

static const char *TAG = "MAIN";


static void feature_test_task(void *arg)
{
    int32_t frame[AUDIO_FRAME_SIZE];

    float features[AUDIO_FEATURE_NUM_MEL];

    audio_frame_info_t info;

    uint32_t processed = 0;

    while (1) {

    if (audio_frames_available()) {

        esp_err_t err =
            audio_frames_get(
                frame,
                &info
            );

        if (err == ESP_OK) {

            int64_t start =
                esp_timer_get_time();

            err =
                audio_features_process(
                    frame,
                    features
                );

            int64_t elapsed =
                esp_timer_get_time() -
                start;

            if (err == ESP_OK) {

                processed++;

                if ((processed % 100) == 0) {

                    ESP_LOGI(
                        "FEATURE_TEST",

                        "frame=%lu "
                        "sample=%llu "
                        "time=%lld us "
                        "mel[0]=%.4f "
                        "mel[20]=%.4f "
                        "mel[39]=%.4f",

                        (unsigned long)
                            info.frame_number,

                        (unsigned long long)
                            info.sample_index,

                        (long long)
                            elapsed,

                        features[0],
                        features[20],
                        features[39]
                    );
                }
            }
        }

        /*
         * Give the other tasks/system CPU time
         * after processing a frame.
         */
        vTaskDelay(pdMS_TO_TICKS(5));

    } else {

        /*
         * No frame available yet.
         * Sleep longer instead of polling rapidly.
         */
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    }
}


void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting audio system"
    );

    ESP_ERROR_CHECK(
        audio_pipeline_init()
    );

    ESP_ERROR_CHECK(
        audio_pipeline_start()
    );

    ESP_ERROR_CHECK(
        audio_frames_init()
    );

    ESP_ERROR_CHECK(
        audio_features_init()
    );

    ESP_LOGI(
        TAG,
        "Audio system running"
    );

    xTaskCreate(
        feature_test_task,
        "feature_test",
        8192,
        NULL,
        3,
        NULL
    );
}   