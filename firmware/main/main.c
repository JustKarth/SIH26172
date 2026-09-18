#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"

#define I2S_PORT    I2S_NUM_0

#define PIN_SCK     GPIO_NUM_26
#define PIN_WS      GPIO_NUM_25
#define PIN_SD      GPIO_NUM_33

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256

static const char *TAG = "MIC_TEST";

void app_main(void)
{
    i2s_chan_handle_t rx_handle;

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &rx_handle)
    );

    i2s_std_config_t i2s_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO
        ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_SCK,
            .ws   = PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_SD,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    i2s_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(rx_handle, &i2s_cfg)
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_handle)
    );

    ESP_LOGI(TAG, "INMP441 microphone started");

    int32_t samples[BUFFER_SIZE];
    size_t bytes_read;

    while (1)
    {
        ESP_ERROR_CHECK(
            i2s_channel_read(
                rx_handle,
                samples,
                sizeof(samples),
                &bytes_read,
                portMAX_DELAY
            )
        );

        int count = bytes_read / sizeof(int32_t);

        /*
         * Send every 4th sample.
         * This gives 64 samples per buffer and keeps
         * serial output manageable.
         */
        printf("DATA ");

        for (int i = 0; i < count; i += 4)
        {
            /*
             * INMP441 data is 24-bit inside the 32-bit slot.
             * Shift down to obtain a useful signed amplitude.
             */
            int32_t sample = samples[i] >> 8;

            printf("%" PRId32, sample);

            if (i + 4 < count)
                printf(",");
        }

        printf("\n");
        fflush(stdout);
    }
}