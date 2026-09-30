#include "audio_pipeline.h"

#include <string.h>

#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "AUDIO";

static i2s_chan_handle_t rx_channel;


/*
 * -----------------------------------------------------------
 * Application-level rolling audio buffer
 * -----------------------------------------------------------
 *
 * The buffer continuously retains the most recent audio.
 *
 * At 16 kHz:
 *
 *     16000 samples = approximately 1 second
 *
 * Once the buffer becomes full, new samples overwrite
 * the oldest samples.
 */
static int32_t audio_ring[AUDIO_RING_SAMPLES];

static size_t write_index = 0;
static size_t read_index = 0;


/*
 * Total number of samples received since boot.
 *
 * Unlike write_index, this never wraps around.
 *
 * Example:
 *
 *     after 1 second  -> ~16000
 *     after 2 seconds  -> ~32000
 *     after 10 seconds -> ~160000
 */
static uint64_t total_samples_written = 0;


/*
 * Number of samples that have been overwritten because
 * the rolling buffer was full.
 */
static uint32_t samples_dropped = 0;


/*
 * Number of samples successfully written into the
 * application ring.
 */
static uint32_t samples_written = 0;


/*
 * Protects all ring-buffer state.
 */
static portMUX_TYPE audio_ring_lock =
    portMUX_INITIALIZER_UNLOCKED;


/* -----------------------------------------------------------
 * Circular buffer helpers
 * --------------------------------------------------------- */

static size_t ring_available_unsafe(void)
{
    if (write_index >= read_index) {
        return write_index - read_index;
    }

    return AUDIO_RING_SAMPLES
         - read_index
         + write_index;
}


/*
 * Write samples into the rolling buffer.
 *
 * If the buffer is full, the oldest sample is overwritten.
 */
static size_t ring_write(
    const int32_t *samples,
    size_t count)
{
    size_t written = 0;

    portENTER_CRITICAL(&audio_ring_lock);

    for (size_t i = 0; i < count; i++) {

        /*
         * Calculate where write_index would move next.
         */
        size_t next_write = write_index + 1;

        if (next_write >= AUDIO_RING_SAMPLES) {
            next_write = 0;
        }

        /*
         * If next_write would collide with read_index,
         * the ring is full.
         */
        if (next_write == read_index) {

            /*
             * Move the oldest-sample pointer forward.
             *
             * This means the oldest sample is discarded.
             */
            read_index++;

            if (read_index >= AUDIO_RING_SAMPLES) {
                read_index = 0;
            }

            samples_dropped++;
        }

        /*
         * Store the new sample.
         */
        audio_ring[write_index] = samples[i];

        /*
         * Advance write position.
         */
        write_index = next_write;

        /*
         * Statistics.
         */
        samples_written++;
        total_samples_written++;

        written++;
    }

    portEXIT_CRITICAL(&audio_ring_lock);

    return written;
}


/* -----------------------------------------------------------
 * Public ring-buffer API
 * --------------------------------------------------------- */

size_t audio_pipeline_read(
    int32_t *destination,
    size_t count)
{
    size_t read = 0;

    if (destination == NULL || count == 0) {
        return 0;
    }

    portENTER_CRITICAL(&audio_ring_lock);

    size_t available =
        ring_available_unsafe();

    size_t to_read = count;

    if (to_read > available) {
        to_read = available;
    }

    for (size_t i = 0; i < to_read; i++) {

        destination[i] =
            audio_ring[read_index];

        read_index++;

        if (read_index >= AUDIO_RING_SAMPLES) {
            read_index = 0;
        }
    }

    read = to_read;

    portEXIT_CRITICAL(&audio_ring_lock);

    return read;
}


size_t audio_pipeline_available(void)
{
    size_t available;

    portENTER_CRITICAL(&audio_ring_lock);

    available =
        ring_available_unsafe();

    portEXIT_CRITICAL(&audio_ring_lock);

    return available;
}


/*
 * Return the total number of samples captured since boot.
 *
 * This is different from write_index:
 *
 * write_index:
 *     wraps around the ring
 *
 * total_samples_written:
 *     keeps increasing
 */
uint64_t audio_pipeline_total_samples(void)
{
    uint64_t total;

    portENTER_CRITICAL(&audio_ring_lock);

    total = total_samples_written;

    portEXIT_CRITICAL(&audio_ring_lock);

    return total;
}


/*
 * Copy samples from the ring without consuming them.
 *
 * offset = 0
 *     oldest currently retained sample
 *
 * offset = 1
 *     second-oldest sample
 *
 * etc.
 *
 * This is what the frame extractor will use.
 */
size_t audio_pipeline_peek(
    int32_t *destination,
    size_t offset,
    size_t count)
{
    size_t copied = 0;

    if (destination == NULL || count == 0) {
        return 0;
    }

    portENTER_CRITICAL(&audio_ring_lock);

    size_t available =
        ring_available_unsafe();

    /*
     * Offset must point inside the available data.
     */
    if (offset >= available) {

        portEXIT_CRITICAL(&audio_ring_lock);

        return 0;
    }

    /*
     * Do not read beyond available samples.
     */
    size_t remaining =
        available - offset;

    if (count > remaining) {
        count = remaining;
    }

    /*
     * Convert the logical offset into a physical
     * ring-buffer index.
     */
    size_t index =
        read_index + offset;

    if (index >= AUDIO_RING_SAMPLES) {
        index -= AUDIO_RING_SAMPLES;
    }

    /*
     * Copy without modifying read_index.
     */
    for (size_t i = 0; i < count; i++) {

        destination[i] =
            audio_ring[index];

        index++;

        if (index >= AUDIO_RING_SAMPLES) {
            index = 0;
        }
    }

    copied = count;

    portEXIT_CRITICAL(&audio_ring_lock);

    return copied;
}


/* -----------------------------------------------------------
 * I2S / DMA
 * --------------------------------------------------------- */

static esp_err_t setup_i2s(void)
{
    i2s_chan_config_t channel_config =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER
        );

    channel_config.dma_desc_num = 8;

    channel_config.dma_frame_num =
        AUDIO_DMA_READ_SAMPLES;

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &channel_config,
            NULL,
            &rx_channel
        )
    );

    i2s_std_config_t std_config = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                AUDIO_SAMPLE_RATE
            ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,

            .bclk = GPIO_NUM_26,

            .ws = GPIO_NUM_25,

            .dout = I2S_GPIO_UNUSED,

            .din = GPIO_NUM_33,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    /*
     * INMP441 L/R is connected to GND,
     * therefore use the left slot.
     */
    std_config.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            rx_channel,
            &std_config
        )
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_channel)
    );

    ESP_LOGI(
        TAG,
        "I2S initialized"
    );

    ESP_LOGI(
        TAG,
        "Sample rate : %d Hz",
        AUDIO_SAMPLE_RATE
    );

    ESP_LOGI(
        TAG,
        "Sample size : 32 bits"
    );

    ESP_LOGI(
        TAG,
        "DMA samples : %d",
        AUDIO_DMA_READ_SAMPLES
    );

    return ESP_OK;
}


/* -----------------------------------------------------------
 * Audio acquisition task
 * --------------------------------------------------------- */

static void audio_capture_task(void *arg)
{
    int32_t dma_samples[
        AUDIO_DMA_READ_SAMPLES
    ];

    ESP_LOGI(
        TAG,
        "Audio capture task started"
    );

    while (1) {

        size_t bytes_read = 0;

        esp_err_t err =
            i2s_channel_read(
                rx_channel,
                dma_samples,
                sizeof(dma_samples),
                &bytes_read,
                portMAX_DELAY
            );

        if (err != ESP_OK) {

            ESP_LOGE(
                TAG,
                "I2S read failed: %s",
                esp_err_to_name(err)
            );

            continue;
        }

        size_t samples_read =
            bytes_read / sizeof(int32_t);

        /*
         * Copy the DMA-delivered block into
         * our rolling application buffer.
         */
        ring_write(
            dma_samples,
            samples_read
        );
    }
}


/* -----------------------------------------------------------
 * Public API
 * --------------------------------------------------------- */

esp_err_t audio_pipeline_init(void)
{
    write_index = 0;

    read_index = 0;

    total_samples_written = 0;

    samples_written = 0;

    samples_dropped = 0;

    memset(
        audio_ring,
        0,
        sizeof(audio_ring)
    );

    return setup_i2s();
}


esp_err_t audio_pipeline_start(void)
{
    BaseType_t result =
        xTaskCreate(
            audio_capture_task,
            "audio_capture",
            4096,
            NULL,
            5,
            NULL
        );

    if (result != pdPASS) {

        ESP_LOGE(
            TAG,
            "Failed to create audio task"
        );

        return ESP_FAIL;
    }

    return ESP_OK;
}


void audio_pipeline_get_stats(
    audio_stats_t *stats)
{
    if (stats == NULL) {
        return;
    }

    portENTER_CRITICAL(&audio_ring_lock);

    stats->samples_written =
        samples_written;

    stats->samples_dropped =
        samples_dropped;

    stats->available_samples =
        ring_available_unsafe();

    portEXIT_CRITICAL(&audio_ring_lock);
}