/**
 * @file wake_led.c
 * @brief Onboard LED control for wake-word detection indication.
 *
 * Target: ESP32 DevKit V1 — GPIO 2 is the onboard LED.
 *
 * Implementation deliberately avoids:
 *   - Dynamic memory allocation
 *   - FreeRTOS tasks or timers
 *   - Any PWM / LEDC peripheral
 *
 * A single gpio_set_level() call is safe to invoke from any
 * FreeRTOS task.
 */

#include "wake_led.h"

#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"

#include <stdbool.h>

static const char *TAG = "WAKE_LED";

esp_err_t wake_led_init(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << WAKE_LED_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };

    esp_err_t err = gpio_config(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "gpio_config failed: %s", esp_err_to_name(err));
        return err;
    }

    /* Start in idle (LED off) state. */
    gpio_set_level(WAKE_LED_GPIO, 0);
    ESP_LOGI(TAG, "Wake LED ready on GPIO %d", WAKE_LED_GPIO);
    return ESP_OK;
}

void wake_led_set(bool active)
{
    gpio_set_level(WAKE_LED_GPIO, active ? 1 : 0);
}
