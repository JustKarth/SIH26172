/**
 * @file wake_led.h
 * @brief Onboard LED control for wake-word detection indication.
 *
 * Uses the ESP32 DevKit V1 onboard LED on GPIO 2.
 * The LED turns ON when a wake event is triggered and turns
 * OFF when the device returns to idle/listening state.
 *
 * This module is intentionally simple: no PWM, no RTOS task,
 * no dynamic allocation.  It only drives one GPIO output.
 */

#ifndef WAKE_LED_H
#define WAKE_LED_H

#include "esp_err.h"

/** GPIO number used by the ESP32 DevKit V1 onboard LED. */
#define WAKE_LED_GPIO 2

/**
 * @brief Initialise the onboard LED GPIO as an output.
 *
 * Must be called once before wake_led_set().
 * Leaves the LED in the OFF (idle) state.
 *
 * @return ESP_OK on success, or an ESP-IDF error code.
 */
esp_err_t wake_led_init(void);

/**
 * @brief Set the LED state.
 *
 * @param active true  → LED ON  (wake word detected / streaming)
 *               false → LED OFF (idle / listening)
 */
void wake_led_set(bool active);

#endif /* WAKE_LED_H */
