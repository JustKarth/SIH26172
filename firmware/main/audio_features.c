#include "audio_features.h"

#include <math.h>
#include <string.h>

#include "esp_log.h"

static const char *TAG = "AUDIO_FEATURES";

#define FRAME_SIZE       400
#define FFT_SIZE         512
#define NUM_MEL          40

#define SAMPLE_RATE      16000.0f

#define LOW_FREQ         20.0f
#define HIGH_FREQ        8000.0f

#define PI                3.14159265358979323846f

/*
 * FFT buffers.
 */
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];

/*
 * Precomputed Hamming window.
 */
static float hamming_window[FRAME_SIZE];

/*
 * Mel filter bank.
 *
 * 40 filters × 257 positive-frequency bins.
 */
static float mel_filters[
    NUM_MEL
][
    FFT_SIZE / 2 + 1
];

static bool initialized = false;


/* -----------------------------------------------------------
 * Utility
 * --------------------------------------------------------- */

static float hz_to_mel(float hz)
{
    return 2595.0f *
           log10f(
               1.0f + hz / 700.0f
           );
}


static float mel_to_hz(float mel)
{
    return 700.0f *
           (
               powf(
                   10.0f,
                   mel / 2595.0f
               ) - 1.0f
           );
}


/* -----------------------------------------------------------
 * Precomputation
 * --------------------------------------------------------- */

static void init_hamming_window(void)
{
    for (int n = 0; n < FRAME_SIZE; n++) {

        hamming_window[n] =
            0.54f -
            0.46f *
            cosf(
                2.0f *
                PI *
                n /
                (FRAME_SIZE - 1)
            );
    }
}


static void init_mel_filters(void)
{
    float low_mel =
        hz_to_mel(LOW_FREQ);

    float high_mel =
        hz_to_mel(HIGH_FREQ);


    /*
     * 42 points create
     * 40 triangular filters.
     */
    float mel_points[NUM_MEL + 2];

    int bin_points[NUM_MEL + 2];


    for (int i = 0;
         i < NUM_MEL + 2;
         i++) {

        mel_points[i] =
            low_mel +
            (
                (float)i /
                (float)(NUM_MEL + 1)
            ) *
            (
                high_mel -
                low_mel
            );

        float hz =
            mel_to_hz(
                mel_points[i]
            );

        bin_points[i] =
            (int)floorf(
                ((FFT_SIZE + 1) * hz) /
                SAMPLE_RATE
            );
    }


    memset(
        mel_filters,
        0,
        sizeof(mel_filters)
    );


    for (int m = 1;
         m <= NUM_MEL;
         m++) {

        int left =
            bin_points[m - 1];

        int center =
            bin_points[m];

        int right =
            bin_points[m + 1];


        /*
         * Rising slope.
         */
        for (int k = left;
             k < center;
             k++) {

            if (k >= 0 &&
                k <= FFT_SIZE / 2 &&
                center != left) {

                mel_filters[m - 1][k] =
                    (float)(k - left) /
                    (float)(center - left);
            }
        }


        /*
         * Falling slope.
         */
        for (int k = center;
             k < right;
             k++) {

            if (k >= 0 &&
                k <= FFT_SIZE / 2 &&
                right != center) {

                mel_filters[m - 1][k] =
                    (float)(right - k) /
                    (float)(right - center);
            }
        }
    }
}


/* -----------------------------------------------------------
 * FFT
 * --------------------------------------------------------- */

/*
 * In-place radix-2 FFT.
 *
 * This is intentionally straightforward.
 * We will optimize it later if profiling requires it.
 */
static void fft(void)
{
    /*
     * Bit-reversal permutation.
     */
    int j = 0;

    for (int i = 1;
         i < FFT_SIZE;
         i++) {

        int bit =
            FFT_SIZE >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {

            float temp =
                fft_real[i];

            fft_real[i] =
                fft_real[j];

            fft_real[j] =
                temp;


            temp =
                fft_imag[i];

            fft_imag[i] =
                fft_imag[j];

            fft_imag[j] =
                temp;
        }
    }


    /*
     * FFT butterfly stages.
     */
    for (int length = 2;
         length <= FFT_SIZE;
         length <<= 1) {

        float angle =
            -2.0f *
            PI /
            (float)length;

        float w_real =
            cosf(angle);

        float w_imag =
            sinf(angle);


        for (int i = 0;
             i < FFT_SIZE;
             i += length) {

            float current_real = 1.0f;
            float current_imag = 0.0f;


            for (int j = 0;
                 j < length / 2;
                 j++) {

                int even =
                    i + j;

                int odd =
                    i + j +
                    length / 2;


                float odd_real =
                    fft_real[odd] *
                    current_real -
                    fft_imag[odd] *
                    current_imag;

                float odd_imag =
                    fft_real[odd] *
                    current_imag +
                    fft_imag[odd] *
                    current_real;


                float even_real =
                    fft_real[even];

                float even_imag =
                    fft_imag[even];


                fft_real[even] =
                    even_real +
                    odd_real;

                fft_imag[even] =
                    even_imag +
                    odd_imag;


                fft_real[odd] =
                    even_real -
                    odd_real;

                fft_imag[odd] =
                    even_imag -
                    odd_imag;


                float next_real =
                    current_real *
                    w_real -
                    current_imag *
                    w_imag;

                float next_imag =
                    current_real *
                    w_imag +
                    current_imag *
                    w_real;


                current_real =
                    next_real;

                current_imag =
                    next_imag;
            }
        }
    }
}


/* -----------------------------------------------------------
 * Feature extraction
 * --------------------------------------------------------- */

esp_err_t audio_features_process(
    const int32_t *raw_frame,
    float *features)
{
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }

    if (raw_frame == NULL ||
        features == NULL) {

        return ESP_ERR_INVALID_ARG;
    }


    /*
     * Convert raw I2S → PCM16 → normalized float.
     *
     * Apply pre-emphasis while doing the conversion.
     */
    float previous =
        0.0f;


    for (int i = 0;
         i < FRAME_SIZE;
         i++) {

        int16_t pcm =
            (int16_t)(
                raw_frame[i] >> 16
            );


        float sample =
            (float)pcm /
            32768.0f;


        float emphasized;

        if (i == 0) {

            emphasized =
                sample;

        } else {

            emphasized =
                sample -
                0.97f * previous;
        }


        previous =
            sample;


        fft_real[i] =
            emphasized *
            hamming_window[i];

        fft_imag[i] =
            0.0f;
    }


    /*
     * Zero padding:
     *
     * 400 real samples
     * →
     * 512 FFT samples.
     */
    for (int i = FRAME_SIZE;
         i < FFT_SIZE;
         i++) {

        fft_real[i] = 0.0f;
        fft_imag[i] = 0.0f;
    }


    /*
     * FFT.
     */
    fft();


    /*
     * Power spectrum.
     */
    float power[
        FFT_SIZE / 2 + 1
    ];


    for (int k = 0;
         k <= FFT_SIZE / 2;
         k++) {

        power[k] =
            fft_real[k] *
            fft_real[k] +
            fft_imag[k] *
            fft_imag[k];
    }


    /*
     * Apply Mel filter bank.
     */
    for (int m = 0;
         m < NUM_MEL;
         m++) {

        float energy = 0.0f;


        for (int k = 0;
             k <= FFT_SIZE / 2;
             k++) {

            energy +=
                power[k] *
                mel_filters[m][k];
        }


        /*
         * Prevent log(0).
         */
        if (energy < 1e-10f) {
            energy = 1e-10f;
        }


        /*
         * Natural logarithm.
         *
         * We will verify whether the Python
         * reference uses ln or log10 later.
         */
        features[m] =
            logf(energy);
    }


    return ESP_OK;
}


/* -----------------------------------------------------------
 * Initialization
 * --------------------------------------------------------- */

esp_err_t audio_features_init(void)
{
    ESP_LOGI(
        TAG,
        "Initializing feature extractor"
    );


    init_hamming_window();

    init_mel_filters();


    initialized = true;


    ESP_LOGI(
        TAG,
        "FFT=%d, Mel filters=%d",
        FFT_SIZE,
        NUM_MEL
    );


    return ESP_OK;
}