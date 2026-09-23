#include <Arduino.h>
#include "driver/i2s.h"
#include <math.h>

// =====================================================
// I2S CONFIGURATION
// =====================================================

#define I2S_PORT I2S_NUM_0

#define I2S_BCLK 26
#define I2S_WS   25
#define I2S_SD   33

#define SAMPLE_RATE 16000

// =====================================================
// AUDIO PARAMETERS
// =====================================================

#define WINDOW_SAMPLES 16000     // 1 second

#define FRAME_SIZE 400           // 25 ms @ 16 kHz
#define HOP_SIZE 160             // 10 ms @ 16 kHz

#define FFT_SIZE 512

#define NUM_MEL 40

// Number of frames in 1 second
#define NUM_FRAMES \
    (1 + (WINDOW_SAMPLES - FRAME_SIZE) / HOP_SIZE)

// =====================================================
// AUDIO MEMORY
// =====================================================

// 1 second PCM16 audio
int16_t audioBuffer[WINDOW_SAMPLES];

// =====================================================
// ML MATRIX
// =====================================================

float melMatrix[NUM_FRAMES][NUM_MEL];

// =====================================================
// FFT MEMORY
// =====================================================

float fftReal[FFT_SIZE];
float fftImag[FFT_SIZE];

// =====================================================
// I2S MEMORY
// =====================================================

#define I2S_BUFFER_SIZE 256

int32_t i2sBuffer[I2S_BUFFER_SIZE];

// =====================================================
// MEL FILTER BANK
// =====================================================

float melFilters[NUM_MEL][FFT_SIZE / 2 + 1];


// =====================================================
// MEL SCALE
// =====================================================

float hzToMel(float hz)
{
    return 2595.0 *
           log10(1.0 + hz / 700.0);
}

float melToHz(float mel)
{
    return 700.0 *
           (pow(10.0, mel / 2595.0) - 1.0);
}


// =====================================================
// CREATE MEL FILTER BANK
// =====================================================

void createMelFilters()
{
    Serial.println("Creating Mel filter bank...");

    const float FMIN = 20.0;
    const float FMAX = 8000.0;

    float melMin = hzToMel(FMIN);
    float melMax = hzToMel(FMAX);

    // 42 points for 40 triangular filters
    float melPoints[NUM_MEL + 2];

    for (int i = 0;
         i < NUM_MEL + 2;
         i++)
    {
        melPoints[i] =
            melMin +
            (melMax - melMin) *
            i /
            (NUM_MEL + 1);
    }

    int bin[NUM_MEL + 2];

    for (int i = 0;
         i < NUM_MEL + 2;
         i++)
    {
        float freq =
            melToHz(melPoints[i]);

        bin[i] =
            floor(
                (FFT_SIZE + 1) *
                freq /
                SAMPLE_RATE
            );

        if (bin[i] < 0)
            bin[i] = 0;

        if (bin[i] > FFT_SIZE / 2)
            bin[i] = FFT_SIZE / 2;
    }

    // Clear filters
    for (int m = 0;
         m < NUM_MEL;
         m++)
    {
        for (int k = 0;
             k <= FFT_SIZE / 2;
             k++)
        {
            melFilters[m][k] = 0.0;
        }
    }

    // Create triangular filters
    for (int m = 0;
         m < NUM_MEL;
         m++)
    {
        int left =
            bin[m];

        int center =
            bin[m + 1];

        int right =
            bin[m + 2];

        // Rising edge
        for (int k = left;
             k < center;
             k++)
        {
            if (center != left)
            {
                melFilters[m][k] =
                    (float)(k - left) /
                    (float)(center - left);
            }
        }

        // Falling edge
        for (int k = center;
             k <= right;
             k++)
        {
            if (right != center)
            {
                melFilters[m][k] =
                    (float)(right - k) /
                    (float)(right - center);
            }
        }
    }

    Serial.println("Mel filters ready.");
}


// =====================================================
// FFT
// =====================================================

void FFT()
{
    // -------------------------------------------------
    // Bit reversal
    // -------------------------------------------------

    int j = 0;

    for (int i = 0;
         i < FFT_SIZE;
         i++)
    {
        if (i < j)
        {
            float temp =
                fftReal[i];

            fftReal[i] =
                fftReal[j];

            fftReal[j] =
                temp;

            temp =
                fftImag[i];

            fftImag[i] =
                fftImag[j];

            fftImag[j] =
                temp;
        }

        int bit =
            FFT_SIZE >> 1;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;
    }

    // -------------------------------------------------
    // FFT stages
    // -------------------------------------------------

    for (int len = 2;
         len <= FFT_SIZE;
         len <<= 1)
    {
        float angle =
            -2.0 *
            PI /
            len;

        float wlenReal =
            cos(angle);

        float wlenImag =
            sin(angle);

        for (int i = 0;
             i < FFT_SIZE;
             i += len)
        {
            float wReal = 1.0;
            float wImag = 0.0;

            for (int j = 0;
                 j < len / 2;
                 j++)
            {
                int u =
                    i + j;

                int v =
                    i + j + len / 2;

                float vReal =
                    fftReal[v] *
                    wReal -
                    fftImag[v] *
                    wImag;

                float vImag =
                    fftReal[v] *
                    wImag +
                    fftImag[v] *
                    wReal;

                float uReal =
                    fftReal[u];

                float uImag =
                    fftImag[u];

                fftReal[u] =
                    uReal + vReal;

                fftImag[u] =
                    uImag + vImag;

                fftReal[v] =
                    uReal - vReal;

                fftImag[v] =
                    uImag - vImag;

                float nextWReal =
                    wReal * wlenReal -
                    wImag * wlenImag;

                float nextWImag =
                    wReal * wlenImag +
                    wImag * wlenReal;

                wReal =
                    nextWReal;

                wImag =
                    nextWImag;
            }
        }
    }
}


// =====================================================
// PROCESS ONE AUDIO FRAME
// =====================================================

void processFrame(
    int frameNumber
)
{
    int start =
        frameNumber * HOP_SIZE;

    // -------------------------------------------------
    // Copy + Hamming window
    // -------------------------------------------------

    for (int i = 0;
         i < FFT_SIZE;
         i++)
    {
        if (i < FRAME_SIZE)
        {
            // PCM16 -> normalized float

            float sample =
                (float)audioBuffer[
                    start + i
                ] / 32768.0;

            // Hamming window

            float window =
                0.54 -
                0.46 *
                cos(
                    2.0 *
                    PI *
                    i /
                    (FRAME_SIZE - 1)
                );

            fftReal[i] =
                sample * window;
        }
        else
        {
            // Zero padding
            fftReal[i] = 0.0;
        }

        fftImag[i] = 0.0;
    }

    // -------------------------------------------------
    // FFT
    // -------------------------------------------------

    FFT();

    // -------------------------------------------------
    // Power spectrum + Mel filtering
    // -------------------------------------------------

    for (int m = 0;
         m < NUM_MEL;
         m++)
    {
        float energy = 0.0;

        for (int k = 0;
             k <= FFT_SIZE / 2;
             k++)
        {
            float power =
                fftReal[k] *
                fftReal[k] +
                fftImag[k] *
                fftImag[k];

            energy +=
                power *
                melFilters[m][k];
        }

        // Avoid log(0)
        if (energy < 1e-10)
            energy = 1e-10;

        // Log-Mel feature
        melMatrix[frameNumber][m] =
            log(energy);
    }
}


// =====================================================
// RECORD 1 SECOND
// =====================================================

void recordOneSecond()
{
    Serial.println();
    Serial.println(
        "Recording 1 second..."
    );

    int samplesRecorded = 0;

    while (
        samplesRecorded <
        WINDOW_SAMPLES
    )
    {
        size_t bytesRead = 0;

        esp_err_t result =
            i2s_read(
                I2S_PORT,

                i2sBuffer,

                sizeof(i2sBuffer),

                &bytesRead,

                portMAX_DELAY
            );

        if (result != ESP_OK)
            continue;

        int count =
            bytesRead /
            sizeof(int32_t);

        for (int i = 0;
             i < count &&
             samplesRecorded <
             WINDOW_SAMPLES;
             i++)
        {
            int32_t raw =
                i2sBuffer[i];

            // Convert INMP441 32-bit slot
            // to PCM16

            int32_t sample =
                raw >> 16;

            // Saturate

            if (sample > 32767)
                sample = 32767;

            if (sample < -32768)
                sample = -32768;

            audioBuffer[
                samplesRecorded
            ] =
                (int16_t)sample;

            samplesRecorded++;
        }
    }

    Serial.println(
        "Recording complete."
    );
}


// =====================================================
// PROCESS 1 SECOND
// =====================================================

void processAudio()
{
    Serial.println(
        "Processing..."
    );

    unsigned long startTime =
        millis();

    for (int frame = 0;
         frame < NUM_FRAMES;
         frame++)
    {
        processFrame(frame);
    }

    unsigned long elapsed =
        millis() - startTime;

    Serial.print(
        "Processing time: "
    );

    Serial.print(elapsed);

    Serial.println(
        " ms"
    );
}


// =====================================================
// PRINT MATRIX
// =====================================================

void printMatrix()
{
    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "LOG-MEL MATRIX"
    );

    Serial.print(
        "Rows: "
    );

    Serial.println(
        NUM_FRAMES
    );

    Serial.print(
        "Columns: "
    );

    Serial.println(
        NUM_MEL
    );

    Serial.println(
        "========================================"
    );

    // Print each row

    for (int i = 0;
         i < NUM_FRAMES;
         i++)
    {
        Serial.print("[ ");

        for (int j = 0;
             j < NUM_MEL;
             j++)
        {
            Serial.print(
                melMatrix[i][j],
                3
            );

            if (j <
                NUM_MEL - 1)
            {
                Serial.print(", ");
            }
        }

        Serial.println(" ]");
    }

    Serial.println(
        "========================================"
    );
}


// =====================================================
// I2S SETUP
// =====================================================

void setupI2S()
{
    i2s_config_t config =
    {
        .mode = (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        ),

        .sample_rate =
            SAMPLE_RATE,

        .bits_per_sample =
            I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format =
            I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format =
            I2S_COMM_FORMAT_I2S,

        .intr_alloc_flags =
            ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 8,

        .dma_buf_len = 64,

        .use_apll = false,

        .tx_desc_auto_clear = false,

        .fixed_mclk = 0
    };

    i2s_pin_config_t pins =
    {
        .bck_io_num =
            I2S_BCLK,

        .ws_io_num =
            I2S_WS,

        .data_out_num =
            I2S_PIN_NO_CHANGE,

        .data_in_num =
            I2S_SD
    };

    if (
        i2s_driver_install(
            I2S_PORT,
            &config,
            0,
            NULL
        ) != ESP_OK
    )
    {
        Serial.println(
            "I2S driver failed"
        );

        while (1);
    }

    if (
        i2s_set_pin(
            I2S_PORT,
            &pins
        ) != ESP_OK
    )
    {
        Serial.println(
            "I2S pin setup failed"
        );

        while (1);
    }

    i2s_zero_dma_buffer(
        I2S_PORT
    );

    Serial.println(
        "I2S initialized."
    );
}


// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(921600);

    delay(1000);

    Serial.println();
    Serial.println(
        "ESP32 LOG-MEL FEATURE EXTRACTION"
    );

    Serial.println(
        "================================="
    );

    Serial.print(
        "Sample rate: "
    );

    Serial.println(
        SAMPLE_RATE
    );

    Serial.print(
        "Frame size: "
    );

    Serial.println(
        FRAME_SIZE
    );

    Serial.print(
        "Hop size: "
    );

    Serial.println(
        HOP_SIZE
    );

    Serial.print(
        "FFT size: "
    );

    Serial.println(
        FFT_SIZE
    );

    Serial.print(
        "Mel filters: "
    );

    Serial.println(
        NUM_MEL
    );

    Serial.print(
        "Matrix: "
    );

    Serial.print(
        NUM_FRAMES
    );

    Serial.print(" x ");

    Serial.println(
        NUM_MEL
    );

    setupI2S();

    createMelFilters();

    Serial.println(
        "Ready."
    );
}


// =====================================================
// LOOP
// =====================================================

void loop()
{
    // -----------------------------------------------
    // 1. Record one second
    // -----------------------------------------------

    recordOneSecond();

    // -----------------------------------------------
    // 2. Convert to Log-Mel matrix
    // -----------------------------------------------

    processAudio();

    // -----------------------------------------------
    // 3. Print matrix
    // -----------------------------------------------

    printMatrix();

    // Then immediately record next second
}