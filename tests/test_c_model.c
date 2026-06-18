#include <stdio.h>
#include <stdlib.h>
#include "../deploy/models/sleep_stage/sleep_stage_detector.h"
#include "../deploy/models/arousals/arousal_detector.h"

int main() {
    float sleep_input_tensor[1][20][60];
    float sleep_output_logits[1][4] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    FILE *file = fopen("../outputs/dummy_sleep_input.bin", "rb");
    if (!file) {
        printf("Error: Could not open dummy_sleep_input.bin\n");
        return 1;
    }
    
    fread(sleep_input_tensor, sizeof(float), 1200, file);
    fclose(file);

    sleep_stage_entry(sleep_input_tensor, sleep_output_logits);

    printf("--- Compiled C Model Logits ---\n");
    printf("Wake:        %.6f\n", sleep_output_logits[0][0]);
    printf("Light Sleep: %.6f\n", sleep_output_logits[0][1]);
    printf("Deep Sleep:  %.6f\n", sleep_output_logits[0][2]);
    printf("REM:         %.6f\n", sleep_output_logits[0][3]);

    // --- ADD THIS: Save the logits for plotting ---
    FILE *out_file = fopen("../outputs/c_logits.bin", "wb");
    if (out_file) {
        fwrite(sleep_output_logits[0], sizeof(float), 4, out_file);
        fclose(out_file);
    } else {
        printf("Error: Could not write c_logits.bin\n");
    }

    return 0;
}