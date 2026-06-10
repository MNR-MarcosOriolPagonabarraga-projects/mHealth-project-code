#pragma once
#include <stdint.h>

void sleep_phase_entry(const float tensor_input_features[1][120][30], float tensor_sleep_stage_logits[1][4]);