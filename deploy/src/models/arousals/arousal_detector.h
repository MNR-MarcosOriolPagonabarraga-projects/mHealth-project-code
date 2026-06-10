#pragma once
#include <stdint.h>

void arousal_detector_entry(const float tensor_temporal_input[1][2][1500], const float tensor_context_input[1][10][149], float tensor_arousal_event_logits[1]);