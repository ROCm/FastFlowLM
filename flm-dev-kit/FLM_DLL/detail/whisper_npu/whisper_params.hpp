// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

constexpr int WHISPER_MAX_LENGTH     = 448;
constexpr int WHISPER_D              = 1280;

constexpr int WHISPER_ENCODER_HEADS  = 20;
constexpr int WHISPER_ENCODER_FFN_D  = 5120;
constexpr int WHISPER_ENCODER_LAYERS = 32;
// Encoder frames per 30 s window. Buffers are padded to 1536 rows; rows from
// here on are padding and must not be attended to.
constexpr int WHISPER_AUDIO_FRAMES   = 1500;

constexpr int WHISPER_DECODER_HEADS  = 20;
constexpr int WHISPER_DECODER_FFN_D  = 5120;
constexpr int WHISPER_DECODER_LAYERS = 4;

constexpr int WHISPER_VOCAB_SIZE     = 51872; // 51866 zero padding to 51872

constexpr int WHISPER_INIT_TOKEN     = 50258;
