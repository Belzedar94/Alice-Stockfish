/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_V2_MANIFEST_H_INCLUDED
#define NNUE_ALICE_NATIVE_V2_MANIFEST_H_INCLUDED

#include <array>
#include <string_view>

#include "../../types.h"
#include "../alice_native/manifest.h"

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

inline constexpr std::string_view ArchitectureId = "AliceNative-v2-M512";
inline constexpr std::string_view PieceSquareFeatureId = "AliceHalfKAv2_hm_Rel-v1";
inline constexpr std::string_view RulesId          = "alice-rules-v1";
inline constexpr std::string_view TensorLayoutId   = "alice-native-compact-tensors-v1";
inline constexpr std::string_view QuantizationId =
  "alice-native-piece-i16-psqt-i32-dense-i8-v1";
inline constexpr std::string_view QuantizationContractSha256 =
  "561B9040F0FC33A36DC05EDC700C9F40130D0B467EF78E957BAA743BB2EEC14F";
inline constexpr std::string_view CheckpointAdapterId =
  "alice-native-checkpoint-v2-m512-expanded-v1";
inline constexpr std::string_view CheckpointAdapterSha256 =
  "5B925781C27B6DF05824BCC79D8CBD46685A130C1608ED3A645EC62A55A808B2";
// Filled from the canonical M512 manifest produced by the exporter.  The
// loader also authenticates the entire file against the mandatory user SHA.
inline constexpr std::string_view ManifestSha256 =
  "6A31720B2DF45AE9B96577E9DBCE767673DEBCADC1A357DAEE2053B864DCA3D4";
inline constexpr std::string_view ArchitectureSha256 =
  "59C11FD6144D08756614962A6F4E3DEF072C6184500946D5639F1F5FFCEAC2AA";
constexpr u32 CompositeArchitectureHash = 0x31B3B6D4u;
constexpr u32 CanonicalManifestBytes     = 1462;
constexpr u64 SerializedBytes            = 47722778;

inline constexpr std::array<u8, 8> WireMagic = {'A', 'L', 'N', 'N', 'V', '2', 0, 0};
constexpr u32                      WireVersion = 0xA11CE201u;
constexpr u64                      HeaderBytes = 32;

constexpr IndexType PieceSquareDimensions = AliceNative::PieceSquareDimensions;
constexpr IndexType TransformerWidth       = 512;
constexpr IndexType DenseInput             = 2 * TransformerWidth;
constexpr IndexType Hidden1                = 16;
constexpr IndexType Hidden2                = 32;
constexpr IndexType PsqtBuckets            = 8;
constexpr IndexType LayerStacks            = 8;
constexpr IndexType TensorCount             = 9;

constexpr u32 FeatureComponentHash = 0x5280C01Eu;
constexpr u32 DenseComponentHash   = 0x633376CAu;

constexpr u64 FtBiasElements            = TransformerWidth;
constexpr u64 PieceSquareWeightElements = u64(PieceSquareDimensions) * TransformerWidth;
constexpr u64 PieceSquarePsqtElements   = u64(PieceSquareDimensions) * PsqtBuckets;
constexpr u64 Fc0BiasElementsPerStack   = Hidden1;
constexpr u64 Fc0WeightElementsPerStack = u64(Hidden1) * DenseInput;
constexpr u64 Fc1BiasElementsPerStack   = Hidden2;
// The logical FC1 input is Hidden1.  The last Hidden1 columns are sealed zero
// padding so the wire rows retain the engine's 32-byte SIMD stride.
constexpr u64 Fc1WeightElementsPerStack = u64(Hidden2) * Hidden2;
constexpr u64 Fc2BiasElementsPerStack   = 1;
constexpr u64 Fc2WeightElementsPerStack = Hidden2;

constexpr u64 FeatureTensorBytes =
  4 + FtBiasElements * 2 + PieceSquareWeightElements * 2 + PieceSquarePsqtElements * 4;
constexpr u64 DenseStackTensorBytes =
  4 + Fc0BiasElementsPerStack * 4 + Fc0WeightElementsPerStack
  + Fc1BiasElementsPerStack * 4 + Fc1WeightElementsPerStack
  + Fc2BiasElementsPerStack * 4 + Fc2WeightElementsPerStack;
constexpr u64 PayloadBytes = FeatureTensorBytes + u64(LayerStacks) * DenseStackTensorBytes;

// Fixed-point scales.  Rounding is performed only by the exporter; inference
// consumes the sealed integers with the explicit shifts below.
constexpr i32 FeatureScale     = 127;
constexpr i32 PsqtScale        = 9600;
constexpr i32 HiddenWeightScale = 64;
constexpr i32 HiddenBiasScale   = FeatureScale * HiddenWeightScale;
constexpr i32 HiddenShift       = 6;
constexpr i32 OutputScale       = 9600;
constexpr i32 EngineOutputDivisor = 16;

static_assert(PieceSquareDimensions == 45056);
static_assert(HiddenBiasScale == 8128);
static_assert(FeatureTensorBytes == 47580164);
static_assert(DenseStackTensorBytes == 17640);
static_assert(PayloadBytes == 47721284);

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2

#endif  // NNUE_ALICE_NATIVE_V2_MANIFEST_H_INCLUDED
