/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_MANIFEST_H_INCLUDED
#define NNUE_ALICE_NATIVE_MANIFEST_H_INCLUDED

#include <string_view>

#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::AliceNative {

inline constexpr std::string_view ArchitectureId       = "AliceNative-v1";
inline constexpr std::string_view PieceSquareFeatureId = "AliceHalfKAv2_hm_Rel-v1";
inline constexpr std::string_view ThreatFeatureId      = "AliceFullThreats_Rel-v1";
inline constexpr std::string_view PairFeatureId        = "None";
inline constexpr std::string_view RulesId              = "alice-rules-v1";
inline constexpr std::string_view TensorLayoutId       = "alice-native-tensors-v1";
inline constexpr std::string_view QuantizationId       = "alice-native-quant-v1";

constexpr u32 WireVersion = 0xA11CE001u;

constexpr IndexType PiecePlanes           = 11;
constexpr IndexType RelationCount         = BOARD_NB;
constexpr IndexType KingBucketCount       = 32;
constexpr IndexType PiecePlaneStride      = SQUARE_NB;
constexpr IndexType RelationStride        = PiecePlanes * PiecePlaneStride;
constexpr IndexType KingBucketStride      = RelationCount * RelationStride;
constexpr IndexType PieceSquareDimensions = KingBucketCount * KingBucketStride;

constexpr IndexType BaseThreatDimensions   = 59808;
constexpr IndexType ThreatDimensions       = RelationCount * BaseThreatDimensions;
constexpr IndexType LogicalInputDimensions = PieceSquareDimensions + ThreatDimensions;

constexpr IndexType L1          = 1024;
constexpr IndexType L2          = 32;
constexpr IndexType L3          = 32;
constexpr IndexType PsqtBuckets = 8;
constexpr IndexType LayerStacks = 8;

constexpr u32 PieceSquareHash       = 0x5280C41Eu;
constexpr u32 ThreatHash            = 0x6EE7B82Cu;
constexpr u32 DenseArchitectureHash = 0x63337116u;

constexpr u32 rotate_left_one(u32 value) { return (value << 1) | (value >> 31); }

// Native v1 intentionally has no pair feature. The component order is threats,
// then piece-square features.
constexpr u32 FeatureTransformerHash    = rotate_left_one(ThreatHash) ^ PieceSquareHash ^ (L1 * 2);
constexpr u32 CompositeArchitectureHash = FeatureTransformerHash ^ DenseArchitectureHash;

static_assert(RelationStride == 704);
static_assert(KingBucketStride == 1408);
static_assert(PieceSquareDimensions == 45056);
static_assert(ThreatDimensions == 119616);
static_assert(LogicalInputDimensions == 164672);
static_assert(FeatureTransformerHash == 0x8F4FBC46u);
static_assert(CompositeArchitectureHash == 0xEC7CCD50u);

}  // namespace Stockfish::Eval::NNUE::AliceNative

#endif  // NNUE_ALICE_NATIVE_MANIFEST_H_INCLUDED
