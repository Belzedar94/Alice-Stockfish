/*
  Alice-Stockfish training-data generator
  Copyright (C) 2026 The Alice-Stockfish developers

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef DATA_ALICE_V2_CHUNK_V1_H_INCLUDED
#define DATA_ALICE_V2_CHUNK_V1_H_INCLUDED

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

#include "sha256.h"
#include "training_data.h"

namespace Stockfish::Data {

inline constexpr std::size_t      AliceV2ChunkV1HeaderSize = 4096;
inline constexpr std::size_t      AliceV2ChunkV1RecordSize = 84;
inline constexpr std::string_view AliceV2ChunkV1SchemaName = "ALICE_V2_CHUNK_V1";
inline constexpr std::string_view AliceV2ChunkV1SchemaSha256 =
  "8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE";
inline constexpr std::string_view AliceV2RecordV1SchemaName = "ALICEV2_RECORD_V1";
inline constexpr std::string_view AliceV2RecordV1SchemaSha256 =
  "1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC";
inline constexpr std::string_view AliceV2TargetContractV1SchemaName = "ALICE_V2_TARGET_CONTRACT_V1";
inline constexpr std::string_view AliceV2TargetContractV1Sha256 =
  "FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD";
inline constexpr std::string_view AliceV2RulesId     = "alice-rules-v1";
inline constexpr std::string_view AliceV2FeatureId   = "AliceHalfKAv2_hm_Rel-v1";
inline constexpr u32              AliceV2RulesIdHash = 0x73F822ABU;
inline constexpr u32              AliceV2FeatureHash = 0x5280C41EU;

using AliceV2ChunkV1Record = std::array<u8, AliceV2ChunkV1RecordSize>;

struct AliceV2ChunkV1Manifest {
    std::string runConfigSha256;
    std::string sourceCommit;
    bool        sourceDirty = false;
    std::string networkSha256;
    std::string bookSha256;

    u64         requestedRecords = 0;
    u64         seed             = 0;
    u64         baseSeed         = 0;
    u64         totalRecords     = 0;
    u64         recordsPerChunk  = 0;
    u64         chunkIndex       = 0;
    u64         totalChunks      = 0;
    u64         trainChunks      = 0;
    u64         validationChunks = 0;
    u64         testChunks       = 0;
    std::string split;
    usize       searchThreads     = 1;
    usize       hashMb            = 0;
    int         depth             = 0;
    u64         nodes             = 0;
    int         randomMoveMinPly  = 0;
    int         randomMoveMaxPly  = 0;
    int         randomMoveCount   = 0;
    int         randomMultiPv     = 0;
    int         randomMultiPvDiff = 0;
    int         writeMinPly       = 0;
    int         writeMaxPly       = 0;
    int         maxGamePly        = 0;
    usize       openingCount      = 0;
};

std::string_view alice_v2_data_schema_json() noexcept;
DataResult       validate_alice_v2_chunk_v1_manifest(const AliceV2ChunkV1Manifest& manifest);
DataResult       encode_alice_v2_record_v1(const TrainingDataSample& sample,
                                           AliceV2ChunkV1Record&     record);

class AliceV2ChunkV1Sink final: public DatasetSink {
   public:
    AliceV2ChunkV1Sink(std::filesystem::path path, AliceV2ChunkV1Manifest manifest);
    ~AliceV2ChunkV1Sink() override;

    AliceV2ChunkV1Sink(const AliceV2ChunkV1Sink&)            = delete;
    AliceV2ChunkV1Sink& operator=(const AliceV2ChunkV1Sink&) = delete;

    DataResult append(const TrainingDataSample& sample) override;
    DataResult finalize() override;
    DataResult abort() override;

    u64 records_written() const noexcept { return recordsWritten; }

   private:
    DataResult open_exclusively();
    DataResult write_bytes(const u8* data, std::size_t size, std::string_view label);
    DataResult seek_to_start();
    DataResult sync_and_close();
    DataResult publish_final();
    void       cleanup_partial(int& closeError, int& removeError) noexcept;

    std::filesystem::path  outputPath;
    std::filesystem::path  temporaryPath;
    AliceV2ChunkV1Manifest manifest;
    Sha256                 payloadHasher;
    u64                    recordsWritten = 0;
    int                    fileDescriptor = -1;
    bool                   created        = false;
    bool                   accepting      = true;
    bool                   finalized      = false;
    bool                   aborted        = false;
};

}  // namespace Stockfish::Data

#endif  // DATA_ALICE_V2_CHUNK_V1_H_INCLUDED
