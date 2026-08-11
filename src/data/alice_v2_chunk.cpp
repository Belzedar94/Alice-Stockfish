/*
  Alice-Stockfish training-data generator
  Copyright (C) 2026 The Alice-Stockfish developers

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "alice_v2_chunk.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include "movegen.h"
#include "position.h"

#ifdef _WIN32
    #include <fcntl.h>
    #include <io.h>
    #include <process.h>
    #include <sys/stat.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace Stockfish::Data {
namespace {

constexpr std::array<u8, 8> FileMagic     = {'A', 'L', 'C', 'H', 'N', 'K', '1', '\0'};
constexpr u16               FormatVersion = 1;
constexpr std::string_view  CapabilityJson =
  R"({"schema":"ALICE_V2_CHUNK_V1","schema_sha256":"8D762FE64B0B8BB196BAFACA26C3514CB338B29BEF2B4AFB268F1974A2E0C6AE","record_schema":{"schema":"ALICEV2_RECORD_V1","schema_sha256":"1FCDD61BD11FC5428C3A8726FD21F853A6446BD6A7C3025DC49A34C936CE85AC"},"target_contract":{"schema":"ALICE_V2_TARGET_CONTRACT_V1","sha256":"FFFE33048B9E3FBB91E378F6208CAB0FE1CADB06C6AF29A7493C0496F4E296CD"},"write":true,"record_size":84,"header_size":4096})";

bool little_endian_host() {
    constexpr u16 value = 1;
    return *reinterpret_cast<const u8*>(&value) == 1;
}

std::string system_error_message(int error) {
    return std::error_code(error, std::generic_category()).message();
}

void write_u16_le(u8* destination, u16 value) {
    destination[0] = u8(value & 0xFF);
    destination[1] = u8(value >> 8);
}

void write_u32_le(u8* destination, u32 value) {
    destination[0] = u8(value & 0xFF);
    destination[1] = u8((value >> 8) & 0xFF);
    destination[2] = u8((value >> 16) & 0xFF);
    destination[3] = u8(value >> 24);
}

u32 crc32(const u8* data, std::size_t size) {
    u32 value = 0xFFFFFFFFU;
    for (std::size_t i = 0; i < size; ++i)
    {
        value ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            value = (value >> 1) ^ (0xEDB88320U & (0U - (value & 1U)));
    }
    return value ^ 0xFFFFFFFFU;
}

bool is_hex_commit(std::string_view text) {
    return text.size() == 40 && std::all_of(text.begin(), text.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

u8 piece_code(Piece piece) {
    const u8 code = u8(piece);
    return piece == NO_PIECE || (code >= 1 && code <= 6) || (code >= 9 && code <= 14) ? code : 0xFF;
}

u32 encode_move(const Position& position, Move move) {
    u32 promotion = 0;
    if (move.type_of() == PROMOTION)
        switch (move.promotion_type())
        {
        case QUEEN :
            promotion = 1;
            break;
        case ROOK :
            promotion = 2;
            break;
        case BISHOP :
            promotion = 3;
            break;
        case KNIGHT :
            promotion = 4;
            break;
        default :
            return 0;
        }

    return u32(move.from_sq()) | (u32(move.to_sq()) << 6) | (promotion << 12)
         | (u32(position.board_of(move.from_sq()) == BOARD_B) << 15)
         | (u32(move.type_of() == CASTLING) << 16);
}

std::string
manifest_json(const AliceV2ChunkV1Manifest& manifest, u64 records, std::string_view payloadSha256) {
    std::ostringstream json;
    json << "{\"schema\":\"" << AliceV2ChunkV1SchemaName << "\",\"schema_sha256\":\""
         << AliceV2ChunkV1SchemaSha256
         << "\",\"format_version\":1,\"header_bytes\":" << AliceV2ChunkV1HeaderSize
         << ",\"record_bytes\":" << AliceV2ChunkV1RecordSize << ",\"record_count\":" << records
         << ",\"byte_order\":\"little\",\"run_config_sha256\":\"" << manifest.runConfigSha256
         << "\",\"source_commit\":\"" << manifest.sourceCommit
         << "\",\"source_dirty\":" << json_bool(manifest.sourceDirty)
         << ",\"network\":{\"schema\":\"ALICE_RUN2RL_LEGACY_V1\",\"sha256\":\""
         << manifest.networkSha256 << "\"},\"book_sha256\":\"" << manifest.bookSha256
         << "\",\"payload_sha256\":\"" << payloadSha256 << "\",\"record_schema\":{\"schema\":\""
         << AliceV2RecordV1SchemaName << "\",\"schema_sha256\":\"" << AliceV2RecordV1SchemaSha256
         << "\",\"rules_id\":\"" << AliceV2RulesId << "\",\"rules_hash\":" << AliceV2RulesIdHash
         << ",\"feature_id\":\"" << AliceV2FeatureId << "\",\"feature_hash\":" << AliceV2FeatureHash
         << "},\"target_contract\":{\"schema\":\"" << AliceV2TargetContractV1SchemaName
         << "\",\"sha256\":\"" << AliceV2TargetContractV1Sha256
         << "\"},\"partition\":{\"base_seed\":\"" << manifest.baseSeed
         << "\",\"total_records\":" << manifest.totalRecords
         << ",\"records_per_chunk\":" << manifest.recordsPerChunk
         << ",\"chunk_index\":" << manifest.chunkIndex
         << ",\"total_chunks\":" << manifest.totalChunks << ",\"split\":\"" << manifest.split
         << "\",\"train_chunks\":" << manifest.trainChunks
         << ",\"validation_chunks\":" << manifest.validationChunks
         << ",\"test_chunks\":" << manifest.testChunks
         << "},\"generation\":{\"requested_records\":" << manifest.requestedRecords
         << ",\"seed\":\"" << manifest.seed << "\",\"search_threads\":" << manifest.searchThreads
         << ",\"hash_mb\":" << manifest.hashMb << ",\"depth\":" << manifest.depth
         << ",\"nodes\":" << manifest.nodes
         << ",\"random_move_min_ply\":" << manifest.randomMoveMinPly
         << ",\"random_move_max_ply\":" << manifest.randomMoveMaxPly
         << ",\"random_move_count\":" << manifest.randomMoveCount
         << ",\"random_multi_pv\":" << manifest.randomMultiPv
         << ",\"random_multi_pv_diff\":" << manifest.randomMultiPvDiff
         << ",\"write_min_ply\":" << manifest.writeMinPly
         << ",\"write_max_ply\":" << manifest.writeMaxPly
         << ",\"max_game_ply\":" << manifest.maxGamePly
         << ",\"opening_count\":" << manifest.openingCount << "}}";
    return json.str();
}

}  // namespace

std::string_view alice_v2_data_schema_json() noexcept { return CapabilityJson; }

DataResult validate_alice_v2_chunk_v1_manifest(const AliceV2ChunkV1Manifest& manifest) {
    if (!is_upper_sha256(manifest.runConfigSha256))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Run-config SHA-256 must be uppercase hexadecimal");
    if (!is_hex_commit(manifest.sourceCommit)
        || std::all_of(manifest.sourceCommit.begin(), manifest.sourceCommit.end(),
                       [](char c) { return c == '0'; }))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Source commit must be a nonzero full 40-digit Git object ID");
    if (manifest.sourceDirty)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "ALICE_V2_CHUNK_V1 generation requires a clean source tree");
    if (!is_upper_sha256(manifest.networkSha256)
        || manifest.networkSha256
             != "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9")
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Network SHA-256 must identify alice_run2rl_e40_l09.nnue");
    if (manifest.bookSha256 != "NONE" && !is_upper_sha256(manifest.bookSha256))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Book SHA-256 must be uppercase hexadecimal or NONE");
    if (!manifest.requestedRecords || !manifest.seed || !manifest.totalRecords
        || !manifest.recordsPerChunk || !manifest.totalChunks || manifest.searchThreads != 1
        || !manifest.hashMb || manifest.depth <= 0 || manifest.depth >= MAX_PLY
        || manifest.randomMoveMinPly < 0 || manifest.randomMoveMaxPly < manifest.randomMoveMinPly
        || manifest.randomMoveMaxPly >= manifest.maxGamePly || manifest.randomMoveCount < 0
        || manifest.randomMoveCount > manifest.maxGamePly || manifest.randomMultiPv < 0
        || manifest.randomMultiPv > int(MAX_MOVES) || manifest.randomMultiPvDiff < 0
        || manifest.writeMinPly < 0 || manifest.writeMaxPly <= manifest.writeMinPly
        || manifest.maxGamePly <= manifest.writeMaxPly || !manifest.openingCount)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Generation settings are outside the ALICE_V2_CHUNK_V1 domain");

    if (manifest.totalChunks != 1 + (manifest.totalRecords - 1) / manifest.recordsPerChunk
        || manifest.chunkIndex >= manifest.totalChunks
        || manifest.baseSeed > std::numeric_limits<u64>::max() - manifest.chunkIndex
        || manifest.seed != manifest.baseSeed + manifest.chunkIndex
        || manifest.chunkIndex > std::numeric_limits<u64>::max() / manifest.recordsPerChunk)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Chunk identity is inconsistent with the preregistered run");

    const u64 offset          = manifest.chunkIndex * manifest.recordsPerChunk;
    const u64 expectedRecords = std::min(manifest.recordsPerChunk, manifest.totalRecords - offset);
    if (manifest.requestedRecords != expectedRecords
        || manifest.trainChunks + manifest.validationChunks + manifest.testChunks
             != manifest.totalChunks)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Chunk count or split sizes disagree with the run partition");

    const std::string_view expectedSplit =
      manifest.chunkIndex < manifest.trainChunks
        ? "train"
        : (manifest.chunkIndex < manifest.trainChunks + manifest.validationChunks ? "validation"
                                                                                  : "test");
    if (manifest.split != expectedSplit)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Chunk split disagrees with its preregistered index range");

    return DataResult::success();
}

DataResult encode_alice_v2_record_v1(const TrainingDataSample& sample,
                                     AliceV2ChunkV1Record&     record) {
    record.fill(0);
    if (!little_endian_host())
        return DataResult::failure(DataError::UNSUPPORTED_BYTE_ORDER,
                                   "ALICE_V2_CHUNK_V1 generation requires a little-endian host");
    if (sample.score < std::numeric_limits<i16>::min()
        || sample.score > std::numeric_limits<i16>::max())
        return DataResult::failure(DataError::SCORE_OUT_OF_RANGE,
                                   "Training score does not fit ALICE_V2_CHUNK_V1 int16");
    if (sample.result < -1 || sample.result > 1)
        return DataResult::failure(DataError::RESULT_OUT_OF_RANGE,
                                   "Training result must be -1, 0, or 1");
    Position  position;
    StateInfo state{};
    if (const auto setError = position.set(sample.fen, false, &state))
        return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                   std::string("Cannot encode Alice FEN: ") + setError->what());
    if (position.is_chess960() || position.count<KING>(WHITE) != 1
        || position.count<KING>(BLACK) != 1 || popcount(position.pieces()) < 2
        || popcount(position.pieces()) > 32 || position.ep_square() != SQ_NONE)
        return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                   "Position violates ALICE_V2_CHUNK_V1 physical invariants");
    if (position.rule50_count() < 0 || position.rule50_count() > std::numeric_limits<u16>::max()
        || position.game_ply() < 0 || position.game_ply() > std::numeric_limits<u16>::max())
        return DataResult::failure(DataError::POSITION_CLOCK_OUT_OF_RANGE,
                                   "Position clocks do not fit ALICE_V2_CHUNK_V1 uint16 fields");
    const MoveList<LEGAL> legalMoves(position);
    if (!sample.bestMove.is_ok() || !legalMoves.contains(sample.bestMove))
        return DataResult::failure(DataError::INVALID_MOVE, "Stored best move must be legal");

    for (int square = 0; square < SQUARE_NB; ++square)
    {
        const u8 code = piece_code(position.piece_on(Square(square)));
        if (code == 0xFF)
        {
            record.fill(0);
            return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                       "Position contains an unregistered physical piece code");
        }
        if (code)
            record[usize(square)] =
              code | (position.board_of(Square(square)) == BOARD_B ? 0x10 : 0);
    }

    write_u32_le(record.data() + 64, encode_move(position, sample.bestMove));
    write_u16_le(record.data() + 68, u16(i16(sample.score)));
    write_u16_le(record.data() + 70, u16(position.game_ply()));
    record[72] = u8(i8(sample.result));
    record[73] = u8(position.side_to_move());
    record[74] =
      u8((position.can_castle(WHITE_OO) ? 1 : 0) | (position.can_castle(WHITE_OOO) ? 2 : 0)
         | (position.can_castle(BLACK_OO) ? 4 : 0) | (position.can_castle(BLACK_OOO) ? 8 : 0));
    record[75] = 1;
    write_u16_le(record.data() + 76, u16(position.rule50_count()));
    write_u16_le(record.data() + 78, u16(position.game_ply() / 2 + 1));
    write_u32_le(record.data() + 80, crc32(record.data(), 80));
    return DataResult::success();
}

AliceV2ChunkV1Sink::AliceV2ChunkV1Sink(std::filesystem::path  path,
                                       AliceV2ChunkV1Manifest manifest_) :
    outputPath(std::move(path)),
    manifest(std::move(manifest_)) {}

AliceV2ChunkV1Sink::~AliceV2ChunkV1Sink() {
    if (!finalized && !aborted)
    {
        int closeError  = 0;
        int removeError = 0;
        cleanup_partial(closeError, removeError);
    }
}

DataResult AliceV2ChunkV1Sink::open_exclusively() {
    if (fileDescriptor != -1)
        return DataResult::success();
    if (DataResult valid = validate_alice_v2_chunk_v1_manifest(manifest); !valid)
        return valid;

    std::error_code inspectError;
    if (std::filesystem::exists(outputPath, inspectError) || inspectError)
        return DataResult::failure(
          inspectError ? DataError::OPEN_FAILED : DataError::OUTPUT_EXISTS,
          inspectError ? "Cannot inspect ALICE_V2_CHUNK_V1 output path: " + inspectError.message()
                       : "Cannot replace an existing ALICE_V2_CHUNK_V1 output");

#ifdef _WIN32
    const auto processId = ::_getpid();
#else
    const auto processId = ::getpid();
#endif
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    temporaryPath    = outputPath;
    temporaryPath += ".partial." + std::to_string(processId) + "." + std::to_string(nonce);

    errno = 0;
#ifdef _WIN32
    fileDescriptor =
      ::_wopen(temporaryPath.c_str(), _O_BINARY | _O_WRONLY | _O_CREAT | _O_EXCL | _O_NOINHERIT,
               _S_IREAD | _S_IWRITE);
#else
    int flags = O_WRONLY | O_CREAT | O_EXCL;
    #ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
    #endif
    fileDescriptor = ::open(temporaryPath.c_str(), flags, 0666);
#endif
    if (fileDescriptor == -1)
    {
        const int error = errno;
        return DataResult::failure(error == EEXIST ? DataError::OUTPUT_EXISTS
                                                   : DataError::OPEN_FAILED,
                                   "Cannot create ALICE_V2_CHUNK_V1 temporary output exclusively: "
                                     + system_error_message(error));
    }

    created = true;
    std::array<u8, AliceV2ChunkV1HeaderSize> placeholder{};
    if (DataResult result =
          write_bytes(placeholder.data(), placeholder.size(), "header placeholder");
        !result)
        return result;
    return DataResult::success();
}

DataResult
AliceV2ChunkV1Sink::write_bytes(const u8* data, std::size_t size, std::string_view label) {
    std::size_t written = 0;
    while (written < size)
    {
        errno = 0;
#ifdef _WIN32
        const int count = ::_write(fileDescriptor, data + written, unsigned(size - written));
#else
        const ssize_t count = ::write(fileDescriptor, data + written, size - written);
#endif
        if (count > 0)
        {
            written += std::size_t(count);
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        const int error = count == 0 ? EIO : errno;
        return DataResult::failure(DataError::WRITE_FAILED, "Cannot write ALICE_V2_CHUNK_V1 "
                                                              + std::string(label) + ": "
                                                              + system_error_message(error));
    }
    return DataResult::success();
}

DataResult AliceV2ChunkV1Sink::append(const TrainingDataSample& sample) {
    if (!accepting)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot write to a closed ALICE_V2_CHUNK_V1 sink");
    if (recordsWritten >= manifest.requestedRecords)
        return DataResult::failure(DataError::RECORD_COUNT_MISMATCH,
                                   "ALICE_V2_CHUNK_V1 output exceeded its declared record count");

    AliceV2ChunkV1Record record{};
    if (DataResult encoded = encode_alice_v2_record_v1(sample, record); !encoded)
        return encoded;
    if (DataResult opened = open_exclusively(); !opened)
        return opened;
    if (DataResult written = write_bytes(record.data(), record.size(), "record"); !written)
    {
        accepting = false;
        return written;
    }

    payloadHasher.update(record.data(), record.size());
    ++recordsWritten;
    return DataResult::success();
}

DataResult AliceV2ChunkV1Sink::seek_to_start() {
    errno = 0;
#ifdef _WIN32
    const auto offset = ::_lseeki64(fileDescriptor, 0, SEEK_SET);
#else
    const auto offset = ::lseek(fileDescriptor, 0, SEEK_SET);
#endif
    if (offset == 0)
        return DataResult::success();
    return DataResult::failure(DataError::SEEK_FAILED, "Cannot seek to ALICE_V2_CHUNK_V1 header: "
                                                         + system_error_message(errno));
}

DataResult AliceV2ChunkV1Sink::sync_and_close() {
    errno = 0;
#ifdef _WIN32
    if (::_commit(fileDescriptor) != 0)
#else
    if (::fsync(fileDescriptor) != 0)
#endif
        return DataResult::failure(DataError::CLOSE_FAILED,
                                   "Cannot synchronize ALICE_V2_CHUNK_V1 output: "
                                     + system_error_message(errno));

    const int descriptor = fileDescriptor;
    fileDescriptor       = -1;
    errno                = 0;
#ifdef _WIN32
    const int result = ::_close(descriptor);
#else
    const int result = ::close(descriptor);
#endif
    if (result != 0)
        return DataResult::failure(DataError::CLOSE_FAILED,
                                   "Cannot close ALICE_V2_CHUNK_V1 output: "
                                     + system_error_message(errno));
    return publish_final();
}

DataResult AliceV2ChunkV1Sink::publish_final() {
#ifdef _WIN32
    std::error_code publishError;
    std::filesystem::rename(temporaryPath, outputPath, publishError);
    if (publishError)
    {
        std::error_code inspectError;
        const bool      outputExists = std::filesystem::exists(outputPath, inspectError);
        return DataResult::failure(
          !inspectError && outputExists ? DataError::OUTPUT_EXISTS : DataError::CLOSE_FAILED,
          "Cannot atomically publish ALICE_V2_CHUNK_V1 output: " + publishError.message()
            + (inspectError ? "; final-path inspection also failed: " + inspectError.message()
                            : ""));
    }
#else
    errno = 0;
    if (::link(temporaryPath.c_str(), outputPath.c_str()) != 0)
    {
        const int error = errno;
        return DataResult::failure(
          error == EEXIST ? DataError::OUTPUT_EXISTS : DataError::CLOSE_FAILED,
          "Cannot atomically publish ALICE_V2_CHUNK_V1 output: " + system_error_message(error));
    }
    // The final pathname is already an immutable hard link to the synchronized
    // inode. A failed unlink can only leave a harmless orphaned temporary name.
    ::unlink(temporaryPath.c_str());
#endif
    created   = false;
    finalized = true;
    return DataResult::success();
}

DataResult AliceV2ChunkV1Sink::finalize() {
    if (finalized)
        return DataResult::success();
    if (aborted || !accepting)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot finalize an aborted or failed ALICE_V2_CHUNK_V1 sink");
    accepting = false;
    if (!recordsWritten || fileDescriptor == -1)
        return DataResult::failure(DataError::EMPTY_DATASET,
                                   "ALICE_V2_CHUNK_V1 forbids an empty dataset");
    if (recordsWritten != manifest.requestedRecords)
        return DataResult::failure(DataError::RECORD_COUNT_MISMATCH,
                                   "ALICE_V2_CHUNK_V1 record count does not match the request");

    const std::string payloadSha = sha256_hex_upper(payloadHasher.digest());
    const std::string json       = manifest_json(manifest, recordsWritten, payloadSha);
    if (json.size() > AliceV2ChunkV1HeaderSize - 48)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "ALICE_V2_CHUNK_V1 manifest exceeds the fixed header");

    std::array<u8, AliceV2ChunkV1HeaderSize> header{};
    std::copy(FileMagic.begin(), FileMagic.end(), header.begin());
    write_u16_le(header.data() + 8, FormatVersion);
    write_u16_le(header.data() + 10, u16(AliceV2ChunkV1HeaderSize));
    write_u32_le(header.data() + 12, u32(json.size()));
    std::copy(json.begin(), json.end(), header.begin() + 16);
    Sha256 headerHasher;
    headerHasher.update(header.data(), AliceV2ChunkV1HeaderSize - 32);
    const Sha256Digest headerDigest = headerHasher.digest();
    std::copy(headerDigest.begin(), headerDigest.end(), header.end() - 32);

    if (DataResult seek = seek_to_start(); !seek)
        return seek;
    if (DataResult written = write_bytes(header.data(), header.size(), "header"); !written)
        return written;
    return sync_and_close();
}

void AliceV2ChunkV1Sink::cleanup_partial(int& closeError, int& removeError) noexcept {
    closeError  = 0;
    removeError = 0;
    accepting   = false;
    if (fileDescriptor != -1)
    {
        const int descriptor = fileDescriptor;
        fileDescriptor       = -1;
        errno                = 0;
#ifdef _WIN32
        if (::_close(descriptor) != 0)
#else
        if (::close(descriptor) != 0)
#endif
            closeError = errno;
    }

    if (created && !temporaryPath.empty())
    {
        errno = 0;
#ifdef _WIN32
        if (::_wunlink(temporaryPath.c_str()) != 0 && errno != ENOENT)
#else
        if (::unlink(temporaryPath.c_str()) != 0 && errno != ENOENT)
#endif
            removeError = errno;
        else
            created = false;
    }
    aborted = !created;
}

DataResult AliceV2ChunkV1Sink::abort() {
    if (aborted)
        return DataResult::success();
    if (finalized)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot abort a finalized ALICE_V2_CHUNK_V1 sink");

    int closeError  = 0;
    int removeError = 0;
    cleanup_partial(closeError, removeError);
    if (closeError || removeError)
        return DataResult::failure(
          DataError::ABORT_FAILED,
          "Cannot fully remove partial ALICE_V2_CHUNK_V1 output: "
            + system_error_message(removeError ? removeError : closeError));
    return DataResult::success();
}

}  // namespace Stockfish::Data
