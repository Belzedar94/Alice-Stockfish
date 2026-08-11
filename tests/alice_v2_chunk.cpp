/*
  Contract tests for ALICE_V2_CHUNK_V1.
*/

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "data/alice_v2_chunk.h"
#include "movegen.h"
#include "position.h"
#include "uci.h"

using namespace Stockfish;
using namespace Stockfish::Data;

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

AliceV2ChunkV1Manifest valid_manifest() {
    AliceV2ChunkV1Manifest manifest;
    manifest.runConfigSha256   = std::string(64, 'B');
    manifest.sourceCommit      = "1234567890abcdef1234567890abcdef12345678";
    manifest.networkSha256     = "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9";
    manifest.bookSha256        = "NONE";
    manifest.requestedRecords  = 1;
    manifest.seed              = 1;
    manifest.baseSeed          = 1;
    manifest.totalRecords      = 1;
    manifest.recordsPerChunk   = 1;
    manifest.chunkIndex        = 0;
    manifest.totalChunks       = 1;
    manifest.trainChunks       = 1;
    manifest.validationChunks  = 0;
    manifest.testChunks        = 0;
    manifest.split             = "train";
    manifest.searchThreads     = 1;
    manifest.hashMb            = 16;
    manifest.depth             = 1;
    manifest.randomMoveMinPly  = 1;
    manifest.randomMoveMaxPly  = 2;
    manifest.randomMoveCount   = 1;
    manifest.randomMultiPv     = 1;
    manifest.randomMultiPvDiff = 200;
    manifest.writeMinPly       = 0;
    manifest.writeMaxPly       = 2;
    manifest.maxGamePly        = 4;
    manifest.openingCount      = 1;
    return manifest;
}

}  // namespace

int main() {
    if (alice_v2_data_schema_json().find(std::string(AliceV2ChunkV1SchemaSha256))
        == std::string_view::npos)
        return fail("capability JSON does not expose the sealed schema SHA-256");

    auto dirty        = valid_manifest();
    dirty.sourceDirty = true;
    if (validate_alice_v2_chunk_v1_manifest(dirty))
        return fail("dirty source manifest was accepted");

    Position  position;
    StateInfo state{};
    if (position.set(StartFEN, false, &state))
        return fail("cannot construct the Alice start position");
    const MoveList<LEGAL> legalMoves(position);
    if (legalMoves.size() == 0)
        return fail("Alice start position has no legal moves");

    TrainingDataSample sample;
    sample.fen      = position.fen();
    sample.score    = 17;
    sample.bestMove = *legalMoves.begin();
    sample.result   = 0;

    AliceV2ChunkV1Record record{};
    if (const DataResult encoded = encode_alice_v2_record_v1(sample, record); !encoded)
        return fail(encoded.message);
    if (record[75] != 1)
        return fail("encoded record does not set HAS_MOVE");

    Position  castlingPosition;
    StateInfo castlingState{};
    if (castlingPosition.set("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false, &castlingState))
        return fail("cannot construct the castling fixture");
    Move castlingMove = Move::none();
    for (const Move move : MoveList<LEGAL>(castlingPosition))
        if (move.type_of() == CASTLING && move.to_sq() == SQ_H1)
            castlingMove = move;
    if (castlingMove == Move::none())
        return fail("castling fixture has no legal king-side castle");
    TrainingDataSample castlingSample;
    castlingSample.fen      = castlingPosition.fen();
    castlingSample.score    = 8;
    castlingSample.bestMove = castlingMove;
    castlingSample.result   = 1;
    AliceV2ChunkV1Record castlingRecord{};
    if (const DataResult encoded = encode_alice_v2_record_v1(castlingSample, castlingRecord);
        !encoded)
        return fail(encoded.message);
    const std::uint32_t storedCastlingMove =
      std::uint32_t(castlingRecord[64]) | (std::uint32_t(castlingRecord[65]) << 8)
      | (std::uint32_t(castlingRecord[66]) << 16) | (std::uint32_t(castlingRecord[67]) << 24);
    if (((storedCastlingMove >> 6) & 0x3F) != SQ_H1 || !(storedCastlingMove & (1U << 16)))
        return fail("castling record did not preserve the internal rook-origin target");

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path  = std::filesystem::temp_directory_path()
                    / ("alice-v2-chunk-test-" + std::to_string(nonce) + ".bin");
    std::error_code removeError;
    std::filesystem::remove(path, removeError);

    {
        AliceV2ChunkV1Sink sink(path, valid_manifest());
        if (const DataResult appended = sink.append(sample); !appended)
            return fail(appended.message);
        if (const DataResult finalized = sink.finalize(); !finalized)
            return fail(finalized.message);
    }

    if (std::filesystem::file_size(path) != AliceV2ChunkV1HeaderSize + AliceV2ChunkV1RecordSize)
        return fail("final file size differs from the sealed layout");
    std::array<char, 8> magic{};
    {
        std::ifstream input(path, std::ios::binary);
        input.read(magic.data(), magic.size());
        if (!input)
            return fail("cannot read finalized header");
    }
    if (std::string(magic.data(), magic.size()) != std::string("ALCHNK1\0", 8))
        return fail("finalized header magic mismatch");

    {
        AliceV2ChunkV1Sink duplicate(path, valid_manifest());
        const DataResult   appended = duplicate.append(sample);
        if (appended || appended.error != DataError::OUTPUT_EXISTS)
            return fail("exclusive-create contract did not reject an existing output");
    }

    std::filesystem::remove(path, removeError);
    if (removeError)
        return fail("cannot remove the test output");
    std::cout << "ALICE_V2_CHUNK_V1 contract tests passed\n";
    return 0;
}
