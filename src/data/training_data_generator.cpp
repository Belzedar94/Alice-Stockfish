/*
  Alice-Stockfish native training-data generator
  Copyright (C) 2026 The Alice-Stockfish developers

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "training_data_generator.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "alice_v2_chunk.h"
#include "engine.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "random_seed.h"
#include "search.h"
#include "sha256.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#ifndef ALICE_SOURCE_COMMIT
    #define ALICE_SOURCE_COMMIT "0000000000000000000000000000000000000000"
#endif

#ifndef ALICE_SOURCE_DIRTY
    #define ALICE_SOURCE_DIRTY 1
#endif

namespace Stockfish::Data {
namespace {

constexpr u64              ReportEvery         = 5000;
constexpr int              MaximumGeneratedPly = 4096;
constexpr usize            DedupeTableSize     = usize(1) << 22;
constexpr std::string_view Run2RLSha256 =
  "9F9E557015A55C0A6981DB64E1F3044DEDB91FD8A8C1A6D4F3C45D0EEE91FBD9";

struct GeneratorParams {
    int   assignedThreads = 0;
    usize hashMb          = 512;
    u64   count           = 0;
    int   depth           = 6;
    u64   nodes           = 0;

    int randomMoveMinPly  = 1;
    int randomMoveMaxPly  = 20;
    int randomMoveCount   = 8;
    int randomMultiPv     = 4;
    int randomMultiPvDiff = 200;

    int writeMinPly = 5;
    int writeMaxPly = 400;
    int maxGamePly  = 512;

    std::string network;
    std::string networkSha256;
    std::string producerSha256;
    std::string book;
    std::string bookSha256;
    std::string outputFile;
    std::string runConfigSha256;
    u64         seed             = 0;
    u64         baseSeed         = 0;
    u64         totalRecords     = 0;
    u64         recordsPerChunk  = 0;
    u64         trainChunks      = 0;
    u64         validationChunks = 0;
    u64         testChunks       = 0;

    bool setRecommendedUciOptions = false;
};

struct RequiredParams {
    bool threads          = false;
    bool hash             = false;
    bool count            = false;
    bool network          = false;
    bool networkSha256    = false;
    bool producerSha256   = false;
    bool book             = false;
    bool bookSha256       = false;
    bool output           = false;
    bool seed             = false;
    bool runConfig        = false;
    bool baseSeed         = false;
    bool totalRecords     = false;
    bool recordsPerChunk  = false;
    bool trainChunks      = false;
    bool validationChunks = false;
    bool testChunks       = false;

    bool complete() const {
        return threads && hash && count && network && networkSha256 && producerSha256 && book
            && bookSha256 && output && seed && runConfig && baseSeed && totalRecords
            && recordsPerChunk && trainChunks && validationChunks && testChunks;
    }
};

struct GameResolution {
    bool                 terminal = false;
    std::optional<Color> winner;
    AliceOutcomeReason   reason = AliceOutcomeReason::STALEMATE;
};

template<typename T>
bool read_value(std::istream& input, T& value, std::string_view name, std::string& error) {
    if (input >> value)
        return true;
    error =
      "Missing or invalid value for alice_v2_generate_training_data option " + std::string(name);
    return false;
}

bool read_u64_value(std::istream& input, u64& value, std::string_view name, std::string& error) {
    std::string token;
    if (!(input >> token) || token.empty())
    {
        error = "Missing or invalid value for alice_v2_generate_training_data option "
              + std::string(name);
        return false;
    }

    u64 parsed = 0;
    for (const unsigned char c : token)
    {
        if (c < '0' || c > '9')
        {
            error = "Unsigned option " + std::string(name) + " must contain decimal digits only";
            return false;
        }
        const u64 digit = c - '0';
        if (parsed > (std::numeric_limits<u64>::max() - digit) / 10)
        {
            error = "Unsigned option " + std::string(name) + " is out of range";
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    value = parsed;
    return true;
}

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return char(std::toupper(c)); });
    return value;
}

bool parse_params(std::istream& input, GeneratorParams& params, std::string& error) {
    RequiredParams required;
    std::string    token;
    while (input >> token)
    {
        if (token == "threads")
        {
            required.threads = read_value(input, params.assignedThreads, token, error);
            if (!required.threads)
                return false;
        }
        else if (token == "hash")
        {
            required.hash = read_value(input, params.hashMb, token, error);
            if (!required.hash)
                return false;
        }
        else if (token == "count")
        {
            required.count = read_u64_value(input, params.count, token, error);
            if (!required.count)
                return false;
        }
        else if (token == "depth")
        {
            if (!read_value(input, params.depth, token, error))
                return false;
        }
        else if (token == "nodes")
        {
            if (!read_u64_value(input, params.nodes, token, error))
                return false;
        }
        else if (token == "network")
        {
            required.network = read_value(input, params.network, token, error);
            if (!required.network)
                return false;
        }
        else if (token == "network_sha256")
        {
            required.networkSha256 = read_value(input, params.networkSha256, token, error);
            if (!required.networkSha256)
                return false;
            params.networkSha256 = uppercase(params.networkSha256);
        }
        else if (token == "producer_sha256")
        {
            required.producerSha256 = read_value(input, params.producerSha256, token, error);
            if (!required.producerSha256)
                return false;
            params.producerSha256 = uppercase(params.producerSha256);
        }
        else if (token == "book")
        {
            required.book = read_value(input, params.book, token, error);
            if (!required.book)
                return false;
        }
        else if (token == "book_sha256")
        {
            required.bookSha256 = read_value(input, params.bookSha256, token, error);
            if (!required.bookSha256)
                return false;
            params.bookSha256 = uppercase(params.bookSha256);
        }
        else if (token == "out")
        {
            required.output = read_value(input, params.outputFile, token, error);
            if (!required.output)
                return false;
        }
        else if (token == "seed")
        {
            required.seed = read_u64_value(input, params.seed, token, error);
            if (!required.seed)
                return false;
        }
        else if (token == "run_config_sha256")
        {
            required.runConfig = read_value(input, params.runConfigSha256, token, error);
            if (!required.runConfig)
                return false;
            params.runConfigSha256 = uppercase(params.runConfigSha256);
        }
        else if (token == "base_seed")
        {
            required.baseSeed = read_u64_value(input, params.baseSeed, token, error);
            if (!required.baseSeed)
                return false;
        }
        else if (token == "total_records")
        {
            required.totalRecords = read_u64_value(input, params.totalRecords, token, error);
            if (!required.totalRecords)
                return false;
        }
        else if (token == "records_per_chunk")
        {
            required.recordsPerChunk = read_u64_value(input, params.recordsPerChunk, token, error);
            if (!required.recordsPerChunk)
                return false;
        }
        else if (token == "train_chunks")
        {
            required.trainChunks = read_u64_value(input, params.trainChunks, token, error);
            if (!required.trainChunks)
                return false;
        }
        else if (token == "validation_chunks")
        {
            required.validationChunks =
              read_u64_value(input, params.validationChunks, token, error);
            if (!required.validationChunks)
                return false;
        }
        else if (token == "test_chunks")
        {
            required.testChunks = read_u64_value(input, params.testChunks, token, error);
            if (!required.testChunks)
                return false;
        }
        else if (token == "random_move_min_ply")
        {
            if (!read_value(input, params.randomMoveMinPly, token, error))
                return false;
        }
        else if (token == "random_move_max_ply")
        {
            if (!read_value(input, params.randomMoveMaxPly, token, error))
                return false;
        }
        else if (token == "random_move_count")
        {
            if (!read_value(input, params.randomMoveCount, token, error))
                return false;
        }
        else if (token == "random_multi_pv")
        {
            if (!read_value(input, params.randomMultiPv, token, error))
                return false;
        }
        else if (token == "random_multi_pv_diff")
        {
            if (!read_value(input, params.randomMultiPvDiff, token, error))
                return false;
        }
        else if (token == "write_min_ply")
        {
            if (!read_value(input, params.writeMinPly, token, error))
                return false;
        }
        else if (token == "write_max_ply")
        {
            if (!read_value(input, params.writeMaxPly, token, error))
                return false;
        }
        else if (token == "max_game_ply")
        {
            if (!read_value(input, params.maxGamePly, token, error))
                return false;
        }
        else if (token == "set_recommended_uci_options")
            params.setRecommendedUciOptions = true;
        else
        {
            error = "Unknown alice_v2_generate_training_data option " + token;
            return false;
        }
    }

    if (!required.complete())
    {
        error = "The Alice v2 DATAGEN identity contract requires threads, hash, count, network, "
                "network_sha256, producer_sha256, book, book_sha256, out, seed, "
                "run_config_sha256, base_seed, total_records, records_per_chunk, "
                "train_chunks, validation_chunks, and test_chunks";
        return false;
    }
    if (params.network == "NONE" || params.networkSha256 != Run2RLSha256)
    {
        error = "Network identity does not match alice_run2rl_e40_l09.nnue";
        return false;
    }
    if (!is_upper_sha256(params.producerSha256) || !is_upper_sha256(params.runConfigSha256)
        || params.outputFile.empty() || params.assignedThreads <= 0 || params.count == 0
        || params.totalRecords == 0 || params.recordsPerChunk == 0 || params.hashMb == 0
        || params.depth <= 0 || params.depth >= MAX_PLY || params.randomMoveMinPly < 0
        || params.randomMoveMaxPly < params.randomMoveMinPly
        || params.randomMoveMaxPly >= params.maxGamePly
        || params.randomMoveMaxPly > MaximumGeneratedPly || params.randomMoveCount < 0
        || params.randomMoveCount > MaximumGeneratedPly || params.randomMultiPv < 0
        || params.randomMultiPv > int(MAX_MOVES) || params.randomMultiPvDiff < 0
        || params.writeMinPly < 0 || params.writeMaxPly <= params.writeMinPly
        || params.maxGamePly <= params.writeMaxPly || params.maxGamePly > MaximumGeneratedPly
        || !params.setRecommendedUciOptions)
    {
        error = "Invalid alice_v2_generate_training_data parameter range";
        return false;
    }
    if ((params.book == "NONE") != (params.bookSha256 == "NONE")
        || (params.book != "NONE" && !is_upper_sha256(params.bookSha256)))
    {
        error = "book and book_sha256 must either both be NONE or identify one authenticated book";
        return false;
    }
    return true;
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::optional<std::string> normalize_book_line(std::string line) {
    line = trim(std::move(line));
    if (line.empty() || line[0] == '#')
        return std::nullopt;

    std::istringstream       stream(line);
    std::vector<std::string> fields;
    std::string              field;
    while (stream >> field)
        fields.push_back(field);
    if (fields.size() < 4)
        return std::nullopt;

    std::ostringstream fen;
    for (usize i = 0; i < 4; ++i)
        fen << (i ? " " : "") << fields[i];

    const auto decimal = [](const std::string& value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
    };
    const bool hasClocks = fields.size() >= 6 && decimal(fields[4]) && decimal(fields[5]);
    fen << (hasClocks ? " " + fields[4] + " " + fields[5] : " 0 1");
    return fen.str();
}

bool is_terminal_position(Position& position) {
    return position.is_draw(0) || MoveList<LEGAL>(position).size() == 0;
}

bool load_book(const GeneratorParams&    params,
               std::vector<std::string>& positions,
               std::string&              error) {
    if (params.book == "NONE")
    {
        positions.emplace_back(StartFEN);
        return true;
    }

    std::string observedSha;
    if (!sha256_file(params.book, observedSha, error))
        return false;
    if (observedSha != params.bookSha256)
    {
        error =
          "Opening-book SHA-256 mismatch: expected " + params.bookSha256 + ", got " + observedSha;
        return false;
    }

    std::ifstream input(params.book);
    if (!input)
    {
        error = "Cannot open Alice training-data opening book";
        return false;
    }

    std::string line;
    u64         lineNumber = 0;
    while (std::getline(input, line))
    {
        ++lineNumber;
        const std::string normalized = trim(line);
        if (normalized.empty() || normalized[0] == '#')
            continue;
        auto candidate = normalize_book_line(normalized);
        if (!candidate)
        {
            error =
              "Opening book contains a malformed EPD/FEN at line " + std::to_string(lineNumber);
            return false;
        }

        Position  position;
        StateInfo state{};
        if (position.set(*candidate, false, &state) || position.is_chess960()
            || position.count<KING>(WHITE) != 1 || position.count<KING>(BLACK) != 1
            || position.count<ALL_PIECES>() < 2 || position.count<ALL_PIECES>() > 32
            || position.ep_square() != SQ_NONE || is_terminal_position(position))
        {
            error = "Opening book contains an invalid, terminal, or unsupported Alice FEN";
            return false;
        }
        positions.push_back(position.fen());
    }

    if (positions.empty())
    {
        error = "Alice training-data opening book contains no usable positions";
        return false;
    }
    return true;
}

Color side_to_move_from_fen(const std::string& fen) {
    const auto separator = fen.find(' ');
    assert(separator != std::string::npos && separator + 1 < fen.size());
    return fen[separator + 1] == 'b' ? BLACK : WHITE;
}

GameResolution current_resolution(Position& position) {
    const MoveList<LEGAL> legalMoves(position);
    if (legalMoves.size() == 0)
        return {
          true, position.checkers() ? std::optional<Color>(~position.side_to_move()) : std::nullopt,
          position.checkers() ? AliceOutcomeReason::CHECKMATE : AliceOutcomeReason::STALEMATE};
    if (position.rule50_count() > 99)
        return {true, std::nullopt, AliceOutcomeReason::FIFTY_MOVE};
    if (position.is_draw(0))
        return {true, std::nullopt, AliceOutcomeReason::REPETITION};
    return {};
}

class SeenPositions {
   public:
    SeenPositions() :
        slots(std::make_unique<std::atomic<Key>[]>(DedupeTableSize)) {
        for (usize i = 0; i < DedupeTableSize; ++i)
            slots[i].store(0, std::memory_order_relaxed);
    }

    bool already_seen(Key key) {
        const usize index = usize(key) & (DedupeTableSize - 1);
        return slots[index].exchange(key, std::memory_order_relaxed) == key;
    }

   private:
    std::unique_ptr<std::atomic<Key>[]> slots;
};

class Generator {
   public:
    Generator(const GeneratorParams&   params_,
              ThreadPool&              threads_,
              TranspositionTable&      tt_,
              std::filesystem::path    outputPath,
              AliceV2ChunkV1Manifest   manifest,
              std::vector<std::string> openingPositions_,
              u64                      resolvedSeed) :
        params(params_),
        threads(threads_),
        tt(tt_),
        output(std::move(outputPath), std::move(manifest)),
        openingPositions(std::move(openingPositions_)) {
        ReplayablePRNG source(resolvedSeed);
        workerSeeds.reserve(threads.size());
        for (usize i = 0; i < threads.size(); ++i)
            workerSeeds.push_back(i == 0 ? source.seed() : source.next_seed());

        if (openingPositions.size() > 1)
        {
            ReplayablePRNG bookRng(workerSeeds[0]);
            for (usize i = 0; i < openingPositions.size(); ++i)
                std::swap(openingPositions[i],
                          openingPositions[i + bookRng.rand(openingPositions.size() - i)]);
            workerSeeds[0] = bookRng.seed();
        }
    }

    bool run(std::string& error) {
        threads.wait_for_search_finished();
        threads.clear();
        tt.clear(threads);
        threads.stop = false;
        tt.new_search();

        std::vector<Thread*> workers;
        workers.reserve(threads.size());
        for (auto& thread : threads)
            workers.push_back(thread.get());

        for (usize i = 0; i < workers.size(); ++i)
        {
            Search::Worker* worker = workers[i]->worker.get();
            threads.run_on_thread(i, [this, worker, i]() { run_worker(*worker, i); });
        }
        for (usize i = 0; i < workers.size(); ++i)
            threads.wait_on_thread(i);
        threads.stop = false;

        std::lock_guard lock(outputMutex);
        if (!failure.empty())
        {
            const DataResult cleanup = output.abort();
            error                    = failure;
            if (!cleanup)
                error += "; cleanup failed: " + cleanup.message;
            return false;
        }
        if (DataResult result = output.finalize(); !result)
        {
            const DataResult cleanup = output.abort();
            error                    = result.message;
            if (!cleanup)
                error += "; cleanup failed: " + cleanup.message;
            return false;
        }
        return true;
    }

    u64 records_written() const { return written.load(std::memory_order_relaxed); }
    u64 draws_written() const { return draws.load(std::memory_order_relaxed); }
    u64 truncated_games() const { return truncated.load(std::memory_order_relaxed); }

   private:
    const GeneratorParams&   params;
    ThreadPool&              threads;
    TranspositionTable&      tt;
    AliceV2ChunkV1Sink       output;
    std::vector<std::string> openingPositions;
    std::vector<u64>         workerSeeds;
    SeenPositions            seen;

    std::atomic<u64>                            written{0};
    std::atomic<u64>                            draws{0};
    std::atomic<u64>                            truncated{0};
    std::atomic<bool>                           finished{false};
    std::mutex                                  outputMutex;
    std::string                                 failure;
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

    void fail(std::string message) {
        {
            std::lock_guard lock(outputMutex);
            if (failure.empty())
                failure = std::move(message);
        }
        finished.store(true, std::memory_order_relaxed);
        threads.stop = true;
    }

    std::string next_opening(usize workerIndex, u64 gameIndex) const {
        const usize index =
          (workerIndex + usize(gameIndex % openingPositions.size()) * threads.size())
          % openingPositions.size();
        return openingPositions[index];
    }

    std::vector<u8> random_move_flags(ReplayablePRNG& rng) const {
        std::vector<u8>  flags(usize(params.maxGamePly), 0);
        std::vector<int> candidates;
        for (int ply = params.randomMoveMinPly; ply <= params.randomMoveMaxPly; ++ply)
            candidates.push_back(ply);

        const int count = std::min(params.randomMoveCount, int(candidates.size()));
        for (int i = 0; i < count; ++i)
        {
            const usize selected = usize(i) + usize(rng.rand(candidates.size() - usize(i)));
            std::swap(candidates[usize(i)], candidates[selected]);
            flags[usize(candidates[usize(i)])] = 1;
        }
        return flags;
    }

    Move choose_played_move(const Position&                     position,
                            const Search::TrainingSearchResult& search,
                            ReplayablePRNG&                     rng,
                            bool                                explore) const {
        assert(!search.pv.empty());
        const Move bestMove = search.pv[0];
        if (!explore)
            return bestMove;

        if (params.randomMultiPv == 0 || search.lines.empty())
        {
            const MoveList<LEGAL> moves(position);
            return *(moves.begin() + rng.rand(moves.size()));
        }

        usize candidates = search.lines.size();
        for (usize i = 1; i < candidates; ++i)
            if (std::int64_t(search.lines.front().value)
                > std::int64_t(search.lines[i].value) + params.randomMultiPvDiff)
            {
                candidates = i;
                break;
            }
        const auto& line = search.lines[rng.rand(candidates)];
        return line.pv.empty() ? bestMove : line.pv[0];
    }

    bool commit_game(std::vector<TrainingDataSample>& samples,
                     std::optional<Color>             winner,
                     AliceOutcomeReason               reason) {
        if (samples.empty())
            return false;

        std::lock_guard lock(outputMutex);
        if (!failure.empty() || finished.load(std::memory_order_relaxed))
            return true;

        const bool draw = !winner.has_value();
        for (auto& sample : samples)
        {
            if (written.load(std::memory_order_relaxed) >= params.count)
            {
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }

            sample.result = winner ? (side_to_move_from_fen(sample.fen) == *winner ? 1 : -1) : 0;
            sample.outcomeReason = reason;
            if (DataResult result = output.append(sample); !result)
            {
                failure = result.message;
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }

            const u64 done = written.fetch_add(1, std::memory_order_relaxed) + 1;
            if (draw)
                draws.fetch_add(1, std::memory_order_relaxed);
            if (done % ReportEvery == 0 || done == params.count)
            {
                const auto elapsed = std::max(
                  1.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                         .count());
                std::cout << "info string Alice v2 training data " << done << "/" << params.count
                          << " records, " << u64(double(done) / elapsed) << " records/s"
                          << std::endl;
            }
            if (done >= params.count)
            {
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }
        }
        return false;
    }

    void run_worker(Search::Worker& worker, usize workerIndex) {
        ReplayablePRNG         rng(workerSeeds[workerIndex]);
        std::vector<StateInfo> states(usize(params.maxGamePly));
        u64                    gameIndex = 0;

        while (!finished.load(std::memory_order_relaxed))
        {
            StateInfo         rootState{};
            Position          position;
            const std::string initialFen = next_opening(workerIndex, gameIndex++);
            if (const auto setError = position.set(initialFen, false, &rootState))
            {
                fail(std::string("Cannot set Alice opening FEN: ") + setError->what());
                return;
            }

            std::vector<TrainingDataSample> samples;
            samples.reserve(usize(params.writeMaxPly - params.writeMinPly));
            const std::vector<u8> explore = random_move_flags(rng);

            for (int ply = 0; !finished.load(std::memory_order_relaxed); ++ply)
            {
                const GameResolution resolution = current_resolution(position);
                if (resolution.terminal)
                {
                    commit_game(samples, resolution.winner, resolution.reason);
                    break;
                }
                if (ply >= params.maxGamePly)
                {
                    truncated.fetch_add(1, std::memory_order_relaxed);
                    break;
                }

                Search::TrainingSearchRequest request;
                request.depth = params.depth;
                request.nodes = params.nodes;
                request.multiPV =
                  explore[usize(ply)] ? usize(std::max(params.randomMultiPv, 1)) : 1;
                const auto search = worker.training_search(position, request);
                if (finished.load(std::memory_order_relaxed))
                    break;
                if (search.value == VALUE_NONE || search.pv.empty())
                {
                    fail("Synchronous Alice training search returned no principal variation");
                    return;
                }
                if (!search.exact
                    || std::any_of(search.lines.begin(), search.lines.end(),
                                   [](const auto& line) { return !line.exact; }))
                {
                    fail("Synchronous Alice training search returned a bound score");
                    return;
                }

                const Move bestMove = search.pv[0];
                const Move playedMove =
                  choose_played_move(position, search, rng, explore[usize(ply)]);
                const MoveList<LEGAL> legalMoves(position);
                if (!legalMoves.contains(bestMove) || !legalMoves.contains(playedMove))
                {
                    fail("Alice training generator selected an illegal move");
                    return;
                }

                if (ply >= params.writeMinPly && ply < params.writeMaxPly
                    && !seen.already_seen(position.key()))
                    samples.push_back({position.fen(), int(search.value), bestMove, playedMove, 0,
                                       AliceOutcomeReason::STALEMATE});

                position.do_move(playedMove, states[usize(ply)], &tt);
            }
        }
    }
};

void set_engine_option(Engine& engine, const std::string& name, const std::string& value) {
    std::istringstream option("name " + name + " value " + value);
    engine.get_options().setoption(option);
}

void print_error(std::string_view message) { sync_cout << "ERROR: " << message << sync_endl; }

}  // namespace

bool generate_training_data(Engine& engine, std::istream& input) {
    GeneratorParams params;
    std::string     error;
    if (!parse_params(input, params, error))
    {
        print_error(error);
        return false;
    }

    engine.wait_for_search_finished();

    std::string observedNetworkSha;
    if (!sha256_file(params.network, observedNetworkSha, error))
    {
        print_error(error);
        return false;
    }
    if (observedNetworkSha != params.networkSha256 || observedNetworkSha != Run2RLSha256)
    {
        print_error("Network bytes do not match alice_run2rl_e40_l09.nnue");
        return false;
    }

    // OpenBench supplies all assigned worker threads for custody evidence. The
    // data search itself is deliberately single-threaded so a retry of one
    // logical chunk has a byte-reproducible search and record order.
    set_engine_option(engine, "Threads", "1");
    set_engine_option(engine, "Hash", std::to_string(params.hashMb));
    set_engine_option(engine, "UCI_Chess960", "false");
    set_engine_option(engine, "Skill Level", "20");
    set_engine_option(engine, "UCI_LimitStrength", "false");
    set_engine_option(engine, "Alice Evaluation", "Legacy");
    set_engine_option(engine, "Use NNUE", "true");
    set_engine_option(engine, "Alice_Frozen_Network", "true");
    set_engine_option(engine, "EvalFile", params.network);
    if (int(engine.options["Threads"]) != 1 || usize(engine.options["Hash"]) != params.hashMb
        || int(engine.options["UCI_Chess960"]) != 0
        || std::string(engine.options["Alice Evaluation"]) != "Legacy"
        || int(engine.options["Use NNUE"]) != 1
        || std::string(engine.options["EvalFile"]) != params.network
        || !engine.legacyEvaluator.loaded())
    {
        print_error("Engine options did not accept the requested Alice v2 DATAGEN configuration");
        return false;
    }
    engine.verify_network();

    std::vector<std::string> openings;
    if (!load_book(params, openings, error))
    {
        print_error(error);
        return false;
    }

    std::error_code ec;
    if (std::filesystem::exists(params.outputFile, ec) || ec)
    {
        print_error(ec ? "Cannot inspect Alice v2 DATAGEN output path"
                       : "Alice v2 DATAGEN output already exists");
        return false;
    }

    const u64              resolvedSeed = params.seed;
    AliceV2ChunkV1Manifest manifest;
    manifest.runConfigSha256  = params.runConfigSha256;
    manifest.sourceCommit     = ALICE_SOURCE_COMMIT;
    manifest.sourceDirty      = ALICE_SOURCE_DIRTY != 0;
    manifest.networkSha256    = params.networkSha256;
    manifest.bookSha256       = params.bookSha256;
    manifest.requestedRecords = params.count;
    manifest.seed             = resolvedSeed;
    manifest.baseSeed         = params.baseSeed;
    manifest.totalRecords     = params.totalRecords;
    manifest.recordsPerChunk  = params.recordsPerChunk;
    manifest.totalChunks      = 1 + (params.totalRecords - 1) / params.recordsPerChunk;
    if (resolvedSeed < params.baseSeed)
    {
        print_error("Chunk seed precedes the preregistered base seed");
        return false;
    }
    manifest.chunkIndex       = resolvedSeed - params.baseSeed;
    manifest.trainChunks      = params.trainChunks;
    manifest.validationChunks = params.validationChunks;
    manifest.testChunks       = params.testChunks;
    manifest.split =
      manifest.chunkIndex < manifest.trainChunks
        ? "train"
        : (manifest.chunkIndex < manifest.trainChunks + manifest.validationChunks ? "validation"
                                                                                  : "test");
    manifest.searchThreads     = 1;
    manifest.hashMb            = params.hashMb;
    manifest.depth             = params.depth;
    manifest.nodes             = params.nodes;
    manifest.randomMoveMinPly  = params.randomMoveMinPly;
    manifest.randomMoveMaxPly  = params.randomMoveMaxPly;
    manifest.randomMoveCount   = params.randomMoveCount;
    manifest.randomMultiPv     = params.randomMultiPv;
    manifest.randomMultiPvDiff = params.randomMultiPvDiff;
    manifest.writeMinPly       = params.writeMinPly;
    manifest.writeMaxPly       = params.writeMaxPly;
    manifest.maxGamePly        = params.maxGamePly;
    manifest.openingCount      = openings.size();
    if (DataResult valid = validate_alice_v2_chunk_v1_manifest(manifest); !valid)
    {
        print_error(valid.message);
        return false;
    }

    std::cout << "INFO: Executing alice_v2_generate_training_data command\n"
              << "INFO: schema_sha256 = " << AliceV2ChunkV1SchemaSha256 << '\n'
              << "INFO: record_schema_sha256 = " << AliceV2RecordV1SchemaSha256 << '\n'
              << "INFO: source_commit = " << manifest.sourceCommit << '\n'
              << "INFO: source_dirty = " << (manifest.sourceDirty ? "true" : "false") << '\n'
              << "PRNG::initial_seed = " << resolvedSeed << '\n'
              << "INFO: run_config_sha256 = " << manifest.runConfigSha256 << '\n'
              << "INFO: chunk_index = " << manifest.chunkIndex << '/' << manifest.totalChunks
              << " split=" << manifest.split << '\n'
              << "INFO: assigned_threads = " << params.assignedThreads << '\n'
              << "INFO: search_threads = 1\n"
              << "INFO: depth = " << params.depth << '\n'
              << "INFO: count = " << params.count << '\n'
              << "INFO: openings = " << openings.size() << std::endl;

    Generator generator(params, engine.threads, engine.tt, params.outputFile, std::move(manifest),
                        std::move(openings), resolvedSeed);
    if (!generator.run(error))
    {
        print_error(error);
        return false;
    }

    std::cout << "INFO: alice_v2_generate_training_data finished.\n"
              << "INFO: records=" << generator.records_written()
              << " draws=" << generator.draws_written()
              << " truncated_games=" << generator.truncated_games() << std::endl;
    return true;
}

}  // namespace Stockfish::Data
