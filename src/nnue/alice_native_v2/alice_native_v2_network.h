/*
  Alice-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Alice-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_V2_NETWORK_H_INCLUDED
#define NNUE_ALICE_NATIVE_V2_NETWORK_H_INCLUDED

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "../../types.h"
#include "alice_native_v2_inference.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::AliceNativeV2 {

struct WireMetadata {
    std::string normalizedPath;
    u64         bytes         = 0;
    u64         payloadBytes  = 0;
    u32         manifestBytes = 0;
    std::string sha256;
    std::string manifestSha256;
    u32         version = 0;
};

struct IncrementalVerificationStats {
    u64 positions              = 0;
    u64 transitions            = 0;
    u64 captures               = 0;
    u64 promotions             = 0;
    u64 castlings              = 0;
    u64 kingMoves              = 0;
    u64 fullRefreshes[COLOR_NB] = {};
    u64 pieceAdds              = 0;
    u64 pieceRemoves           = 0;
    u64 accumulatorComparisons = 0;
    u64 stageComparisons       = 0;
    u64 fastPathComparisons    = 0;
    u64 undoChecks             = 0;
};

class Network {
   private:
    struct Parameters;

   public:
    class Lease {
       public:
        Lease() noexcept = default;
        ~Lease();

        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;

        explicit      operator bool() const noexcept;
        ParameterView parameter_view() const noexcept;
        u64           generation() const noexcept;
        std::string_view normalized_path() const noexcept;
        std::string_view sha256() const noexcept;

       private:
        friend class Network;
        Lease(const Network* owner, const Parameters* parameters) noexcept;
        void reset() noexcept;

        const Network*    network = nullptr;
        const Parameters* pinned  = nullptr;
    };

    Network();
    ~Network();

    Network(const Network&)            = delete;
    Network(Network&&)                 = delete;
    Network& operator=(const Network&) = delete;
    Network& operator=(Network&&)      = delete;

    std::optional<std::string> load(const std::filesystem::path& file,
                                    std::string_view             expectedSha256);
    std::optional<Lease>       acquire_lease(std::string& error) const noexcept;

    bool               loaded() const;
    u64                generation() const;
    const std::string& last_error() const;
    std::string        status_line() const;
    std::string        tensor_status_line() const;
    std::optional<std::string>
    probe(std::string_view tensor, u64 index, std::string& report) const;
    std::optional<std::string> integer_trace(const Position& position, std::string& report) const;
    std::optional<std::string>
    verify_incremental(Position& position, Depth depth, IncrementalVerificationStats& stats) const;
    std::optional<std::string>
    verify_session(Position& position, Depth depth, std::string& report) const;

   private:
    void release_lease() const noexcept;

    std::unique_ptr<Parameters> active;
    std::string                 lastError;
    mutable std::atomic<u64>    activeLeases{0};
    std::atomic_bool            replacementInProgress{false};
};

}  // namespace Stockfish::Eval::NNUE::AliceNativeV2

#endif  // NNUE_ALICE_NATIVE_V2_NETWORK_H_INCLUDED
