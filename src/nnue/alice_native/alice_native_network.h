/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef NNUE_ALICE_NATIVE_NETWORK_H_INCLUDED
#define NNUE_ALICE_NATIVE_NETWORK_H_INCLUDED

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "../../types.h"
#include "manifest.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::AliceNative {

struct ParameterView;

struct WireMetadata {
    std::string normalizedPath;
    u64         bytes = 0;
    std::string sha256;
    std::string manifestSha256;
    u32         version      = 0;
    u32         architecture = 0;
};

struct LoadedIncrementalVerificationStats {
    u64 positions               = 0;
    u64 transitions             = 0;
    u64 captures                = 0;
    u64 promotions              = 0;
    u64 castlings               = 0;
    u64 kingMoves               = 0;
    u64 fullRefreshes[COLOR_NB] = {};
    u64 pieceAdds               = 0;
    u64 pieceRemoves            = 0;
    u64 threatAdds              = 0;
    u64 threatRemoves           = 0;
    u64 maxPieceEvents          = 0;
    u64 maxThreatEvents         = 0;
    u64 accumulatorComparisons  = 0;
    u64 integerStageComparisons = 0;
    u64 featureSimdComparisons  = 0;
    u64 denseSimdComparisons    = 0;
    u64 fixedAccumulatorChecks  = 0;
    u64 fixedDeltaUpdates       = 0;
    u64 undoChecks              = 0;
};

// Validates the complete native integer wire container without exposing it as
// an evaluator. Parameter allocation and evaluation routing remain separate
// qualification gates.
class WireValidator {
   public:
    void reset();

    std::optional<std::string> validate(const std::filesystem::path&      file,
                                        const std::optional<std::string>& expectedSha256 = {});

    bool                valid() const;
    const WireMetadata& metadata() const;
    const std::string&  last_error() const;
    std::string         status_line() const;

   private:
    bool         ready = false;
    WireMetadata current;
    std::string  lastError;
};

// Owns one fully parsed native parameter set. Loading
// requires a caller-trusted SHA-256 and commits with one pointer swap only
// after same-handle authentication, parsing, and canonical traversal checks.
// A move-only lease pins the immutable object for a complete search session.
class QualificationNetwork {
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

        explicit         operator bool() const noexcept;
        ParameterView    parameter_view() const noexcept;
        u64              generation() const noexcept;
        std::string_view normalized_path() const noexcept;
        std::string_view sha256() const noexcept;
        u32              version() const noexcept;
        u32              architecture() const noexcept;

       private:
        friend class QualificationNetwork;
        Lease(const QualificationNetwork* owner, const Parameters* parameters) noexcept;
        void reset() noexcept;

        const QualificationNetwork* network = nullptr;
        const Parameters*           pinned  = nullptr;
    };

    QualificationNetwork();
    ~QualificationNetwork();

    QualificationNetwork(const QualificationNetwork&)            = delete;
    QualificationNetwork(QualificationNetwork&&)                 = delete;
    QualificationNetwork& operator=(const QualificationNetwork&) = delete;
    QualificationNetwork& operator=(QualificationNetwork&&)      = delete;

    std::optional<std::string> load(const std::filesystem::path& file,
                                    std::string_view             expectedSha256);

    std::optional<Lease>       acquire_lease(std::string& error) const noexcept;
    bool                       has_active_lease() const noexcept;
    std::optional<std::string> verify_lease_contract(std::string& report);

    bool                       loaded() const;
    u64                        generation() const;
    const std::string&         last_error() const;
    std::string                status_line() const;
    std::string                tensor_status_line() const;
    std::optional<std::string> probe(std::string_view tensor, u64 index, std::string& report) const;
    std::optional<std::string> integer_trace(const Position& position, std::string& report) const;
    std::optional<std::string> verify_incremental(Position&                           position,
                                                  Depth                               depth,
                                                  LoadedIncrementalVerificationStats& stats) const;
    std::optional<std::string>
    verify_session(Position& position, Depth depth, std::string& report) const;

   private:
    void release_lease() const noexcept;

    std::unique_ptr<Parameters> active;
    std::string                 lastError;
    mutable std::atomic<u64>    activeLeases{0};
    std::atomic_bool            replacementInProgress{false};
};

}  // namespace Stockfish::Eval::NNUE::AliceNative

#endif  // NNUE_ALICE_NATIVE_NETWORK_H_INCLUDED
