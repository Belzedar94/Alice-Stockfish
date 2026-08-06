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

#include <filesystem>
#include <optional>
#include <string>

#include "../../types.h"
#include "manifest.h"

namespace Stockfish::Eval::NNUE::AliceNative {

struct WireMetadata {
    std::string normalizedPath;
    u64         bytes = 0;
    std::string sha256;
    std::string manifestSha256;
    u32         version      = 0;
    u32         architecture = 0;
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

}  // namespace Stockfish::Eval::NNUE::AliceNative

#endif  // NNUE_ALICE_NATIVE_NETWORK_H_INCLUDED
