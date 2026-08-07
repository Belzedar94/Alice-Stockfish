/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_H_INCLUDED
#define ENGINE_H_INCLUDED

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "misc.h"
#include "history.h"
#include "legacy_alice_nnue.h"
#include "nnue/alice_native/alice_native_network.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "syzygy/tbprobe.h"  // for Stockfish::Depth
#include "thread.h"
#include "tt.h"
#include "ucioption.h"

namespace Stockfish {

class Engine {
   public:
    using InfoShort = Search::InfoShort;
    using InfoFull  = Search::InfoFull;
    using InfoIter  = Search::InfoIteration;

    Engine(std::optional<std::filesystem::path> path = std::nullopt);

    // Cannot be movable due to components holding backreferences to fields
    Engine(const Engine&)            = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(const Engine&) = delete;
    Engine& operator=(Engine&&)      = delete;

    ~Engine();

    std::variant<u64, PositionSetError> perft(const std::string& fen, Depth depth, bool isChess960);

    // non blocking call to start searching
    std::optional<std::string> go(Search::LimitsType&);
    // non blocking call to stop searching
    void stop();

    // blocking call to wait for search to finish
    void wait_for_search_finished();
    // set a new position, moves are in UCI format
    std::optional<PositionSetError> set_position(const std::string&              fen,
                                                 const std::vector<std::string>& moves);

    // modifiers

    bool set_numa_config_from_option(const std::string& o);
    void resize_threads();
    void set_tt_size(usize mb);
    void set_ponderhit(bool);
    void search_clear();

    void set_on_update_no_moves(std::function<void(const InfoShort&)>&&);
    void set_on_update_full(std::function<void(const InfoFull&)>&&);
    void set_on_iter(std::function<void(const InfoIter&)>&&);
    void set_on_bestmove(std::function<void(std::string_view, std::string_view)>&&);
    void set_on_start(std::function<void()>&&);
    void set_on_search_error(std::function<void(std::string_view)>&&);
    void set_on_verify_network(std::function<void(std::string_view)>&&);

    // network related

    void verify_network() const;

    // utility functions

    std::optional<std::string> trace_eval() const;
    std::optional<std::string> verify_search_contract(std::string& report);
    std::string                trace_native_features();
    std::optional<std::string> verify_native_incremental(Depth depth, std::string& report);
    std::optional<std::string>
                               validate_native_wire(const std::filesystem::path&      file,
                                                    const std::optional<std::string>& expectedSha256 = {});
    std::string                native_wire_status() const;
    std::optional<std::string> load_native_qualification(const std::filesystem::path& file,
                                                         std::string_view expectedSha256);
    std::string                native_qualification_status() const;
    std::string                native_tensor_status() const;
    std::optional<std::string>
    probe_native_parameter(std::string_view tensor, u64 index, std::string& report) const;
    std::optional<std::string> trace_native_integer(std::string& report);
    std::optional<std::string> verify_loaded_native_incremental(Depth depth, std::string& report);
    std::optional<std::string> verify_native_search_session(Depth depth, std::string& report);
    std::optional<std::string> verify_native_lease(std::string& report);
    std::optional<std::string> verify_legacy_incremental(Depth depth, u64& positions);

    const OptionsMap& get_options() const;
    OptionsMap&       get_options();

    int get_hashfull(int maxAge = 0) const;

    std::string                          fen() const;
    std::optional<PositionSetError>      flip();
    std::string                          visualize() const;
    std::vector<std::pair<usize, usize>> get_bound_thread_count_by_numa_node() const;
    std::string                          get_numa_config_as_string() const;
    std::string                          numa_config_information_as_string() const;
    std::string                          thread_allocation_information_as_string() const;
    std::string                          thread_binding_information_as_string() const;

   private:
    NumaReplicationContext numaContext;

    Position     pos;
    StateListPtr states;

    OptionsMap                                        options;
    ThreadPool                                        threads;
    TranspositionTable                                tt;
    LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;
    LegacyAliceExact                                  legacyEvaluator;
    Eval::NNUE::AliceNative::WireValidator            nativeWireValidator;
    Eval::NNUE::AliceNative::QualificationNetwork     nativeQualification;

    Search::SearchManager::UpdateContext  updateContext;
    std::function<void(std::string_view)> onSearchError;
    std::function<void(std::string_view)> onVerifyNetwork;
    std::map<NumaIndex, SharedHistories>  sharedHists;

    std::thread      aliceSearchThread;
    std::atomic_bool aliceSearchStop{false};
    std::atomic_bool alicePondering{false};

    std::optional<std::string> configure_legacy_network(const std::filesystem::path&);
    std::optional<std::string> configure_native_network();
    std::optional<Eval::NNUE::AliceNative::QualificationNetwork::Lease>
    lease_native_network(std::string& error) const;
};

}  // namespace Stockfish


#endif  // #ifndef ENGINE_H_INCLUDED
