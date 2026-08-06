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

#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cassert>
#include <deque>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "alice_search.h"
#include "evaluate.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "numa.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "shm.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

constexpr int MaxHashMB  = Is64Bit ? 33554432 : 2048;
int           MaxThreads = std::max(1024, 4 * int(get_hardware_concurrency()));

// The default configuration will attempt to group L3 domains up to 32 threads.
// This size was found to be a good balance between the Elo gain of increased
// history sharing and the speed loss from more cross-cache accesses (see
// PR#6526). The user can always explicitly override this behavior.
constexpr NumaAutoPolicy DefaultNumaPolicy = BundledL3Policy{32};

Engine::Engine(std::optional<std::filesystem::path> path) :
    binaryDirectory(path ? CommandLine::get_binary_directory(*path) : std::filesystem::path{}),
    numaContext(NumaConfig::from_system(DefaultNumaPolicy)),
    states(new std::deque<StateInfo>(1)),
    threads(),
    networkFile{std::nullopt, ""},
    network(numaContext, get_default_network()) {

    pos.set(StartFEN, false, &states->back());

    options.add(  //
      "Debug Log File", Option("", [](const Option& o) {
          start_logger(path_from_utf8(std::string(o)));
          return std::nullopt;
      }));

    options.add(  //
      "NumaPolicy", Option("auto", [this](const Option& o) {
          if (!set_numa_config_from_option(o))
              return "NumaPolicy: invalid value '" + std::string(o) + "', keeping previous config.";
          return numa_config_information_as_string() + "\n"
               + thread_allocation_information_as_string();
      }));

    options.add(  //
      "Threads", Option(1, 1, MaxThreads, [this](const Option&) {
          resize_threads();
          return thread_allocation_information_as_string();
      }));

    options.add(  //
      "Hash", Option(16, 1, MaxHashMB, [this](const Option& o) {
          set_tt_size(o);
          return std::nullopt;
      }));

    options.add(  //
      "Clear Hash", Option([this](const Option&) {
          search_clear();
          return std::nullopt;
      }));

    options.add(  //
      "Ponder", Option(false));

    options.add(  //
      "MultiPV", Option(1, 1, MAX_MOVES));

    options.add("Skill Level", Option(20, 0, 20));

    options.add("Move Overhead", Option(10, 0, 5000));

    options.add("nodestime", Option(0, 0, 10000));

    options.add("UCI_Chess960", Option(false));

    options.add("UCI_LimitStrength", Option(false));

    options.add("UCI_Elo",
                Option(Stockfish::Search::Skill::LowestElo, Stockfish::Search::Skill::LowestElo,
                       Stockfish::Search::Skill::HighestElo));

    options.add("UCI_ShowWDL", Option(false));

    threads.clear();
    threads.ensure_network_replicated();
    resize_threads();
}

Engine::~Engine() {
    stop();
    wait_for_search_finished();
}

std::variant<u64, PositionSetError>
Engine::perft(const std::string& fen, Depth depth, bool isChess960) {
    wait_for_search_finished();
    return Benchmark::perft(fen, depth, isChess960);
}

void Engine::go(Search::LimitsType& limits) {
    assert(limits.perft == 0);

    wait_for_search_finished();
    aliceSearchStop.store(false, std::memory_order_relaxed);
    alicePondering.store(limits.ponderMode, std::memory_order_relaxed);
    threads.stop = false;

    std::vector<Move> legalMoves;
    for (Move move : MoveList<LEGAL>(pos))
        legalMoves.push_back(move);

    std::vector<Move> rootMoves;
    for (const std::string& moveText : limits.searchmoves)
    {
        const Move move = UCIEngine::to_move(pos, moveText);
        if (move != Move::none()
            && std::find(rootMoves.begin(), rootMoves.end(), move) == rootMoves.end())
            rootMoves.push_back(move);
    }

    // Match the established UCI behavior: an empty or wholly invalid
    // searchmoves list falls back to the complete legal root set.
    if (rootMoves.empty())
        rootMoves = legalMoves;

    AliceSearch::Limits aliceLimits;
    if (limits.depth > 0)
        aliceLimits.depth = std::clamp(limits.depth, 1, MAX_PLY);
    else if (limits.mate > 0)
        aliceLimits.depth = std::clamp(2 * limits.mate, 1, MAX_PLY);
    else if (limits.infinite || limits.ponderMode || limits.nodes || limits.movetime
             || limits.use_time_management())
        aliceLimits.depth = MAX_PLY;
    else
        aliceLimits.depth = 5;

    aliceLimits.nodes = limits.nodes;

    if (!limits.ponderMode)
    {
        const TimePoint overhead = TimePoint(options["Move Overhead"]);
        TimePoint       budget   = 0;

        if (limits.movetime > 0)
            budget = std::max(TimePoint(1), limits.movetime - overhead);
        else if (limits.use_time_management())
        {
            const Color     us        = pos.side_to_move();
            const TimePoint remaining = limits.time[us];
            const int       moves     = limits.movestogo > 0 ? limits.movestogo : 30;
            const TimePoint share     = remaining / moves + 3 * limits.inc[us] / 4;
            budget = std::clamp(share - overhead, TimePoint(1), std::max(TimePoint(1), remaining));
        }

        if (budget > 0)
            aliceLimits.deadline = now() + budget;
    }

    assert(states);
    const std::string rootFen     = pos.fen();
    const StateInfo   rootState   = states->back();
    const bool        isChess960  = pos.is_chess960();
    const bool        waitForStop = limits.infinite;
    aliceSearchThread = std::thread([this, rootMoves = std::move(rootMoves), aliceLimits, rootFen,
                                     rootState, isChess960, waitForStop]() mutable {
        StateInfo  searchRootState;
        Position   searchPos;
        const auto error = searchPos.set(rootFen, isChess960, &searchRootState);
        assert(!error.has_value());
        (void) error;
        searchRootState = rootState;

        const TimePoint started = now();

        if (updateContext.onStart)
            updateContext.onStart();

        const auto onIteration = [this, started, &searchPos](const AliceSearch::Result& result) {
            std::string pv;
            for (Move move : result.pv)
            {
                if (!pv.empty())
                    pv += ' ';
                pv += UCIEngine::move(move, searchPos.is_chess960());
            }

            const std::string wdl     = result.score > VALUE_DRAW ? "1000 0 0"
                                      : result.score < VALUE_DRAW ? "0 0 1000"
                                                                  : "0 1000 0";
            const TimePoint   elapsed = std::max(TimePoint(1), now() - started);

            InfoFull info;
            info.depth    = result.depth;
            info.selDepth = result.depth;
            info.multiPV  = 1;
            info.score    = Score(result.score, searchPos);
            info.wdl      = wdl;
            info.bound    = "";
            info.timeMs   = usize(elapsed);
            info.nodes    = usize(result.nodes);
            info.nps      = usize(result.nodes * 1000 / u64(elapsed));
            info.tbHits   = 0;
            info.pv       = pv;
            info.hashfull = 0;

            if (updateContext.onUpdateFull)
                updateContext.onUpdateFull(info);
        };

        const AliceSearch::Result result =
          AliceSearch::search(searchPos, rootMoves, aliceLimits, aliceSearchStop, onIteration);

        if (rootMoves.empty() && updateContext.onUpdateNoMoves)
            updateContext.onUpdateNoMoves({0, Score(result.score, searchPos)});

        while (!aliceSearchStop.load(std::memory_order_relaxed)
               && (waitForStop || alicePondering.load(std::memory_order_relaxed)))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        std::string bestmove = UCIEngine::move(result.bestMove, searchPos.is_chess960());
        std::string ponder;
        if (result.pv.size() > 1)
            ponder = UCIEngine::move(result.pv[1], searchPos.is_chess960());

        if (updateContext.onBestmove)
            updateContext.onBestmove(bestmove, ponder);
    });
}
void Engine::stop() {
    aliceSearchStop.store(true, std::memory_order_relaxed);
    alicePondering.store(false, std::memory_order_relaxed);
    threads.stop = true;
}

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(threads);
    threads.clear();
}

void Engine::set_on_update_no_moves(std::function<void(const Engine::InfoShort&)>&& f) {
    updateContext.onUpdateNoMoves = std::move(f);
}

void Engine::set_on_update_full(std::function<void(const Engine::InfoFull&)>&& f) {
    updateContext.onUpdateFull = std::move(f);
}

void Engine::set_on_iter(std::function<void(const Engine::InfoIter&)>&& f) {
    updateContext.onIter = std::move(f);
}

void Engine::set_on_bestmove(std::function<void(std::string_view, std::string_view)>&& f) {
    updateContext.onBestmove = std::move(f);
}

void Engine::set_on_start(std::function<void()>&& f) { updateContext.onStart = std::move(f); }

void Engine::set_on_verify_network(std::function<void(std::string_view)>&& f) {
    onVerifyNetwork = std::move(f);
}

void Engine::wait_for_search_finished() {
    if (aliceSearchThread.joinable())
        aliceSearchThread.join();
    threads.main_thread()->wait_for_search_finished();
}

std::optional<PositionSetError> Engine::set_position(const std::string&              fen,
                                                     const std::vector<std::string>& moves) {
    wait_for_search_finished();

    // Validate the complete command on an isolated position. A bad FEN or a bad
    // move therefore leaves the current game and every StateInfo pointer intact.
    Position     candidate;
    StateListPtr candidateStates(new std::deque<StateInfo>(1));
    auto         err = candidate.set(fen, options["UCI_Chess960"], &candidateStates->back());
    if (err.has_value())
        return err;

    std::vector<Move> resolvedMoves;
    resolvedMoves.reserve(moves.size());
    for (const auto& move : moves)
    {
        const Move resolved = UCIEngine::to_move(candidate, move);

        if (resolved == Move::none())
            return PositionSetError("Illegal move: " + move);

        resolvedMoves.push_back(resolved);
        candidateStates->emplace_back();
        candidate.do_move(resolved, candidateStates->back());
    }

    StateListPtr newStates(new std::deque<StateInfo>(1));
    err = pos.set(fen, options["UCI_Chess960"], &newStates->back());
    assert(!err.has_value());
    for (Move move : resolvedMoves)
    {
        newStates->emplace_back();
        pos.do_move(move, newStates->back());
    }
    states = std::move(newStates);

    return std::nullopt;
}

// modifiers

bool Engine::set_numa_config_from_option(const std::string& o) {
    if (o == "auto" || o == "system")
    {
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy));
    }
    else if (o == "hardware")
    {
        // Don't respect affinity set in the system.
        numaContext.set_numa_config(NumaConfig::from_system(DefaultNumaPolicy, false));
    }
    else if (o == "none")
    {
        numaContext.set_numa_config(NumaConfig{});
    }
    else
    {
        auto parsed = NumaConfig::from_string(o);
        if (!parsed.has_value())
            return false;
        numaContext.set_numa_config(std::move(*parsed));
    }

    // Force reallocation of threads in case affinities need to change.
    resize_threads();
    threads.ensure_network_replicated();
    return true;
}

void Engine::resize_threads() {
    threads.wait_for_search_finished();
    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists, network},
                updateContext);

    // Reallocate the hash with the new threadpool size
    set_tt_size(options["Hash"]);
    threads.ensure_network_replicated();
}

void Engine::set_tt_size(usize mb) {
    wait_for_search_finished();
    tt.resize(mb, threads);
}

void Engine::set_ponderhit(bool b) {
    if (!b && alicePondering.exchange(false, std::memory_order_relaxed))
        aliceSearchStop.store(true, std::memory_order_relaxed);
    threads.main_manager()->ponder = b;
}

// network related

void Engine::verify_network() const {
    if (onVerifyNetwork)
        onVerifyNetwork("No Alice evaluator is loaded; deterministic zero evaluation is active.");
}

std::unique_ptr<Eval::NNUE::Network> Engine::get_default_network() {

    auto network_ = std::make_unique<NN::Network>();

    network_->load(binaryDirectory, std::filesystem::path{}, networkFile);

    return network_;
}

void Engine::load_network(const std::filesystem::path& file) {
    network.modify_and_replicate(
      [this, &file](NN::Network& network_) { network_.load(binaryDirectory, file, networkFile); });
    threads.clear();
    threads.ensure_network_replicated();
}

void Engine::save_network(const std::optional<std::filesystem::path>& file) {
    network.modify_and_replicate(
      [&file, this](NN::Network& network_) { network_.save(networkFile, file); });
}

// utility functions

void Engine::trace_eval() const {
    sync_cout << "info string Evaluation is unavailable until a compatible Alice network is loaded."
              << sync_endl;
}

const OptionsMap& Engine::get_options() const { return options; }
OptionsMap&       Engine::get_options() { return options; }

std::string Engine::fen() const { return pos.fen(); }

std::optional<PositionSetError> Engine::flip() {
    wait_for_search_finished();
    return pos.flip();
}

std::string Engine::visualize() const {
    std::stringstream ss;
    ss << pos;
    return ss.str();
}

int Engine::get_hashfull(int maxAge) const { return tt.hashfull(maxAge); }

std::vector<std::pair<usize, usize>> Engine::get_bound_thread_count_by_numa_node() const {
    auto                                 counts = threads.get_bound_thread_count_by_numa_node();
    const NumaConfig&                    cfg    = numaContext.get_numa_config();
    std::vector<std::pair<usize, usize>> ratios;
    NumaIndex                            n = 0;
    for (; n < counts.size(); ++n)
        ratios.emplace_back(counts[n], cfg.num_cpus_in_numa_node(n));
    if (!counts.empty())
        for (; n < cfg.num_numa_nodes(); ++n)
            ratios.emplace_back(0, cfg.num_cpus_in_numa_node(n));
    return ratios;
}

std::string Engine::get_numa_config_as_string() const {
    return numaContext.get_numa_config().to_string();
}

std::string Engine::numa_config_information_as_string() const {
    auto cfgStr = get_numa_config_as_string();
    return "Available processors: " + cfgStr;
}

std::string Engine::thread_binding_information_as_string() const {
    auto              boundThreadsByNode = get_bound_thread_count_by_numa_node();
    std::stringstream ss;
    if (boundThreadsByNode.empty())
        return ss.str();

    bool isFirst = true;

    for (auto&& [current, total] : boundThreadsByNode)
    {
        if (!isFirst)
            ss << ":";
        ss << current << "/" << total;
        isFirst = false;
    }

    return ss.str();
}

std::string Engine::thread_allocation_information_as_string() const {
    std::stringstream ss;

    usize threadsSize = threads.size();
    ss << "Using " << threadsSize << (threadsSize > 1 ? " threads" : " thread");

    auto boundThreadsByNodeStr = thread_binding_information_as_string();
    if (boundThreadsByNodeStr.empty())
        return ss.str();

    ss << " with NUMA node thread binding: ";
    ss << boundThreadsByNodeStr;

    return ss.str();
}
}
