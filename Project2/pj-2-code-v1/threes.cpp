/**
 * Framework for Threes! and its variants (C++ 11)
 * threes.cpp: Main file for Threes!
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "board.h"
#include "action.h"
#include "agent.h"
#include "episode.h"
#include "statistics.h"

int main(int argc, const char* argv[]) {
	std::cout << "Threes! Demo: ";
	std::copy(argv, argv + argc, std::ostream_iterator<const char*>(std::cout, " "));
	std::cout << std::endl << std::endl;

	size_t total = 1000, block = 0, limit = 0;
	std::string slide_args, place_args;
	std::string load_path, save_path;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		auto match_arg = [&](std::string flag) -> bool {
			auto it = arg.find_first_not_of('-');
			return arg.find(flag, it) == it;
		};
		auto next_opt = [&]() -> std::string {
			auto it = arg.find('=') + 1;
			return it ? arg.substr(it) : argv[++i];
		};
		if (match_arg("total")) {
			total = std::stoull(next_opt());
		} else if (match_arg("block")) {
			block = std::stoull(next_opt());
		} else if (match_arg("limit")) {
			limit = std::stoull(next_opt());
		} else if (match_arg("slide") || match_arg("play")) {
			slide_args = next_opt();
		} else if (match_arg("place") || match_arg("env")) {
			place_args = next_opt();
		} else if (match_arg("load")) {
			load_path = next_opt();
		} else if (match_arg("save")) {
			save_path = next_opt();
		}
	}

	statistics stats(total, block, limit);

	if (load_path.size()) {
		std::ifstream in(load_path, std::ios::in);
		in >> stats;
		in.close();
		if (stats.is_finished()) stats.summary();
	}

	//random_slider slide(slide_args);
	random_placer place(place_args);
	// greedy_slider slide(slide_args);
	learning_slider slide(slide_args);
	std::vector<state> path;

	while (!stats.is_finished()) {
//		std::cerr << "======== Game " << stats.step() << " ========" << std::endl;
		slide.open_episode("~:" + place.name());
		place.open_episode(slide.name() + ":~");

		stats.open_episode(slide.name() + ":" + place.name());
		episode& game = stats.back();
		while (true) {
			float state_value = 0.0;
			int reward = 0;
			state s1;
			s1.board_before = game.state();
			agent& who = game.take_turns(slide, place);
			action move = who.take_action(game.state(), state_value, reward);
//			std::cerr << game.state() << "#" << game.step() << " " << who.name() << ": " << move << std::endl;
			if (game.apply_action(move) != true) break;
			s1.board_after = game.state();
			s1.reward = reward;
			s1.value = state_value;

			// player's move
			if (reward != 0 || state_value != 0) {
				path.push_back(s1);
			}

			if (who.check_for_win(game.state())) break;
		}
		agent& win = game.last_turns(slide, place);
		stats.close_episode(win.name());

		slide.update(path);
    	path.clear();
		slide.close_episode(win.name());
		place.close_episode(win.name());
	}

	if (save_path.size()) {
		std::ofstream out(save_path, std::ios::out | std::ios::trunc);
		out << stats;
		out.close();
	}
	return 0;
}
